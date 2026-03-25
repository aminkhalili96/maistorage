# Lustre Parallel File System for AI and HPC Storage

Lustre is the most widely deployed parallel file system in high-performance computing, powering the majority of Top500 supercomputers and the largest AI training clusters. It scales to exabyte capacities and terabytes-per-second aggregate bandwidth, with separated metadata and data paths enabling massive parallelism for I/O-intensive distributed GPU training.

## Architecture

Lustre uses a client-server architecture with distinct roles:

- **MGS (Management Server)** — stores cluster-wide configuration. Clients contact the MGS at mount time to discover filesystem topology. Typically colocated with the primary MDS.
- **MDS / MDT (Metadata Server / Target)** — handles namespace operations: create, open, stat, unlink, readdir. The MDT is typically an NVMe-backed ZFS or ldiskfs volume.
- **OSS / OST (Object Storage Server / Target)** — stores file data as objects. Each OSS manages multiple OSTs. File data is striped across OSTs for parallel access.
- **Clients** — mount via the kernel module (`lustre.ko`), communicating directly with MDSes for metadata and OSSes for data.

## Striping Configuration

Lustre stripes files across OSTs using `lfs setstripe`:

```bash
lfs setstripe -c 8 -S 4M /mnt/lustre/training-data
lfs getstripe /mnt/lustre/training-data/dataset.tar
```

- **Stripe count (`-c`)** — number of OSTs per file. `-c -1` stripes across all OSTs for maximum bandwidth.
- **Stripe size (`-S`)** — contiguous bytes per OST before rotating. Default 1 MB; use 4-16 MB for checkpoints to reduce lock contention.

**Progressive File Layout (PFL)** dynamically adjusts striping as files grow:

```bash
lfs setstripe -E 64K -c 1 -S 64K -E 1G -c 4 -S 1M -E -1 -c -1 -S 4M /mnt/lustre/mixed
```

## DNE (Distributed Namespace)

DNE spreads metadata across multiple MDTs. DNE1 assigns subdirectories to specific MDTs (`lfs mkdir -i <index>`). DNE2 hash-distributes a single directory's entries across MDTs (`lfs mkdir -c 4 /mnt/lustre/high-iops-dir`), parallelizing readdir, create, and stat for directories with millions of files. Essential for ImageNet-scale datasets.

## File Locking (LDLM)

Lustre's Distributed Lock Manager coordinates concurrent access via extent locks and inode locks. AI workload contention patterns include checkpoint storms (mitigate with per-rank directories or `lfs mkdir -c`) and shared-file stat contention (tune with `lctl set_param llite.*.statahead_max=128`).

## Performance Tuning

Critical parameters for AI workloads:

- **RPC size** — `lctl set_param osc.*.max_pages_per_rpc=4096` (16 MB with 4K pages)
- **OST threads** — `lctl set_param ost.OSS.ost_io.threads_max=512`
- **Read-ahead** — `lctl set_param llite.*.max_read_ahead_mb=256` for sequential loading
- **Stripe alignment** — align I/O to stripe boundaries; PyTorch DataLoader buffers should be multiples of stripe size
- **Client cache** — `lctl set_param llite.*.max_cached_mb=2048`; increase for re-read workloads, reduce for streaming

## Lustre and GPUDirect Storage

GPUDirect Storage enables direct transfers between OSTs and GPU memory, bypassing CPU page cache and DMA bounce buffers. Reduces checkpoint latency by up to 3x. Requires `nvidia-fs` kernel module, Lustre 2.15+ with GDS patches, and the `cuFile` API (DALI, KvikIO, Magnum IO).

## Checkpoint I/O Patterns

A 175B parameter model produces ~350 GB per checkpoint. Lustre strategies: stripe across all OSTs (`-c -1 -S 8M`), stagger write timing across ranks, use `O_DIRECT` to bypass page cache, write per-rank shard files to avoid lock contention, and overlap checkpoint I/O with training via asynchronous writes.

## Data Loading Challenges

Small-file random reads stress metadata. Mitigations: tar-based datasets (WebDataset, FFCV, TFRecord) converting millions of reads into sequential I/O, DNE2 striped directories for high-file-count directories, and local NVMe caching to read from Lustre only once per epoch.

## HSM (Hierarchical Storage Management)

Lustre HSM tiers data to external storage (tape, object, cloud). Archived files leave stubs on Lustre; access triggers transparent restore. Useful for retaining old checkpoints without consuming NVMe capacity. Managed via `lhsmtool_posix` or `lhsmtool_s3` copytool agents.

## Lustre Networking (LNet)

LNet supports TCP (`tcp`), InfiniBand RDMA (`o2ib`), multi-rail for bandwidth aggregation (`lctl set_param lnet.networks="o2ib0(ib0),o2ib1(ib1)"`), and routing to bridge heterogeneous networks.

## Monitoring

- **`lctl get_param`** — runtime parameters and statistics (`osc.*.stats`, `llite.*.stats`)
- **Lustre Jobstats** — per-job I/O accounting: `lctl set_param jobid_var=procname_uid`
- **`lfs df`** — per-OST and per-MDT space usage
- **Health checks** — `lctl get_param health_check` on clients and servers

## Scale and Deployments

The largest deployments exceed 1 exabyte and 1 TB/s aggregate bandwidth (Frontier, El Capitan). Commercial appliances from DDN (EXAScaler), HPE (ClusterStor), and NetApp provide integrated hardware, distribution, and support. DDN AI400X appliances deliver 90 GB/s per unit with NVMe-backed OSTs and GPUDirect Storage support.

## GPU Node Client Configuration

Lustre clients on GPU compute nodes require specific tuning for AI workloads. Key client-side parameters:

**Read-ahead tuning** — GPU data loaders typically read large sequential files (datasets, model weights). Increase read-ahead for sequential patterns:

```bash
lctl set_param llite.*.max_read_ahead_mb=512
lctl set_param llite.*.max_read_ahead_whole_mb=64
```

**Client cache sizing** — GPU nodes typically have 256-1024 GB system RAM. Allocate a significant portion to Lustre client cache for re-read workloads (multi-epoch training):

```bash
lctl set_param llite.*.max_cached_mb=8192    # 8 GB for nodes with 512+ GB RAM
```

For streaming workloads (one-pass data loading), reduce cache to avoid eviction overhead:

```bash
lctl set_param llite.*.max_cached_mb=512
```

**Statahead** — pre-fetches metadata for directory listings, critical for readdir-heavy data loading:

```bash
lctl set_param llite.*.statahead_max=256
lctl set_param llite.*.statahead_agl=1        # async glimpse lock for size info
```

**Write-back cache** — for checkpoint writes, enable write-back caching to buffer small writes:

```bash
lctl set_param llite.*.max_dirty_mb=256
```

**NUMA awareness** — on multi-socket servers, pin the Lustre client's LNet threads to the NUMA node closest to the InfiniBand HCA for lower-latency I/O:

```bash
lctl set_param lnet.numa_range=0              # strict NUMA binding
```

## Progressive File Layout (PFL) for AI

PFL dynamically adjusts stripe parameters as files grow, which is ideal for AI workloads where file sizes vary dramatically:

```bash
# Small files (metadata, configs) stay on 1 OST; medium files stripe across 4;
# large files (checkpoints, datasets) stripe across all OSTs
lfs setstripe -E 1M -c 1 -S 256K \
              -E 1G -c 4 -S 1M \
              -E -1 -c -1 -S 4M \
              /mnt/lustre/training/
```

This eliminates the need for separate directories with different stripe configs. Small configuration files don't waste OST capacity; large checkpoint files automatically stripe for maximum bandwidth.

## DNE for Metadata Scaling

For datasets with millions of small files (ImageNet: 14M images, Common Crawl: billions of documents), metadata operations become the bottleneck. DNE2 striped directories distribute metadata across MDTs:

```bash
# Stripe the dataset directory across 4 MDTs
lfs mkdir -c 4 /mnt/lustre/datasets/imagenet/train

# Verify metadata distribution
lfs getdirstripe /mnt/lustre/datasets/imagenet/train
```

DNE2 parallelizes `readdir`, `stat`, and `create` operations across MDTs. A 4-MDT configuration provides roughly 3.5x the metadata throughput of a single MDT (scaling is near-linear with slight coordination overhead).

For AI training pipelines that generate many output files (per-rank checkpoints, per-step logs), create output directories with DNE striping before the job starts to avoid metadata hotspots.

## Monitoring with Lustre Jobstats

Jobstats provides per-job I/O accounting, essential for understanding which training jobs are consuming storage bandwidth:

```bash
# Enable jobstats (set on OSS and MDS)
lctl set_param jobid_var=procname_uid

# View per-job statistics
lctl get_param obdfilter.*.job_stats

# Reset statistics
lctl set_param obdfilter.*.job_stats=clear
```

Jobstats output includes read/write bytes, operations, and latency histograms per job. Integrate with monitoring:

- **Prometheus**: `lustre_exporter` scrapes `lctl get_param` and exports as Prometheus metrics. Key metrics: `lustre_ost_read_bytes_total`, `lustre_ost_write_bytes_total`, `lustre_mdt_open_total`, `lustre_mdt_mkdir_total`.
- **Grafana dashboards**: Per-OST throughput heatmap, MDS operations/second, client-side cache hit rate, per-job bandwidth allocation.
- **Capacity alerts**: Monitor `lfs df` output for OST imbalance (>20% variance indicates striping misconfiguration) and approaching capacity limits (trigger cleanup of old checkpoints at 80% full).

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

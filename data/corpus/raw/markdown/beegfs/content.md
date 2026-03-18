# BeeGFS Parallel File System for AI Storage

BeeGFS (formerly FhGFS) is a parallel file system designed for high throughput and ease of deployment. Originally developed at the Fraunhofer Institute, BeeGFS is widely used in AI training clusters where its simplicity, linear bandwidth scaling, and native RDMA support make it a practical alternative to Lustre for medium-to-large scale GPU deployments.

## Architecture

BeeGFS uses a distributed architecture with four service roles:

- **Management server (`beegfs-mgmtd`)** — central registry tracking all services. Lightweight; handles no data or metadata I/O. Config: `/etc/beegfs/beegfs-mgmtd.conf`.
- **Metadata servers (`beegfs-meta`)** — handle file and directory metadata operations (create, stat, open, readdir). Metadata is hash-distributed across servers, each storing data on local ext4 over NVMe. Config: `/etc/beegfs/beegfs-meta.conf`.
- **Storage servers (`beegfs-storage`)** — store file data in chunks across one or more storage targets (individual disks or RAID groups). Config: `/etc/beegfs/beegfs-storage.conf`.
- **Clients (`beegfs-client`)** — a kernel module (`beegfs.ko`) and helper daemon (`beegfs-helperd`) that mount the filesystem and communicate directly with metadata and storage servers. Config: `/etc/beegfs/beegfs-client.conf`.

## Striping

BeeGFS stripes files across multiple storage targets for parallel I/O. Key parameters:

- **Stripe count (`tuneNumTargets`)** — how many targets each file spans. Higher counts yield higher aggregate bandwidth for large files.
- **Chunk size (`tuneChunkSize`)** — contiguous bytes per target before rotating. Default 512 KB; increase to 1-2 MB for AI checkpoint I/O to reduce metadata overhead.

Per-directory stripe settings:

```bash
beegfs-ctl --setpattern --numtargets=8 --chunksize=1M /mnt/beegfs/training-data
```

For data loading with many small random reads, lower stripe counts (1-2) reduce network round-trips. Large sequential checkpoint writes benefit from striping across all targets.

## Buddy Mirroring

BeeGFS provides redundancy through buddy mirroring, replicating data across pairs of storage targets (buddy groups) on different physical servers. When a target fails, its buddy serves the data automatically.

```bash
beegfs-ctl --setpattern --buddymirror /mnt/beegfs/checkpoints
```

Metadata mirroring pairs metadata servers similarly, with automatic failover. Mirroring halves effective capacity and adds write latency, so many AI clusters mirror only checkpoint directories while leaving training data unmirrored (reloadable from object storage).

## Performance Characteristics

BeeGFS scales aggregate bandwidth linearly with storage servers. Typical per-server performance with 10x NVMe drives: 20-25 GB/s sequential read, 15-20 GB/s sequential write, 100K-500K metadata ops/sec. A 16-server deployment delivers 300+ GB/s aggregate read bandwidth. BeeGFS metadata performance generally exceeds Lustre for small-file workloads due to its lightweight protocol.

## AI Workload Tuning

Key tuning for AI training:

- **Checkpoint I/O** — `tuneChunkSize=1M` or higher, stripe across all targets, use `O_DIRECT` to bypass page cache
- **Data loading** — increase `connMaxInternodeNum` for more parallel connections; use tar-based datasets (WebDataset) to convert small-file reads into sequential reads
- **Client caching** — `tuneFileCacheType=buffered` for read caching; `tuneRemoteFSync=false` to skip unnecessary fsync when the training framework handles durability
- **Network** — `connRDMAEnabled=true` with empty `connTcpOnlyFilterFile` to force RDMA on all connections

## RDMA Integration

BeeGFS natively supports RDMA over InfiniBand and RoCE (`connRDMAEnabled=true`). All data transfers use RDMA verbs, bypassing kernel TCP/IP. This is critical for line-rate bandwidth on HDR (200 Gb/s) and NDR (400 Gb/s) fabrics.

For GPUDirect Storage, BeeGFS integrates with NVIDIA Magnum IO and the `cuFile` API, allowing GPU memory to read/write directly from storage targets without CPU bounce buffers. Requires `nvidia-fs` kernel module.

## BeeGFS On-Demand

BeeGFS On-Demand creates ephemeral instances within a job scheduler. A training job spins up a dedicated BeeGFS instance on local NVMe of allocated nodes, uses it as scratch storage, and tears it down on completion. This eliminates contention on the shared filesystem. Configured via Slurm burst buffer plugins or Kubernetes CSI drivers.

## Monitoring

- **`beegfs-ctl --listnodes --nodetype=storage`** — list storage servers and reachability
- **`beegfs-ctl --storagebench`** — built-in throughput benchmark
- **`beegfs-ctl --getquota`** — per-user/group quota checks
- **Admon GUI** — real-time throughput, IOPS, and node health dashboard
- **Prometheus** — BeeGFS 7.3+ exports metrics via HTTP for Grafana

## BeeGFS vs Lustre

BeeGFS is significantly easier to deploy: no kernel patching (DKMS client module), simpler configuration, and rolling upgrade support. Lustre has deeper maturity at extreme scale (exabyte deployments) and tighter HPC ecosystem integration (DDN, HPE appliances). For AI clusters up to ~100 storage servers, BeeGFS offers comparable performance with lower operational complexity.

## Common Deployment Pattern

A typical AI cluster uses NVMe-backed storage servers (8-12 SSDs per server), ext4 or XFS on each target, InfiniBand HDR/NDR networking, and 2-4 metadata servers on separate NVMe nodes. Training data lives on a high-stripe-count directory, checkpoints go to a buddy-mirrored directory, and compute nodes mount BeeGFS via the kernel client with RDMA enabled.

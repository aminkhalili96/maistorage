# Data Path and Storage Architecture — Technical Reference

Storage architecture for AI infrastructure is organized in tiers that balance speed, capacity, cost, and durability. Getting the data path right is critical because GPU clusters are expensive to idle — every minute a GPU waits for data is wasted compute budget.

## Storage Tiers

Tier 0 is local NVMe storage directly attached to compute nodes. It provides the lowest latency and highest per-node bandwidth (typically 6-14 GB/s per NVMe drive, with multiple drives per node). Local NVMe serves as scratch space for intermediate results, shuffled dataset shards, and GPU-accessible storage via GPUDirect Storage. Local storage is ephemeral — data is lost if the node is re-imaged or fails.

Tier 1 is the parallel filesystem layer. Lustre, BeeGFS, and IBM Spectrum Scale (GPFS) provide a shared POSIX-compliant namespace accessible from every compute node simultaneously. These systems stripe data across multiple object storage targets (OSTs in Lustre, storage targets in BeeGFS) to deliver aggregate read bandwidth that scales with the number of storage servers. A typical Lustre deployment for AI workloads might use 4-8 OSS nodes with NVMe-backed OSTs to deliver 50-100 GB/s aggregate read throughput. BeeGFS is popular for its simpler deployment and metadata scalability with multiple metadata targets.

Tier 2 is S3-compatible object storage, commonly implemented with MinIO for on-premises deployments. Object storage handles datasets, model checkpoints, training artifacts, and experiment logs. It scales horizontally, supports versioning and lifecycle policies, and integrates natively with frameworks and MLOps tools. Object storage trades latency and POSIX semantics for durability, scalability, and cost efficiency.

Tier 3 is archive and cold storage — tape libraries, cold S3 tiers, or offline storage for compliance, data retention, and long-term experiment reproducibility.

## Data Loading Pipeline

The production data path for training workloads typically flows: object store (durable source of truth) to parallel filesystem (staging area with high-bandwidth shared access) to local NVMe cache (node-local fast access) to GPU memory (via DMA). PyTorch DataLoader with multiple workers, NVIDIA DALI, or custom data pipelines orchestrate this flow. Prefetching and asynchronous loading are essential — the data pipeline must stay ahead of GPU computation to avoid stalls. Sharded datasets (WebDataset, TFRecord, Mosaic StreamingDataset) reduce metadata overhead and small-file problems that plague parallel filesystems.

## Bandwidth Planning

Storage bandwidth requirements are derived from GPU throughput. If 8 H100 GPUs can process training samples at a rate that requires 10 GB/s of data ingestion, the storage system must sustain at least 10 GB/s of read throughput to that node — and more during data loading bursts. For a 64-node cluster, the aggregate storage bandwidth requirement could reach 640 GB/s or more. Bandwidth planning must account for read amplification from data augmentation, re-reads during multiple epochs, and contention from checkpoint writes happening concurrently with data reads.

## GPUDirect Storage

GPUDirect Storage (GDS) enables direct DMA transfers between NVMe storage and GPU memory, bypassing the CPU and host memory entirely. This eliminates a memory copy and reduces CPU overhead during I/O. GDS is supported on local NVMe, some NVMe-oF (NVMe over Fabrics) configurations, and compatible parallel filesystems (Lustre with GDS-enabled client, VAST Data). Performance gains are most significant for workloads with large sequential reads — GDS can deliver 2-3x the effective storage bandwidth compared to the traditional CPU-bounce-buffer path. GDS requires the `nvidia-fs` kernel module and a compatible storage driver.

## POSIX vs Object Semantics

Training data loading almost universally requires POSIX semantics — random access to files, directory listing, file locking — because data loaders open files by path and seek within them. Parallel filesystems satisfy this requirement. Checkpointing, model artifact storage, and dataset versioning work well with object storage semantics (PUT/GET, immutable objects, versioning). The common pattern is to use POSIX for hot training data and S3 for everything else.

## Checkpoint Strategies

Checkpointing saves model state (weights, optimizer state, learning rate schedule, RNG state) to durable storage at regular intervals. Synchronous checkpointing pauses training, writes the checkpoint, and resumes — simple but creates a throughput gap proportional to checkpoint size divided by storage write bandwidth. Asynchronous checkpointing overlaps the write with continued training by staging the checkpoint in host memory or local NVMe first, then flushing to durable storage in the background. PyTorch distributed checkpointing (`torch.distributed.checkpoint`) writes sharded checkpoints in parallel from each rank, reducing wall-clock checkpoint time. Checkpoint frequency is a tradeoff: too frequent wastes bandwidth, too infrequent risks losing hours of training progress on failure. A common starting point is every 15-30 minutes for large training runs.

## Network Storage Connectivity

InfiniBand is preferred for storage network traffic in GPU clusters because it provides low-latency RDMA (Remote Direct Memory Access) that parallel filesystems exploit for high-throughput data transfer. Lustre's `o2ib` LND (Lustre Network Driver) runs natively over InfiniBand RDMA. NFS over RDMA (NFSoRDMA) is available for simpler shared storage needs. Ethernet-based storage (25-100 GbE with RoCE) is a lower-cost alternative but introduces higher tail latency under contention.

## Common Bottlenecks

The most frequent storage bottlenecks in AI clusters are metadata operations (listing directories with millions of small files saturates metadata servers), client-side caching misconfiguration (insufficient Lustre `llite.*.max_cached_mb`), network congestion when storage and compute traffic share the same fabric, and write contention when multiple jobs checkpoint simultaneously. Capacity planning must account for dataset size, checkpoint frequency, checkpoint retention policy, and experiment artifact accumulation — storage fills faster than expected in active research environments.

# NVMe over Fabrics (NVMe-oF) for AI Storage

NVMe over Fabrics (NVMe-oF) extends the NVMe protocol across network fabrics including RDMA, TCP, and Fibre Channel, enabling remote NVMe storage access with near-local latency. For GPU clusters running large-scale AI training and inference workloads, NVMe-oF provides a critical building block for disaggregated storage architectures where compute and storage scale independently.

## Protocol Overview

NVMe-oF maps the full NVMe command set onto network transport protocols, preserving the parallelism and low-overhead design of NVMe while enabling access to storage devices across a fabric. Three primary transports are supported:

- **RDMA (InfiniBand/RoCE):** The lowest-latency transport, adding less than 10 microseconds compared to local NVMe. RDMA bypasses the kernel networking stack entirely, performing direct memory-to-memory transfers. InfiniBand RDMA is the preferred transport for GPU clusters already using InfiniBand for inter-GPU communication (NCCL). RoCE provides RDMA semantics over standard Ethernet with DCB for lossless operation.

- **TCP:** Works on standard Ethernet without specialized NICs or fabric configuration. Higher latency (30-80 microseconds additional) and CPU overhead compared to RDMA, but dramatically simpler deployment. Suitable for environments where storage latency is not the primary bottleneck.

- **Fibre Channel:** NVMe-oF over FC (FC-NVMe) leverages existing Fibre Channel SANs. Less common in GPU clusters but relevant for enterprises integrating AI infrastructure with legacy storage.

## Architecture

An NVMe-oF deployment consists of the **target** (storage server) and the **initiator** (compute node).

The target exposes local NVMe namespaces over the fabric using the `nvmet` kernel module. Each target can expose multiple NVMe subsystems, each containing one or more namespaces backed by physical NVMe drives or logical volumes.

The initiator discovers available targets using `nvme discover -t <transport> -a <target-address>` and connects with `nvme connect -t <transport> -n <subsystem-nqn> -a <target-address>`. Once connected, remote NVMe namespaces appear as local block devices (`/dev/nvmeXnY`) and can be used with any filesystem or accessed via raw block I/O.

## GPUDirect Storage Integration

NVMe-oF combined with NVIDIA GPUDirect Storage (GDS) enables direct DMA transfers from remote NVMe to GPU memory, bypassing the CPU and system memory entirely. This eliminates both the CPU bounce buffer and page cache overhead.

The `cuFile` API reads and writes remote NVMe-oF volumes as if local. The `nvidia-fs` kernel module orchestrates RDMA transfers from the NVMe-oF target directly into GPU HBM.

Requirements for NVMe-oF with GPUDirect Storage:
- MLNX_OFED driver stack (for RDMA transport support)
- `nvidia-fs` kernel module (GDS kernel component)
- RDMA transport (TCP does not support GDS direct path)
- ConnectX-6 or later network adapters
- GDS-compatible NVMe-oF target (SPDK-based targets provide best performance)

## Performance Characteristics

Typical performance for enterprise NVMe drives (PCIe Gen4 x4):

| Configuration | Throughput per Drive | IOPS (4K random) |
|---------------|---------------------|-------------------|
| Local NVMe | ~3.5 GB/s | ~500,000 |
| NVMe-oF RDMA | ~3.2 GB/s (8-10% overhead) | ~450,000 |
| NVMe-oF TCP | ~2.5 GB/s | ~300,000 |

Aggregate bandwidth scales linearly with drive count. A 10-drive JBOF (Just a Bunch of Flash) over RDMA delivers approximately 30 GB/s sustained read bandwidth per initiator.

## Deployment Patterns

**Composable infrastructure:** Disaggregate NVMe storage pools from GPU compute nodes. Storage resources are dynamically allocated to GPU nodes based on workload requirements, preventing stranded capacity.

**Shared scratch storage:** Multiple GPU nodes access a common NVMe pool for training data staging. Data is loaded from a parallel filesystem (Lustre, BeeGFS) into the NVMe pool once, then served to GPU nodes at NVMe speeds.

**Checkpoint storage:** Dedicated NVMe-oF targets for high-throughput checkpoint writes. Checkpointing large models (70B+ parameters) generates hundreds of gigabytes per checkpoint. NVMe-oF targets with multiple drives provide sustained write bandwidth to minimize checkpoint overhead.

## AI Cluster Use Cases

**Data loading:** NVMe-oF replaces or supplements parallel filesystems for low-latency small-file reads. Training workloads with many small files benefit from NVMe-oF's lower metadata overhead. NVIDIA DALI supports `cuFile` for GDS-accelerated reads from NVMe-oF volumes.

**Checkpoint I/O:** Dedicated NVMe-oF targets provide consistent write bandwidth without shared filesystem contention. For distributed checkpointing with PyTorch FSDP or DeepSpeed, each GPU node writes to its own NVMe-oF namespace in parallel.

**Model loading:** Fast model weight loading for inference cold starts. Large model weights (tens to hundreds of gigabytes) load from NVMe-oF in seconds rather than minutes, reducing inference pod startup times in Kubernetes environments. Particularly valuable for autoscaling inference deployments.

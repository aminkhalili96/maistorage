# NCCL Developer Guide — Multi-GPU Communication Library

## Overview

NVIDIA Collective Communications Library (NCCL) implements multi-GPU and multi-node collective communication primitives optimized for NVIDIA GPUs and networking. NCCL provides routines such as AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, and point-to-point Send/Receive that are tuned for high bandwidth and low latency across PCIe, NVLink, NVSwitch, and InfiniBand interconnects. NCCL is the communication backbone for distributed deep learning frameworks including PyTorch Distributed Data Parallel (DDP), Horovod, and NVIDIA Megatron-Core.

## Collective Operations

NCCL supports the standard MPI-style collective operations that distributed training relies on. AllReduce is the most commonly used operation in data-parallel training — it sums gradients across all GPUs and distributes the result back to every GPU in a single operation. The implementation uses ring-based or tree-based algorithms depending on the topology and message size. ReduceScatter splits the reduction output across GPUs, which is essential for ZeRO-style optimizer partitioning.

Broadcast distributes data from one GPU to all others, commonly used for initial parameter synchronization. AllGather collects data from all GPUs into a single buffer on each GPU, used in tensor-parallel inference and model-parallel training. Point-to-point Send and Recv enable direct GPU-to-GPU transfers for pipeline parallelism stages.

## Topology Detection and Path Selection

NCCL automatically detects the interconnect topology at initialization time. It identifies NVLink connections between GPUs, NVSwitch presence in HGX/DGX systems, PCIe switch hierarchies, InfiniBand HCAs, and network interfaces. Based on this topology graph, NCCL selects optimal communication algorithms and channel configurations.

On systems with NVLink (e.g., DGX H100 with NVSwitch), NCCL routes traffic through NVLink for maximum bandwidth — 900 GB/s bidirectional per GPU on H100 systems. On PCIe-only systems, NCCL uses PCIe peer-to-peer transfers when supported, falling back to staged copies through host memory when P2P is unavailable. For multi-node communication, NCCL uses InfiniBand verbs or TCP/IP sockets, with support for GPUDirect RDMA to bypass CPU staging entirely.

## Environment Variables and Tuning

NCCL behavior is controlled through environment variables that are critical for performance tuning in production clusters.

`NCCL_DEBUG=INFO` enables diagnostic logging showing topology detection, algorithm selection, and transport choices. `NCCL_DEBUG=WARN` is recommended for production to log only potential issues.

`NCCL_ALGO` selects the communication algorithm: Ring, Tree, or CollNetDirect. Ring algorithms achieve maximum bandwidth utilization for large messages. Tree algorithms reduce latency for smaller messages. CollNetDirect uses NVIDIA SHARP for in-network reduction when available.

`NCCL_PROTO` selects the transport protocol: Simple, LL (Low Latency), or LL128. Simple protocol uses large buffers for maximum bandwidth. LL protocol reduces latency for small messages. LL128 is an optimized variant of LL for 128-byte transfers.

`NCCL_SOCKET_IFNAME` restricts which network interfaces NCCL uses for inter-node communication. Setting this to `eth0` or `ib0` prevents NCCL from using management or storage network interfaces.

`NCCL_IB_HCA` selects specific InfiniBand HCAs. On systems with multiple HCAs (e.g., 8 ConnectX-7 in DGX H100), pinning each GPU to its closest HCA maximizes bandwidth. `NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3` specifies the HCA list.

`NCCL_P2P_LEVEL` controls peer-to-peer (NVLink/PCIe) transfer policy: NVL (NVLink only), PIX (PCIe only within the same switch), PXB (PCIe across switches), PHB (PCIe through host bridge), or SYS (any path).

`NCCL_NET_GDR_LEVEL` controls GPUDirect RDMA usage level, enabling direct GPU-to-network transfers bypassing host memory.

`NCCL_CROSS_NIC` controls whether NCCL uses multiple NICs for inter-node traffic (set to 1 for multi-rail InfiniBand configurations).

## Multi-Node Communication

For multi-node distributed training, NCCL establishes connections between GPUs on different nodes through InfiniBand or Ethernet. GPUDirect RDMA (GDR) enables direct transfers between GPU memory and the InfiniBand NIC, bypassing host memory copies and reducing latency by 30-50%.

NCCL supports multi-rail configurations where each GPU is paired with a dedicated InfiniBand HCA. In an 8-GPU system with 8 InfiniBand ports, each GPU gets dedicated network bandwidth. The `NCCL_IB_HCA` and `NCCL_CROSS_NIC` variables control the mapping.

For large clusters with non-blocking fat-tree InfiniBand fabrics, NCCL's traffic patterns interact with the fabric's routing. Adaptive routing on the switch side and NCCL's internal channel management work together to avoid congestion.

## Communicator Management

NCCL communicators define the group of GPUs participating in collective operations. `ncclCommInitRank()` initializes a communicator with a unique ID, total rank count, and local rank. Multiple communicators can coexist for different parallelism dimensions — one for data parallelism, one for tensor parallelism, one for pipeline parallelism.

NCCL operations are asynchronous and use CUDA streams for overlap with compute kernels. The typical pattern is to launch NCCL operations on a dedicated communication stream while compute runs on another stream, using CUDA events for synchronization. This overlap is essential for hiding communication latency behind computation.

## Performance Debugging

When multi-GPU scaling is poor, NCCL provides several diagnostic tools. Setting `NCCL_DEBUG=INFO` reveals the detected topology, selected algorithms, and transport protocols. The NCCL Tests suite (`nccl-tests`) benchmarks raw collective performance: `all_reduce_perf` measures AllReduce bandwidth and latency across message sizes from 8 bytes to 8 GB.

Expected AllReduce bandwidth on 8x H100 with NVSwitch is approximately 450 GB/s (bus bandwidth). Significantly lower numbers indicate topology issues, driver problems, or incorrect NCCL configuration. Common root causes include disabled P2P access, wrong NIC selection, missing GPUDirect RDMA, or fabric congestion.

## Integration with Frameworks

PyTorch uses NCCL as its default backend for distributed training via `torch.distributed.init_process_group(backend="nccl")`. All DDP gradient synchronization, FSDP parameter sharding, and tensor-parallel communication flows through NCCL. Horovod wraps NCCL collectives with its own API but relies on the same underlying library. Megatron-Core uses NCCL communicators for all three parallelism dimensions (data, tensor, pipeline) in large-scale LLM training.

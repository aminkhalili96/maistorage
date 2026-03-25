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

## Algorithm Selection Per Topology

NCCL's algorithm choice significantly impacts performance and should be matched to the cluster topology:

**Ring algorithm** (`NCCL_ALGO=Ring`): Best for large messages (>1 MB) on systems with uniform bandwidth between all GPUs. Ring achieves maximum bandwidth utilization by pipelining data through all GPUs in a ring. On 8-GPU NVSwitch systems, ring AllReduce achieves ~450 GB/s bus bandwidth. Less optimal for small messages due to latency proportional to the number of GPUs.

**Tree algorithm** (`NCCL_ALGO=Tree`): Better for small-to-medium messages (<1 MB) where latency dominates. Tree reduces AllReduce latency from O(N) to O(log N) by aggregating in a binary tree pattern. Combines well with ring for mixed message sizes (NCCL auto-selects by default).

**CollNetDirect** (`NCCL_ALGO=CollNetDirect`): Uses NVIDIA SHARP for in-network aggregation on InfiniBand switches. Best for multi-node AllReduce on SHARP-capable fabrics (Quantum switches). Reduces inter-node traffic by ~50% and latency by 2-4x for small/medium messages.

**CollNetChain** (`NCCL_ALGO=CollNetChain`): Variant of SHARP that chains operations across switch levels. Useful when the SHARP tree cannot cover all nodes in a single aggregation pass.

For most deployments, leaving `NCCL_ALGO` unset allows NCCL to auto-select the best algorithm per message size. Override only after benchmarking with `nccl-tests`.

## Advanced Tuning Environment Variables

Beyond the basic variables, these advanced settings can resolve specific performance issues:

**`NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`**: Control the number of parallel communication channels. More channels improve bandwidth for large messages but increase memory usage and launch overhead. Default is auto-tuned based on topology. For H100 with NVSwitch, typical range is 16-32 channels.

**`NCCL_BUFFSIZE`**: Size of NCCL's internal communication buffers. Default 4 MB. Increase to 8-16 MB for large-scale AllReduce operations to reduce synchronization overhead. Each channel allocates `NCCL_BUFFSIZE` bytes.

**`NCCL_NTHREADS`**: Number of CUDA threads per NCCL kernel. Default 512. Increasing to 1024 can improve throughput on newer GPUs but may compete with compute kernels for SM resources.

**`NCCL_IB_QPS_PER_CONNECTION`**: Number of InfiniBand Queue Pairs per connection. Default 1. Increasing to 2-4 can improve multi-rail bandwidth but consumes more NIC resources.

**`NCCL_IB_GID_INDEX`**: GID index for RoCE v2 (ignored for InfiniBand). Must match the network interface's GID index. Wrong value causes connection failures on RoCE fabrics.

**`NCCL_CHECKS_DISABLE=1`**: Disables runtime error checking for production deployments. Reduces per-operation overhead but masks errors. Use only after thorough validation.

**`NCCL_IB_TIMEOUT`**: InfiniBand timeout exponent. Default 22 (~17 seconds). Increase on lossy fabrics to avoid spurious timeouts. Decrease on reliable fabrics for faster failure detection.

## NCCL Tests — Benchmarking Collective Performance

The `nccl-tests` suite is essential for validating cluster communication performance:

```bash
# Build nccl-tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make MPI=1 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/lib/x86_64-linux-gnu

# Single-node 8-GPU AllReduce benchmark
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8

# Multi-node benchmark (4 nodes × 8 GPUs, using MPI)
mpirun -np 32 --hostfile hosts -x NCCL_DEBUG=INFO \
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8
```

**Expected results on DGX H100 (8× H100 SXM5, NVSwitch)**:
- Intra-node AllReduce: ~450 GB/s bus bandwidth at 1 GB+ message sizes
- Inter-node AllReduce (NDR InfiniBand): ~380 GB/s per node pair
- AllGather: similar to AllReduce bandwidth
- ReduceScatter: ~90% of AllReduce bandwidth

**Red flags in nccl-tests output**:
- Bus bandwidth significantly below expected (>20% deviation) → topology issue, disabled P2P, or wrong NIC mapping
- High latency for small messages → incorrect algorithm selection or network congestion
- `NCCL WARN` about fallback transports → GPUDirect RDMA not working, falling back to staged copies
- Inconsistent performance across message sizes → memory contention or thermal throttling

## Multi-Node Debugging

When multi-node training scales poorly, systematic debugging steps:

1. **Verify single-node performance**: Run `nccl-tests` with `-g 8` on a single node. If intra-node bandwidth is low, check NVLink with `nvidia-smi nvlink --status` and Fabric Manager logs.

2. **Verify point-to-point bandwidth**: Run `ib_write_bw` between specific node pairs. Each InfiniBand link should achieve >380 Gb/s. Low bandwidth indicates cable issues, switch port problems, or PCIe bottlenecks.

3. **Check GPU-NIC affinity**: Each GPU should communicate through its NUMA-local InfiniBand HCA. Mismatched affinity causes PCIe cross-traffic. Verify with `nvidia-smi topo -m` and compare GPU-NIC pairing.

4. **Enable NCCL debug logging**: `NCCL_DEBUG=INFO` shows the topology graph, selected algorithms, and transport choices. Look for unexpected fallbacks (e.g., `NET/Socket` instead of `NET/IB` for inter-node, or `SHM` instead of `P2P/NVLink` for intra-node).

5. **Check for fabric congestion**: If bandwidth drops only at scale (not point-to-point), the issue is likely fabric oversubscription or routing. Use UFM or `perfquery` to check switch port counters for congestion and retransmit errors.

6. **Profile communication-computation overlap**: Use `NCCL_DEBUG_SUBSYS=COLL` with `NCCL_DEBUG=TRACE` to log individual collective timestamps. Compare with PyTorch Profiler traces to verify that communication overlaps with computation (gradient AllReduce should overlap with backward pass computation on earlier layers).

# NVIDIA SHARP — Scalable Hierarchical Aggregation and Reduction Protocol

SHARP offloads collective operations (AllReduce, Barrier, Broadcast) to InfiniBand switches, reducing network traffic and CPU/GPU overhead for distributed training. By performing data aggregation directly in the network fabric, SHARP eliminates redundant data transfers and dramatically lowers the latency of communication-bound training workloads.

## How SHARP Works

Traditional collective operations like AllReduce require data to traverse multiple hops between endpoints. In a ring AllReduce, every GPU sends and receives data proportional to the number of participants, resulting in O(N) network traffic. SHARP replaces this with in-network computing, where InfiniBand switches perform partial reduction operations as data passes through each level of the network topology.

On a fat-tree topology, SHARP constructs an aggregation tree that mirrors the physical switch hierarchy. As gradient data flows upward from leaf switches to spine switches, each SHARP-capable switch performs partial sum operations on the data. The fully reduced result is then broadcast back down the tree. This reduces the effective data traversal from O(N) to O(log N), where N is the number of participating endpoints. Switches already see all the data in transit — SHARP adds arithmetic capability to the switch ASIC, combining data from multiple input ports before forwarding at line rate.

## Architecture

The SHARP architecture consists of two main components: the SHARP Aggregation Manager (SAM) and SHARP-capable switch hardware.

The **SHARP Aggregation Manager (SAM)** runs as a daemon on a designated management node. SAM constructs and manages aggregation trees across the InfiniBand fabric by communicating with the subnet manager (OpenSM) to discover topology, then computing optimal tree structures. SAM handles tree allocation, resource management, and recovery when topology changes occur.

**SHARP-capable switches** are NVIDIA Quantum and Quantum-2 InfiniBand switches with dedicated ALUs in their ASICs. These ALUs perform floating-point addition, integer addition, min/max, and bitwise operations directly on packet payloads. Quantum-2 supports FP16, BF16, FP32, FP64, and INT32 data types for in-network reduction.

SHARP requires the MLNX_OFED driver stack with SHARP support enabled, SHARP-enabled switch firmware, and the `sharp_manager` utility for tree configuration.

## NCCL Integration

NCCL provides transparent SHARP integration through the CollNet algorithm. Enable with `NCCL_ALGO=CollNetDirect` (best for small/medium messages) or `NCCL_ALGO=CollNetChain` (better for large messages).

NCCL automatically detects SHARP availability by querying HCA capabilities via `hca_cap`. No application code changes are needed — PyTorch `DistributedDataParallel`, Megatron-LM, and DeepSpeed all benefit transparently. Verify SHARP is active by setting `NCCL_DEBUG=INFO` and checking for `CollNet` or `SHARP` in initialization logs.

## Performance Impact

SHARP delivers the most significant improvements for data-parallel training workloads that perform frequent gradient synchronization:

- **AllReduce latency** is reduced by 2-4x for small and medium messages (up to ~1 MB), where the operation is latency-bound rather than bandwidth-bound.
- **Bandwidth savings** of approximately 50% for large AllReduce operations, because data traverses fewer network hops. In a 256-node cluster on a 2-tier fat-tree, each gradient tensor crosses at most 4 switch hops instead of traversing the full ring.
- **CPU/GPU overhead** is reduced because endpoints perform fewer send/receive operations. GPUs spend less time waiting on communication and more time on computation.

SHARP is most impactful for **data-parallel** training, where every iteration requires an AllReduce across all GPUs for gradient synchronization. It is less impactful for **pipeline-parallel** patterns (point-to-point communication) or **tensor-parallel** patterns (typically confined within a single node via NVLink).

For large language model training using 3D parallelism, SHARP benefits the data-parallel dimension while NVLink handles tensor parallelism and point-to-point handles pipeline parallelism.

## Requirements

- **Switches**: NVIDIA Quantum (HDR, 200 Gb/s) or Quantum-2 (NDR, 400 Gb/s) InfiniBand switches with SHARP-enabled firmware.
- **HCAs**: ConnectX-6 or later Host Channel Adapters.
- **Driver stack**: MLNX_OFED 5.x or later with SHARP support compiled in.
- **Fabric configuration**: Properly configured subnet manager (OpenSM) and `sharp_manager` daemon on a management node.
- **Topology**: Fat-tree or similar hierarchical topology. SHARP trees cannot be constructed on non-hierarchical topologies.

## Limitations

SHARP has several important limitations to consider during cluster design:

- **InfiniBand only**: SHARP is not available on RoCE (RDMA over Converged Ethernet) networks. Clusters using Ethernet-based RDMA cannot use SHARP.
- **Homogeneous fabric**: All switches in the aggregation tree must be SHARP-capable. A mixed fabric with non-SHARP switches breaks the aggregation tree at those points.
- **Limited operation support**: SHARP supports AllReduce, Broadcast, and Barrier operations. It does not accelerate AllGather, ReduceScatter, or AlltoAll operations, which must still use traditional endpoint-based algorithms.
- **Tree reconfiguration**: Topology changes (node additions, switch failures) require SHARP tree reconfiguration via SAM, which may briefly disrupt running jobs.
- **Resource contention**: Concurrent SHARP trees are limited by switch resources. Large clusters may exhaust SHARP capacity, falling back to non-SHARP algorithms.

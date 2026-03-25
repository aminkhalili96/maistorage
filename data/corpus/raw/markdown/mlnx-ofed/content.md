# NVIDIA MLNX_OFED — InfiniBand and RoCE Driver Stack

MLNX_OFED (Mellanox OpenFabrics Enterprise Distribution) is the unified driver stack for NVIDIA/Mellanox ConnectX network adapters. It is required for InfiniBand and RoCE RDMA networking in GPU clusters, providing kernel drivers, userspace libraries, firmware management tools, and diagnostic utilities.

## Overview

MLNX_OFED packages everything needed to operate ConnectX-6, ConnectX-6 Dx, ConnectX-7, and BlueField DPU adapters. It supports InfiniBand HDR (200 Gb/s per port), InfiniBand NDR (400 Gb/s per port), and RoCE v2 over standard Ethernet networks. The distribution is validated against specific Linux kernel versions for RHEL, Ubuntu, and SLES.

MLNX_OFED replaces the in-kernel `rdma-core` and `mlx5` drivers with optimized versions that include capabilities not yet upstreamed to mainline Linux, such as SHARP support, GPUDirect RDMA, and advanced traffic management.

## Components

The MLNX_OFED distribution includes the following key components:

- **`mlx5_core`** — Core kernel driver for ConnectX adapters. Handles PCIe initialization, resource management, and low-level hardware access.
- **`mlx5_ib`** — InfiniBand kernel driver that registers the ConnectX adapter as an RDMA device and exposes it through the kernel verbs interface.
- **`libibverbs`** — Userspace RDMA verbs library. Provides the standard API for applications to perform RDMA operations (send, receive, read, write) without kernel involvement in the data path.
- **`libmlx5`** — ConnectX-specific provider library for libibverbs. Implements hardware-specific fast paths for optimal performance.
- **`ibverbs-utils`** — Diagnostic utilities including `ibv_devinfo` (display adapter info), `ibv_rc_pingpong` (basic connectivity test), and `ibv_bandwidth` (throughput measurement).
- **`infiniband-diags`** — Fabric diagnostic tools: `ibstat` (local port status), `ibdiagnet` (fabric-wide health check), `perfquery` (port performance counters), `ibnetdiscover` (topology discovery).
- **`mst`** — Mellanox Software Tools for firmware management. Provides `mst start` to create device nodes, `mlxfwmanager` for firmware queries and updates, and `mlxconfig` for firmware parameter configuration.
- **`opensm`** — OpenSM subnet manager for InfiniBand fabrics. Manages LID assignment, path computation, and routing table distribution across the fabric.

## Installation

MLNX_OFED is distributed as an `.iso` image or through a package repository. The recommended installation uses `mlnxofedinstall`:

```
./mlnxofedinstall --add-kernel-support --dpdk --upstream-libs
```

The `--add-kernel-support` flag compiles kernel modules against the running kernel. DKMS ensures automatic rebuild on kernel updates. The MLNX_OFED version must match the ConnectX firmware version — mismatches cause initialization failures or degraded performance. After installation, verify with `ibstat` (port state should show `Active`) and `ofed_info` for the installed version.

## ConnectX Adapter Configuration

ConnectX adapters expose configurable firmware parameters through `mlxconfig`:

- **Link speed**: Force specific speed or auto-negotiate when mixing HDR and NDR switches.
- **SR-IOV**: Enable virtual function passthrough with `mlxconfig set SRIOV_EN=1 NUM_OF_VFS=16`.
- **RoCE settings**: Configure ECN, DSCP priority, and PFC for lossless RoCE operation.

Firmware updates use `mlxfwmanager --online -u`. A cold reboot (full power cycle) is required after firmware updates on some adapter models.

## GPUDirect RDMA

MLNX_OFED includes kernel modules for GPUDirect RDMA, enabling direct DMA transfers between GPU memory and the ConnectX network adapter without staging through host CPU memory. This is critical for multi-node NCCL communication performance.

The legacy `nv_peer_mem` module has been replaced by `nvidia-peermem` (included with NVIDIA GPU driver 515+). Enable with `modprobe nvidia-peermem` and verify with `cat /sys/kernel/mm/memory_peers/nv_mem/version`.

Without GPUDirect RDMA, all inter-node GPU communication stages through host memory, adding latency and consuming PCIe bandwidth. For 8-GPU nodes with 8 InfiniBand ports, GPUDirect RDMA ensures each GPU communicates directly through its NUMA-local NIC.

## Performance Tuning

Key tuning parameters for maximizing RDMA throughput and minimizing latency:

- **IRQ affinity**: Use `set_irq_affinity_bynode.sh` to pin NIC interrupt handlers to CPUs on the same NUMA node as the adapter. Misaligned IRQ affinity can reduce throughput by 30-50%.
- **Adaptive interrupt coalescing**: Balances throughput and latency. Disable coalescing (`ethtool -C <dev> adaptive-rx off rx-usecs 0`) for latency-sensitive workloads.
- **PCI Max Read Request Size**: Set to 4096 bytes for maximum throughput: `setpci -s <BDF> 68.w=5936`.
- **Receive queue depth**: Increase for high-message-rate workloads to avoid queue overflow.

Verify point-to-point performance using the `perftest` suite:

- `ib_write_bw` — RDMA write bandwidth (should approach line rate: ~24.5 GB/s for HDR, ~49 GB/s for NDR)
- `ib_read_bw` — RDMA read bandwidth
- `ib_send_bw` — send/receive bandwidth
- `ib_write_lat` — RDMA write latency (typically ~1 us for InfiniBand)

## Troubleshooting

Common diagnostic workflows for MLNX_OFED issues:

- **`ibdiagnet`** — Fabric-wide health check reporting link errors, port counters, routing inconsistencies. Run from the subnet manager node.
- **`ibclearerrors`** — Resets port error counters. Useful for clean baselines before benchmarks.
- **Firmware mismatch** — All nodes should run the same MLNX_OFED and firmware versions. Check with `mlxfwmanager --query`.
- **Subnet manager conflicts** — Only one active OpenSM per subnet. Use `sminfo` to verify.
- **PCIe width negotiation** — ConnectX-7 needs PCIe Gen5 x16 for full NDR bandwidth. Verify with `lspci -vvv | grep Width`. x8 negotiation halves bandwidth.

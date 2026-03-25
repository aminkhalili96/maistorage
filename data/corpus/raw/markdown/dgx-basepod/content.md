# DGX BasePOD — Software Stack, Network Topology, and Deployment Reference

This supplement covers the software stack, network topology details, storage integration, and operational deployment patterns for DGX BasePOD that go beyond the basic deployment guide.

## Software Stack

A production DGX BasePOD deployment uses a layered software stack:

**Base OS**: Ubuntu 22.04 LTS (DGX OS) or RHEL 9 with NVIDIA-validated kernel. DGX OS includes pre-configured NVIDIA drivers, CUDA toolkit, container runtime, and monitoring agents. OTA updates via `nv-apt` repositories.

**GPU Software**: NVIDIA driver (535.x+ for H100), CUDA 12.x, cuDNN 9.x, NCCL 2.20+, Fabric Manager (for NVSwitch management), DCGM (for GPU health monitoring).

**Container Stack**: NVIDIA Container Toolkit, Docker CE or containerd, NGC CLI for pulling optimized containers. All training workloads run in containers — bare-metal execution is discouraged for reproducibility.

**Orchestration Options**:
- **Base Command Manager (BCM)**: NVIDIA's recommended orchestration layer. Provisions nodes, manages jobs, integrates with NGC. Web UI + CLI.
- **Slurm**: Most common for HPC-style batch scheduling. NVIDIA provides Slurm plugins for GPU resource management (`gres.conf` with `AutoDetect=nvml`), Pyxis for container integration, and Enroot for rootless container execution.
- **Kubernetes + GPU Operator**: For cloud-native deployments. GPU Operator auto-installs drivers, device plugin, DCGM exporter, and MIG manager. Kubeflow Training Operator for distributed training jobs.

**Monitoring**: DCGM exporter → Prometheus → Grafana. Fabric Manager logs for NVSwitch health. UFM (Unified Fabric Manager) for InfiniBand fabric monitoring. Syslog aggregation (rsyslog/Fluentd) for node-level diagnostics.

## Network Topology

### Compute Fabric (InfiniBand)

DGX BasePOD uses a rail-optimized InfiniBand topology:

- Each DGX H100 node has 8 ConnectX-7 NICs (one per GPU), each connected to a dedicated InfiniBand NDR (400 Gb/s) leaf switch.
- **Rail-optimized**: GPU 0 on every node connects to leaf switch 0, GPU 1 to leaf switch 1, etc. This means each leaf switch carries traffic for the same "rail" across all nodes.
- **Spine switches**: Leaf switches connect to spine switches for cross-rail communication. A 32-node BasePOD uses 8 leaf switches and 8 spine switches in a full fat-tree with 1:1 oversubscription ratio.
- **Non-blocking**: The full bisection bandwidth ensures AllReduce operations achieve maximum throughput regardless of GPU placement across nodes.

For larger deployments (64+ nodes), additional spine layers or director-class switches (Quantum-2 NDR) provide the required port density.

### Management Network

A separate 1/10 GbE management network handles:
- Node provisioning (PXE boot, DHCP, TFTP)
- Out-of-band management (BMC/IPMI)
- Orchestration traffic (Slurm control, Kubernetes API)
- Monitoring (Prometheus scrapes, syslog)
- Storage metadata (if using metadata-separated storage architecture)

### Storage Network

Storage traffic may share the InfiniBand compute fabric or use a dedicated storage network:
- **Shared fabric**: Lustre/BeeGFS clients use the same InfiniBand NICs as NCCL. Simpler topology but storage I/O competes with training communication. Use QoS (InfiniBand SL/VL mapping) to prioritize NCCL traffic.
- **Dedicated storage network**: Separate InfiniBand or Ethernet network for storage traffic. Eliminates contention but doubles network hardware. Recommended for I/O-heavy workloads (large-dataset training, frequent checkpointing).

## Storage Integration

### Three-Tier Storage Architecture

1. **Local NVMe scratch** (per-node): Each DGX H100 has 4× 3.84 TB NVMe SSDs (~15 TB raw). Used for local caching of training data, temporary checkpoint staging, and container image layers. Fastest I/O path but not shared across nodes.

2. **Parallel filesystem** (shared): BeeGFS or Lustre mounted on all compute nodes via InfiniBand RDMA. Stores training datasets (read-mostly) and checkpoints (write bursts). Sized at 10-20× total GPU memory for datasets, plus 5-10× for checkpoint history.

3. **Object storage** (archive): MinIO, S3, or Ceph for long-term storage of model artifacts, datasets, and checkpoint archives. Slower access but unlimited capacity. HSM integration with Lustre for transparent tiering.

### GPUDirect Storage

DGX BasePOD supports GPUDirect Storage for direct DMA between storage (local NVMe or parallel filesystem) and GPU memory. Reduces checkpoint write latency by bypassing CPU page cache. Requires `nvidia-fs` kernel module and `cuFile` API support in the training framework or I/O library (DALI, KvikIO).

## Deployment Checklist

Pre-deployment verification steps for a production BasePOD:

1. **Network validation**: Run `ib_write_bw` between all node pairs to verify InfiniBand bandwidth. Each link should achieve >380 Gb/s (95% of NDR 400 Gb/s line rate).
2. **GPU health**: Run `dcgmi diag -r 3` (Level 3 diagnostic) on all nodes. Checks GPU memory, NVLink bandwidth, PCIe bandwidth, and power subsystem.
3. **NCCL verification**: Run `nccl-tests` all_reduce_perf on all nodes simultaneously. 8-GPU intra-node bandwidth should reach ~450 GB/s (bus bandwidth). Multi-node bandwidth depends on InfiniBand fabric configuration.
4. **Storage benchmark**: Run `fio` or `mdtest` on the parallel filesystem from all compute nodes simultaneously. Verify aggregate bandwidth meets design target (e.g., 200+ GB/s for a 16-server BeeGFS deployment).
5. **Container runtime**: Pull and run an NGC PyTorch container on every node. Verify GPU visibility (`nvidia-smi`), NCCL communication, and filesystem mounts.
6. **Monitoring stack**: Verify Prometheus scrapes DCGM metrics from all nodes, Grafana dashboards render correctly, and alert rules fire on simulated failures.

## Upgrade Paths

**GPU generation transition** (e.g., H100 → B200):
- Add B200 nodes to the fabric alongside existing H100 nodes. Use Slurm partitions or Kubernetes node labels to separate GPU types.
- Training jobs specify GPU type via `--gres=gpu:b200:8` (Slurm) or `nvidia.com/gpu.product` node selector (K8s).
- Mixed-generation clusters are supported but training jobs should use homogeneous GPU types within a single job for consistent performance.

**Network expansion**: Add leaf/spine switches for new nodes. InfiniBand supports hot-add without fabric reconfiguration (subnet manager re-routes automatically). Verify routing tables after expansion with `ibdiagnet`.

**Storage expansion**: Add storage servers to BeeGFS/Lustre. New OSTs/targets join the pool automatically. Rebalance data with `beegfs-ctl --migrate` or `lfs migrate` to distribute across new targets.

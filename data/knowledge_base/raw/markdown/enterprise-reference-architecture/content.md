# NVIDIA Enterprise Reference Architecture for AI Infrastructure

This reference architecture covers the design of multi-node GPU clusters for AI training and inference at scale, including compute topology, networking, storage, power/cooling, software stack, and expansion planning. It is based on NVIDIA DGX and HGX platforms for enterprise deployments.

## Compute Topology

The fundamental compute building block is an 8-GPU node using SXM5 form factor GPUs connected via NVSwitch. Within each node, NVSwitch provides all-to-all NVLink connectivity between all 8 GPUs at 900 GB/s bidirectional bandwidth per GPU (NVLink 4.0 on H100). This intra-node fabric is the **scale-up** domain, where tensor-parallel workloads operate with minimal communication overhead.

Between nodes, InfiniBand or Ethernet provides the **scale-out** domain for data-parallel and pipeline-parallel communication. Each GPU connects to a dedicated network port, giving an 8-GPU node 8 independent network connections for maximum bisection bandwidth.

Typical deployment sizes based on use case:

- **4-32 nodes (32-256 GPUs)**: Departmental clusters for fine-tuning, inference serving, and small-scale training.
- **32-256 nodes (256-2048 GPUs)**: Production training clusters for large language models (7B-70B parameters) and enterprise AI workloads.
- **256+ nodes (2048+ GPUs)**: Frontier model training for 100B+ parameter models, requiring dedicated facilities.

DGX nodes provide fully integrated, validated configurations. HGX nodes use the same GPU baseboard in OEM server chassis with more flexibility in CPU, memory, and storage.

## Network Design

The recommended network topology is a two-tier InfiniBand fat-tree with 1:1 oversubscription (non-blocking). Leaf switches connect to GPU ports on compute nodes, while spine switches provide full bisection bandwidth. With NDR (400 Gb/s) switches offering 64 ports, a single leaf can connect 32 GPU ports at 1:1 ratio or 64 at 2:1 oversubscription.

**Rail-optimized topology** is preferred for GPU clusters: each GPU index across all nodes connects to the same dedicated leaf switch (rail). GPU 0 on all nodes connects to rail switch 0, GPU 1 to rail switch 1, and so on. This ensures AllReduce operations within a single rail never cross spine switches.

A separate **management network** (1/10 GbE) handles SSH, BMC/IPMI, job scheduling, and monitoring. This network must never carry training data traffic.

## Storage Architecture

AI training storage follows a three-tier hierarchy optimized for different access patterns:

- **Tier 1 — Local NVMe scratch**: NVMe SSDs (2-8 TB per node) for dataset caching, shuffle buffers, and checkpoint staging. Lowest latency and highest IOPS but not shared across nodes.
- **Tier 2 — Parallel filesystem**: Shared parallel filesystem (BeeGFS, Lustre, or GPFS) for training datasets and checkpoints. Sizing: 10-20x aggregate GPU memory for datasets, 5-10x for checkpoints. A 256-GPU H100 cluster needs 200-400 TB of parallel filesystem capacity.
- **Tier 3 — Object storage**: S3-compatible object storage (MinIO, Ceph) for archived model weights, datasets, and experiment artifacts.

**GPUDirect Storage** enables direct DMA between GPU memory and storage devices, bypassing CPU page cache. Valuable for loading large datasets where CPU-mediated I/O is a bottleneck.

## Power and Cooling

GPU clusters are among the most power-dense computing environments:

- **Per-node power**: DGX H100 draws ~10.2 kW peak (8 x 700W GPUs + CPUs/NICs/fans). DGX B200 draws up to 14.3 kW.
- **Rack density**: 40-80 kW per rack with liquid cooling. Air-cooled limited to 25-35 kW per rack.
- **PDU and UPS sizing**: Plan 1.1-1.2x peak compute draw for distribution losses. UPS should cover 5-10 minutes for checkpoint save.

**Liquid cooling** is strongly recommended for SXM5 GPU deployments. Direct-to-chip cooling uses a CDU (Coolant Distribution Unit) to circulate coolant through cold plates on GPU and CPU packages, removing 70-80% of heat load from air. Rear-door heat exchangers cool remaining air-cooled components.

## Software Stack

The validated software stack for NVIDIA AI infrastructure consists of:

- **Base OS**: Ubuntu 22.04 LTS or RHEL 9.x with HPC tuning (disabled power management, locked CPU frequencies, NUMA-aware scheduling).
- **GPU software**: NVIDIA driver (535+), CUDA Toolkit (12.x), cuDNN, and NCCL.
- **Container runtime**: NVIDIA Container Toolkit for GPU containers. NGC base images provide validated PyTorch, TensorFlow, and Megatron-LM environments.
- **Kubernetes**: GPU Operator automates driver, toolkit, device plugin, and DCGM deployment. Provides GPU scheduling, MIG partitioning, and monitoring.
- **Cluster management**: Base Command Manager (BCM) for bare-metal, Slurm for HPC scheduling, or Kubernetes with Volcano/Kueue for cloud-native batch.
- **Monitoring**: DCGM exports GPU metrics (utilization, temperature, memory, ECC errors, NVLink throughput) to Prometheus/Grafana via DCGM Exporter.

## Upgrade and Expansion

Enterprise AI clusters must support iterative growth and technology transitions:

- **Compute expansion**: Hot-add nodes by connecting to available leaf switch ports. The InfiniBand subnet manager automatically discovers new endpoints and recomputes routing.
- **GPU generation transitions**: Mixed GPU generations (A100 + H100) supported via Slurm GRES types or Kubernetes node labels. Avoid mixing generations within a single training job.
- **Storage expansion**: Add OSTs or storage servers. BeeGFS and Lustre support online capacity expansion without downtime.
- **Network expansion**: Add spine switches for more bisection bandwidth. Rail-optimized topologies support non-disruptive spine additions.

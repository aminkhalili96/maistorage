# Advanced Kubernetes Patterns for GPU Clusters

Beyond basic GPU scheduling, production Kubernetes deployments for AI infrastructure require specialized patterns for multi-GPU, multi-node training and inference. These patterns address the unique challenges of distributed deep learning workloads: gang scheduling to prevent NCCL deadlocks, topology-aware placement to exploit NVLink interconnects, GPU partitioning for inference efficiency, and resilient storage configurations for checkpoint and dataset management. This guide covers the production-grade Kubernetes patterns used in NVIDIA GPU clusters running large-scale training and inference workloads.

## GPU Scheduling Fundamentals

Kubernetes schedules GPUs as extended resources using the `nvidia.com/gpu` resource type. The NVIDIA GPU Operator deploys a device plugin DaemonSet that advertises available GPUs to the kubelet on each node. Pods request GPUs in their resource spec:

```yaml
resources:
  limits:
    nvidia.com/gpu: 4
```

The GPU Operator automatically labels nodes with GPU metadata, including `nvidia.com/gpu.product` (e.g., `NVIDIA-H100-SXM5-80GB`), `nvidia.com/gpu.count`, `nvidia.com/gpu.memory`, and `nvidia.com/cuda.driver.major`. These labels enable precise node selection through node affinity rules. For example, a training job requiring H100 GPUs can use a `nodeAffinity` with `matchExpressions` on `nvidia.com/gpu.product`. GPU nodes should be tainted (e.g., `nvidia.com/gpu=present:NoSchedule`) so non-GPU workloads are not scheduled on expensive GPU nodes. GPU pods include a corresponding toleration to bypass the taint.

## Gang Scheduling

Distributed training with NCCL requires all participating processes to initialize collectively. If Kubernetes schedules only a subset of pods in a training job, the running pods block indefinitely waiting for peers during `nccl_init`. The Volcano batch scheduler solves this with all-or-nothing scheduling using the PodGroup CRD:

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: llm-training-group
spec:
  minMember: 8
  queue: gpu-training
  priorityClassName: training-high
```

The `minMember` field must match the total `WORLD_SIZE` of the distributed training job. Volcano holds all pod scheduling until sufficient resources are available for the entire group. Without gang scheduling, a cluster running multiple training jobs can enter a circular deadlock where each job holds a partial GPU allocation, and none can proceed. Volcano also supports fair-share scheduling across queues, preventing a single large job from starving smaller jobs.

## Topology-Aware Placement

GPU interconnect topology significantly impacts training throughput. Within a single node, GPUs connected via NVLink achieve 450-900 GB/s bandwidth (NVLink 4.0), compared to 32 GB/s over PCIe Gen4. The NVIDIA Topology-Aware GPU Scheduling plugin integrates with the kubelet topology manager to schedule pods on GPUs that share NVLink domains. Configure the kubelet with `topologyManagerPolicy: best-effort` or `restricted` to enable topology-aware resource alignment.

For multi-node training, pod anti-affinity rules spread replicas across physical nodes to maximize aggregate bandwidth. Nodes can be labeled with NVLink domain identifiers (e.g., `nvidia.com/nvlink.domain`) so the scheduler can group pods requiring high-bandwidth intra-node communication. In DGX systems with 8 GPUs per node, scheduling a 4-GPU job on GPUs within the same NVLink domain can double all-reduce throughput compared to arbitrary GPU assignment.

## MIG and MPS Sharing

Multi-Instance GPU (MIG) partitions A100 and H100 GPUs into isolated GPU instances, each with dedicated compute units, memory bandwidth, and L2 cache. The GPU Operator's MIG manager automates profile configuration through ConfigMaps, supporting profiles like `1g.10gb`, `2g.20gb`, `3g.40gb`, and `7g.80gb` on the H100. Each MIG instance appears as a separate resource (e.g., `nvidia.com/mig-1g.10gb`), enabling fine-grained allocation for inference workloads.

Multi-Process Service (MPS) provides time-sharing of a single GPU across multiple pods. MPS is suited for inference workloads where individual models underutilize GPU compute. The GPU Operator enables MPS through the `nvidia.com/mps.capable` label and device plugin configuration.

Trade-offs: MIG provides hardware-level isolation with separate memory and fault domains, but reduces scheduling flexibility since profiles must be pre-configured and the GPU cannot be recombined without draining workloads. MPS provides better GPU utilization for small models but shares the fault domain — a CUDA error in one process affects all processes sharing the GPU.

## Multi-Node Training Jobs

The Kubeflow Training Operator provides the PyTorchJob CRD for managing distributed PyTorch training on Kubernetes. It automatically configures the required environment variables: `MASTER_ADDR` (set to the master pod's DNS name), `MASTER_PORT`, `WORLD_SIZE`, and `RANK` for each worker pod. A headless Service provides stable DNS resolution for inter-pod communication during training.

For RDMA/InfiniBand-based training clusters, pods must set `hostNetwork: true` to bypass kube-proxy and access the host's InfiniBand interfaces directly. NCCL environment variables are configured in the pod spec:

```yaml
env:
  - name: NCCL_IB_DISABLE
    value: "0"
  - name: NCCL_NET_GDR_LEVEL
    value: "5"
  - name: NCCL_IB_HCA
    value: "mlx5"
  - name: NCCL_SOCKET_IFNAME
    value: "eth0"
  - name: NCCL_DEBUG
    value: "INFO"
```

Setting `NCCL_NET_GDR_LEVEL=5` enables GPUDirect RDMA, allowing GPU memory to be transferred directly over InfiniBand without staging through host memory. The `NCCL_IB_HCA` variable selects the InfiniBand HCA to use for NCCL communication.

## Network Policies for GPU Clusters

Network policies must allow unrestricted NCCL traffic between training pods. NCCL uses dynamically assigned TCP/UDP ports, so policies should allow all traffic within the training namespace or between pods matching the training job's label selector. RDMA traffic requires either `hostNetwork: true` or a secondary network via Multus CNI with SR-IOV device plugin for InfiniBand interfaces.

Network attachment definitions in Multus enable pods to have secondary network interfaces for dedicated high-bandwidth networks:

- SR-IOV for InfiniBand: exposes virtual functions directly to pods for near-native RDMA performance
- Macvlan/IPVLAN: provides direct access to high-performance storage networks (e.g., dedicated NFS or Lustre subnets) without NAT overhead
- Host-device: binds a host network interface directly to a pod for maximum throughput

## Persistent Storage Patterns

Training workloads have distinct storage patterns for datasets, checkpoints, and scratch space. Datasets are typically mounted as ReadOnlyMany PVCs backed by parallel filesystems (Lustre, BeeGFS, or GPFS via CSI drivers), allowing all training pods to read the same data concurrently. Checkpoint volumes require ReadWriteMany semantics since multiple replicas may write checkpoints simultaneously, typically using NFS, Lustre, or BeeGFS.

Local NVMe storage is accessed via `emptyDir` volumes with `medium: ""` (or hostPath for persistent local storage), providing high-bandwidth scratch space for data loading pipelines and temporary shuffled datasets. Init containers can pre-stage datasets from S3 or MinIO to local NVMe before training begins, reducing I/O contention during training iterations.

For GPUDirect Storage, the NVIDIA GPUDirect Storage CSI driver enables direct data path between NVMe/NFS storage and GPU memory, bypassing the CPU bounce buffer. This is critical for workloads with large dataset I/O requirements.

## Health Checks and Fault Tolerance

The DCGM (Data Center GPU Manager) exporter exposes GPU health metrics as Prometheus metrics, including temperature, power, ECC error counts, and GPU utilization. Custom readiness probes can query DCGM to verify GPU health before accepting training workloads:

- Xid errors (uncorrectable ECC, GPU fallen off bus) trigger automatic pod eviction
- Double-bit ECC errors indicate hardware degradation and should trigger node cordon
- GPU throttling due to thermal limits is detected via `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS`

PyTorch Elastic (torchelastic) enables fault-tolerant distributed training by allowing the training job to continue when individual workers fail. The Elastic Training Operator automatically manages worker replacement and rendezvous. Combined with periodic checkpoint saving, training can recover from single-node failures without restarting the entire job. The `c10d` rendezvous backend integrates with Kubernetes endpoints for dynamic membership.

## Helm Patterns

Production GPU clusters use Helm charts with standardized values for resource management. Key patterns include:

**Resource quotas** limit GPU consumption per namespace, preventing a single team from monopolizing cluster GPUs:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
spec:
  hard:
    requests.nvidia.com/gpu: "16"
    limits.nvidia.com/gpu: "16"
```

**LimitRange** enforces default GPU requests and prevents pods from requesting excessive GPUs without explicit approval.

**PriorityClasses** establish a scheduling hierarchy across workload types: training jobs (high priority, preemption enabled), inference serving (medium priority, non-preemptible for SLA compliance), and interactive/notebook workloads (low priority, preemptible). When cluster resources are exhausted, lower-priority workloads are preempted to make room for higher-priority training jobs. This tiered approach ensures that production inference is never disrupted while training jobs receive priority over development workloads.

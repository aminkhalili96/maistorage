# Kubernetes for AI and GPU Workloads

Kubernetes has become the standard platform for deploying, scaling, and managing containerized AI workloads. With the NVIDIA GPU Operator and device plugin ecosystem, Kubernetes provides automated GPU lifecycle management, scheduling, and orchestration for both training and inference.

## Pod-Level GPU Allocation

GPUs are exposed to pods as extended resources via the NVIDIA device plugin. Pods request GPUs in their resource spec:

```yaml
resources:
  limits:
    nvidia.com/gpu: 2
```

The NVIDIA device plugin (`k8s-device-plugin`) runs as a DaemonSet, discovers GPUs on each node via NVML, and registers them with the kubelet. GPUs are allocated as whole devices with no overcommit. A pod requesting 2 GPUs sees exactly those 2 via `NVIDIA_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` environment variables injected by the NVIDIA container runtime.

## DaemonSets for GPU Infrastructure

DaemonSets ensure one pod per node, making them the standard pattern for GPU infrastructure: the NVIDIA driver container (manages driver lifecycle without host installation), NVIDIA device plugin (registers GPUs with kubelet), DCGM exporter (exports GPU telemetry to Prometheus), and Node Feature Discovery (labels nodes with GPU model, driver version, MIG support). The GPU Operator deploys all of these as managed DaemonSets.

## Jobs and CronJobs for Training

Kubernetes `Job` resources fit batch training runs: they create pods, retry on failure via `backoffLimit`, enforce time limits with `activeDeadlineSeconds`, and support distributed data-parallel training via `completions`/`parallelism`. `CronJob` schedules periodic retraining or data preprocessing. For multi-node distributed training, PyTorch Elastic (`torchrun`) or the Kubeflow Training Operator (`PyTorchJob`) manage worker coordination and elastic scaling.

## StatefulSets for Model Serving

StatefulSets provide stable network identities and ordered deployment for model serving replicas needing persistent volumes. Each replica gets a predictable hostname and its own PersistentVolumeClaim. NVIDIA Triton Inference Server is commonly deployed as a StatefulSet behind a Kubernetes Service, with HPA driven by GPU utilization or queue depth metrics from DCGM.

## NVIDIA GPU Operator

The GPU Operator automates the full GPU software stack lifecycle. It deploys: containerized NVIDIA driver, `nvidia-container-toolkit`, device plugin, DCGM exporter, MIG manager, and GPU Feature Discovery. Installation is a single Helm chart: `helm install gpu-operator nvidia/gpu-operator --namespace gpu-operator`. It watches for GPU nodes joining the cluster and provisions the full stack automatically.

## MIG (Multi-Instance GPU) Support

Multi-Instance GPU partitions a single A100 or H100 into up to 7 isolated instances, each with dedicated compute, memory, and cache. In Kubernetes, MIG devices appear as separate resource types:

```yaml
resources:
  limits:
    nvidia.com/mig-3g.20gb: 1
```

The MIG manager configures profiles via node labels, and the MIG device plugin advertises slices to the scheduler. This enables multiple inference workloads to share a physical GPU with hardware-level isolation.

## Resource Quotas and Scheduling

`ResourceQuota` objects prevent GPU overcommit per namespace (`nvidia.com/gpu: "8"`). For GPU-specific scheduling, use `nodeSelector` or `nodeAffinity` to target GPU pools, and `taints`/`tolerations` to prevent non-GPU workloads from landing on expensive GPU nodes:

```bash
kubectl taint nodes gpu-node-01 nvidia.com/gpu=present:NoSchedule
```

## Topology-Aware Scheduling

GPU topology matters for multi-GPU training. GPUs connected via NVLink perform 5-10x better for all-reduce than those on PCIe. The `TopologyManager` kubelet policy (`topologyManagerPolicy: best-effort` or `restricted`) ensures pods receive GPUs with optimal interconnect. NUMA-aware scheduling (`topologyManagerScope: pod`) co-locates CPUs, memory, and GPUs in the same NUMA domain, reducing cross-socket latency.

## Network Plugins for RDMA

High-performance distributed training requires RDMA for inter-node GPU communication via NCCL. Kubernetes supports this through Multus CNI (secondary InfiniBand/RoCE interfaces), SR-IOV device plugin (InfiniBand VFs as schedulable resources), or host networking (`hostNetwork: true`). For GPUDirect RDMA, the `nvidia-peermem` kernel module enables NCCL transfers directly between GPUs on different nodes without CPU staging.

## Storage for AI Workloads

PersistentVolumeClaims back pods with storage via CSI drivers. Common patterns: BeeGFS or Lustre CSI for high-bandwidth parallel I/O, NFS CSI for smaller datasets, and local NVMe (hostPath or local PV) for lowest-latency data caching. For GPUDirect Storage, the `cuFile` API enables direct GPU-to-storage transfers via compatible CSI drivers.

## Kubeflow and ML Pipelines

Kubeflow extends Kubernetes with ML-specific resources: `TFJob`, `PyTorchJob`, `MPIJob` for distributed training; `KServe` for model serving with canary rollouts; `Katib` for hyperparameter tuning; and `Pipelines` for DAG-based ML workflows.

## Kubernetes vs Slurm

Kubernetes excels at service deployment, operator-driven automation, and heterogeneous workload orchestration. Its scheduler is extensible but lacks Slurm's backfill, gang scheduling, and mature fair-share policies. For pure multi-node training at HPC scale, Slurm remains more efficient. Many organizations run both: Slurm for large-scale training on bare-metal GPU clusters, Kubernetes for inference serving, MLOps tooling, and smaller training jobs.

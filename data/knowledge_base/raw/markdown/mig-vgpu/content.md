# NVIDIA MIG and vGPU -- GPU Virtualization and Partitioning

Multi-Instance GPU (MIG) and vGPU are NVIDIA's complementary technologies for GPU partitioning in multi-tenant environments. MIG provides hardware-level isolation by physically partitioning a GPU into independent instances on A100, H100, H200, and B200 architectures. vGPU enables software-based time-shared GPU access for VDI, AI, and compute workloads across a broader range of GPUs.

## MIG Architecture

MIG partitions a single physical GPU into up to 7 independent instances, each with dedicated streaming multiprocessors (SMs), memory bandwidth, L2 cache, and memory controllers. This is hardware-level isolation: one MIG instance cannot access another's memory, and a noisy-neighbor workload cannot degrade another instance's performance.

Supported GPUs:
- **A100 (40GB/80GB):** Up to 7 instances
- **H100 (80GB):** Up to 7 instances with higher per-instance performance
- **H200 (141GB):** Up to 7 instances with the largest per-instance memory
- **B200:** Next-generation MIG support

## MIG Profiles

MIG profiles define resource allocation using the convention `<compute_slices>g.<memory>gb`. On an A100 80GB:

| Profile | Compute (SMs) | Memory | GPU Fraction |
|---------|--------------|--------|--------------|
| 1g.10gb | 1/7 of SMs | 10 GB | 1/7 |
| 2g.20gb | 2/7 of SMs | 20 GB | 2/7 |
| 3g.40gb | 3/7 of SMs | 40 GB | 3/7 |
| 4g.40gb | 4/7 of SMs | 40 GB | 4/7 |
| 7g.80gb | All SMs | 80 GB | Full GPU |

Mixed profiles are supported: one 3g.40gb plus two 2g.20gb instances on a single A100 80GB.

Profile management via `nvidia-smi`:
- Enable MIG: `nvidia-smi -i <gpu-id> -mig 1` (requires GPU reset)
- Create instance: `nvidia-smi mig -cgi <profile> -C`
- List instances: `nvidia-smi mig -lgi`
- Destroy instances: `nvidia-smi mig -dci` then `nvidia-smi mig -dgi`

## MIG Use Cases

**Inference multi-tenancy:** Multiple models on one GPU, each in its own MIG instance with guaranteed resources. A single A100 serves 7 small models simultaneously without interference.

**Development and testing:** Small GPU slices (1g.10gb) for individual developers, making expensive GPUs accessible to more team members.

**Kubernetes orchestration:** The NVIDIA GPU Operator's MIG Manager auto-configures profiles based on pod resource requests. Each MIG instance appears as a separate GPU resource (`nvidia.com/mig-1g.10gb`), enabling standard pod scheduling.

**Mixed workload consolidation:** Training on a 4g.40gb instance while serving inference on three 1g.10gb instances from the same physical GPU.

## vGPU Architecture

NVIDIA vGPU is software-based GPU virtualization enabling multiple VMs to share a physical GPU through time-slicing. The vGPU Manager kernel module on the hypervisor mediates GPU access and enforces scheduling in round-robin fashion.

vGPU product types:
- **vCS (Virtual Compute Server):** Optimized for AI/ML and HPC compute
- **vWS (Virtual Workstation):** Quadro-equivalent graphics for professional visualization
- **vApps (Virtual Applications):** Lightweight GPU acceleration for app streaming

vGPU requires a license from NVIDIA, managed through a License Server (on-premises or cloud-based).

## vGPU vs MIG

| Feature | MIG | vGPU |
|---------|-----|------|
| Isolation | Hardware (physical partitioning) | Software (time-slicing) |
| Overhead | Near-zero (<2%) | Moderate (5-15%) |
| Minimum GPU | A100 / H100 / H200 / B200 | T4, V100, A-series, and newer |
| Max instances | 7 per GPU | Up to 32 per GPU (varies) |
| Flexibility | Fixed profiles, reset required | Dynamic, no reset needed |
| Use case fit | Inference serving, Kubernetes | VDI, VM-based AI, legacy workloads |

**MIG + vGPU combined:** On A100 and H100, MIG instances can be passed through to VMs via vGPU, combining hardware isolation with virtualization management. Each MIG instance becomes a vGPU-capable device with deterministic performance.

## Enterprise Deployment

**NVIDIA AI Enterprise** includes vGPU licenses bundled with AI frameworks, supporting both MIG and vGPU deployment modes.

**Kubernetes integration:** The GPU Operator handles MIG profile configuration and device plugin registration for MIG, and deploys vGPU-aware device plugins for vGPU environments.

**Hypervisor support:** vGPU works on VMware vSphere, KVM/QEMU (including Red Hat OpenShift Virtualization), Citrix Hypervisor, and Nutanix AHV. MIG pass-through works with any hypervisor supporting PCIe or mediated device (mdev) pass-through.

**Monitoring:** DCGM provides per-MIG-instance metrics (SM utilization, memory, temperature, ECC errors). The vGPU Manager reports per-VM GPU utilization and framebuffer usage. Both integrate with Prometheus and Grafana for unified GPU observability.

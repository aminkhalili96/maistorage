# Cluster Management and Linux Operations — Technical Reference

GPU compute clusters require disciplined systems administration across networking, OS provisioning, driver management, monitoring, and workload scheduling. Operational failures in AI infrastructure most often originate from version drift, misconfigured nodes, network issues, and storage-path problems rather than model code.

## Cluster Topology and Networking

A production GPU cluster uses multiple physically separated networks. The compute fabric is typically InfiniBand (NDR 400 Gb/s or HDR 200 Gb/s) in a spine-leaf topology for NCCL collective operations during distributed training. A separate management network (1-10 GbE) carries SSH, Ansible, monitoring, and provisioning traffic. Storage traffic may ride a dedicated InfiniBand partition or a separate Ethernet fabric to avoid contention with compute collectives. A BMC/IPMI network provides out-of-band hardware management, remote console access, and power control for every node. IP addressing, DNS, and NTP synchronization must be configured correctly on all networks — NTP is particularly critical because distributed training frameworks use timestamps for coordination, and clock skew causes subtle failures.

## OS Deployment and Image Management

Bare metal GPU nodes are provisioned via PXE network boot using tools such as Warewulf, xCAT, or Foreman. The preferred approach is image-based provisioning where a golden OS image is built once, tested, and deployed identically to every node. This ensures consistent kernel versions, driver installations, and library paths across the cluster. Immutable or semi-immutable OS images (where the root filesystem is read-only or re-imaged on every boot) reduce configuration drift. Stateful data (user home directories, local scratch) resides on separate partitions or network mounts.

## NVIDIA Driver Stack and CUDA Toolkit

The NVIDIA software stack has strict version dependencies: the kernel-mode GPU driver must be compatible with the installed CUDA toolkit version, which must be compatible with cuDNN, which must be compatible with the deep learning framework (PyTorch, TensorFlow, JAX). NVIDIA publishes a compatibility matrix for each release. The GPU driver is a kernel module installed via DKMS (Dynamic Kernel Module Support), which automatically recompiles the driver when the kernel is updated. Kernel version pinning is strongly recommended to avoid surprise driver breakage after unattended upgrades. Secure Boot complicates driver installation because DKMS-built modules require MOK (Machine Owner Key) enrollment for signing. Multiple CUDA toolkit versions can coexist on the same node using environment modules (`module load cuda/12.4`) or the `update-alternatives` system. Containers largely sidestep this problem by bundling the CUDA user-space toolkit inside the image, requiring only a compatible host driver.

## Node Health Checks and Diagnostics

Before a node enters the scheduling pool, automated health checks should verify GPU functionality. NVIDIA DCGM (Data Center GPU Manager) provides `dcgmi diag` which runs multi-level diagnostics: Level 1 checks basic GPU availability and driver binding, Level 2 runs memory stress tests, and Level 3 performs sustained compute stress tests that detect marginal hardware. `nvidia-smi` reports GPU temperature, power draw, ECC error counts, and NVLink status. PCIe bandwidth tests (`cuda_bandwidthTest` from CUDA samples or `dcgmi diag -r 3`) verify that each GPU has full PCIe bandwidth to the host. Xid errors in `dmesg` or kernel logs indicate GPU hardware or driver faults — certain Xid codes (e.g., Xid 79: GPU fallen off the bus) require immediate node removal from the cluster.

## Configuration Management

Ansible is the most common configuration management tool for GPU clusters due to its agentless architecture and straightforward YAML playbooks. Typical Ansible roles for GPU nodes include driver installation, CUDA toolkit setup, NCCL configuration, InfiniBand driver and subnet manager setup, Slurm client configuration, and monitoring agent deployment. Playbooks should be idempotent — running them twice produces the same result. Salt is an alternative with stronger event-driven capabilities. Configuration management must handle GPU-specific details: setting `nvidia-persistenced` to keep driver state loaded, configuring GPU compute mode and power limits via `nvidia-smi`, and deploying NCCL environment variables (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`) for optimal collective performance.

## Workload Scheduling

Slurm is the dominant scheduler for HPC-style GPU clusters. Key Slurm configuration for GPU workloads includes `GresTypes=gpu` in `slurm.conf`, per-node GPU GRES definitions, and job submission via `srun --gres=gpu:8` or `sbatch` scripts with `#SBATCH --gres=gpu:4`. Slurm's `cgroup` integration isolates GPU device access per job. Kubernetes is the alternative when the cluster serves mixed workloads including inference services, notebooks, and batch training. The NVIDIA GPU Operator deploys the driver, device plugin, DCGM exporter, and MIG manager as a DaemonSet, making GPU resources schedulable via `nvidia.com/gpu` resource requests. `gpu-feature-discovery` labels nodes with GPU model and driver version for affinity scheduling.

## Monitoring and Observability

A production GPU cluster monitoring stack typically combines Prometheus for metrics collection, Grafana for dashboards, and the DCGM Exporter for GPU-specific metrics (utilization, memory usage, temperature, ECC errors, NVLink throughput, PCIe bandwidth). Alerting rules should fire on sustained GPU temperatures above 83C, double-bit ECC errors, GPU utilization dropping to zero during active jobs (indicating a hang), and NVLink errors. Centralized log aggregation via rsyslog forwarding, Loki, or the ELK stack captures kernel-level GPU errors (Xid messages in `dmesg`), driver crash traces, and application-level training logs.

## Bare Metal vs Virtualized

AI training workloads almost universally run on bare metal because GPU virtualization (SR-IOV or vGPU) adds overhead, limits NVLink access, and complicates multi-GPU topologies. Virtualization with MIG or vGPU is appropriate for inference workloads where GPU sharing and multi-tenancy matter more than peak single-job throughput. User management relies on centralized authentication (LDAP or Active Directory), SSH key distribution, and shared home directories via NFS with autofs.

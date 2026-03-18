# RAG Supplemental Field Guide

## AI Server Platform Design

AI server design is not only about picking a GPU. The motherboard determines expansion options, PCIe layout, networking slots, storage attachment, and power delivery for accelerators and high-speed devices. Enterprise CPUs such as AMD EPYC or Intel Xeon are commonly chosen to provide memory channels, PCIe lanes, and host-side throughput for data loading, preprocessing, and orchestration. High-capacity RAM matters because loaders, host buffers, shuffles, and preprocessing pipelines can stall accelerators if the host side is undersized. PCIe is the baseline host-to-device interconnect, while NVLink is used when the workload benefits from tighter GPU-to-GPU communication bandwidth and lower communication overhead.

## Cluster Management And Linux Operations

AI compute clusters are usually managed either with an HPC scheduler such as Slurm or with container orchestration such as Kubernetes. Slurm-style scheduling is a strong fit when the cluster is optimized for queued batch jobs and tightly managed resource reservations. Kubernetes is attractive when the platform also needs repeatable service deployment, device plugins, operators, and container-native workflows. Linux operations remain critical because GPU clusters depend on correct kernel modules, driver installation, CUDA toolkit compatibility, and repeatable host configuration. A large share of operational troubleshooting still comes from version drift, bad node images, network bottlenecks, and storage-path misconfiguration rather than model code alone.

## Data Path And Storage

Parallel file systems are commonly used when many training workers need shared, high-throughput access to the same dataset namespace. Systems such as Lustre, BeeGFS, or GPFS are valuable when aggregate bandwidth and shared namespace behavior matter more than single-node convenience. S3-compatible object storage is often used for large datasets, checkpoints, model artifacts, and data lake style workflows because it scales well and integrates with modern data tooling. RAID remains a practical local-server decision: RAID 0 improves throughput but offers no redundancy, RAID 1 mirrors for safety, RAID 5 and RAID 6 trade capacity for redundancy, and RAID 10 balances mirrored protection with stronger performance at higher disk cost.

## Containerization And Delivery

Docker-style containerization helps package AI dependencies, CUDA user-space libraries, serving stacks, and supporting tools in a repeatable way. Containers reduce environment drift between workstations, servers, and clusters, but they still rely on correct host GPU drivers and runtime configuration. CI/CD pipelines matter because model deployment is not just code shipping; teams need automated image builds, tests, promotion gates, registry publishing, and controlled rollout of model-serving changes. In MLOps practice, CI/CD is what turns an experimental model artifact into a repeatable deployment process.

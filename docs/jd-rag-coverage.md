# JD RAG Coverage Map

This project is tailored to the MaiStorage AI Solution and Deployment Lead JD across two layers:

1. NVIDIA-first infrastructure sources already in the corpus
2. An offline local operations pack for the non-NVIDIA operational topics the JD expects

## Already Covered By The Existing Corpus

- CUDA installation and Linux driver alignment
- NVIDIA Container Toolkit
- GPU Operator for Kubernetes
- NCCL and NVLink / Fabric Manager
- Deep learning performance and mixed precision
- GPUDirect Storage and storage throughput
- H100 / A100 / L40S hardware positioning
- Megatron Core and distributed parallelism

## Added For Better JD Alignment

These topics were previously covered only by short supplemental notes. They now have dedicated offline local resources derived from primary documentation:

- Slurm workload management
- Kubernetes workload primitives
- BeeGFS architecture
- Lustre architecture
- MinIO / S3-compatible object storage
- Linux mdraid / RAID management
- Docker build and CI/CD workflows

## Remaining Intentional Limitations

- There is no single universal official document that gives exact BIOS settings for every EPYC motherboard model. That remains an intentional refusal / insufficient-evidence case.
- Server motherboard tuning remains a design-judgment topic rather than a single canonical vendor-neutral manual.

## Demo Framing

You can now ask the system about:

- multi-GPU and GPU architecture choices
- Slurm vs Kubernetes for AI clusters
- Linux drivers and CUDA installation
- parallel file systems, S3 object storage, and RAID
- Docker/containerization and CI/CD for model deployment
- NVIDIA-specific deployment, scaling, and troubleshooting topics

# NVIDIA H100 Tensor Core GPU -- Technical Reference

The NVIDIA H100 is the flagship data center GPU based on the Hopper architecture, designed for large-scale AI training, HPC simulation, and inference workloads. It succeeds the A100 (Ampere) and introduces the Transformer Engine, fourth-generation Tensor Cores, and native FP8 precision support.

## Architecture

The H100 is built on the GH100 GPU die, fabricated on TSMC 4N process technology with approximately 80 billion transistors. The full die contains 144 streaming multiprocessors (SMs); the SXM5 product ships with 132 SMs enabled. Each SM contains 128 CUDA cores and 4 fourth-generation Tensor Cores, yielding 16896 CUDA cores and 528 Tensor Cores across the GPU.

The Hopper architecture introduces a new Transformer Engine that dynamically selects between FP8 and FP16 precision on a per-layer basis during transformer model training and inference. This is managed automatically by the engine and requires no manual precision tuning. Hopper also adds the DPX instruction set for dynamic programming algorithms, delivering up to 7x speedup over the A100 on problems such as Smith-Waterman sequence alignment.

## Compute Performance

All Tensor Core TFLOPS figures below include the 2:4 structured sparsity multiplier where noted.

| Precision | TFLOPS (Dense) | TFLOPS (Sparse) |
|---|---|---|
| FP8 Tensor Core | 1979 | 3958 |
| FP16 / BF16 Tensor Core | 990 | 1979 |
| TF32 Tensor Core | 495 | 989 |
| FP64 Tensor Core | 67 | -- |
| FP32 (non-Tensor) | 67 | -- |
| INT8 Tensor Core | 1979 | 3958 |

The H100 delivers approximately 3x the FP16 training throughput and 6x the FP8 inference throughput compared to the A100 80GB SXM.

## Memory Subsystem

The SXM5 variant ships with 80 GB of HBM3 memory across a 5120-bit memory interface, providing 3.35 TB/s of peak memory bandwidth. This represents a 1.6x bandwidth improvement over the A100's 2 TB/s HBM2e interface.

The large memory capacity and bandwidth are critical for training large language models where optimizer states, activations, and model weights must reside in GPU memory. The H100 supports hardware-managed L2 cache partitioning (50 MB L2) for improved multi-tenant cache isolation.

## Interconnect

NVLink 4.0 provides 900 GB/s bidirectional bandwidth through 18 links per GPU. This is a 1.5x improvement over the A100's NVLink 3.0 (600 GB/s, 12 links). Within an HGX H100 8-GPU baseboard, all eight GPUs are fully connected via NVSwitch 3.0, providing non-blocking all-to-all communication at the full 900 GB/s per GPU.

The PCIe variant connects via PCIe Gen5 x16 with 128 GB/s bidirectional bandwidth. Both variants support GPUDirect RDMA and GPUDirect Storage for direct data paths between the GPU and network adapters or NVMe storage.

## Multi-Instance GPU (MIG)

The H100 supports MIG with up to 7 GPU instances, each with dedicated compute, memory, and memory bandwidth resources. MIG instances are hardware-isolated, making them suitable for multi-tenant cloud and enterprise deployments. Compared to A100 MIG, the H100 adds support for Confidential Computing within MIG instances.

## Confidential Computing

The H100 is the first NVIDIA GPU to support hardware-based Confidential Computing. Data and code running on the GPU are protected from the host CPU, hypervisor, and other tenants via hardware encryption of GPU memory. This enables secure multi-tenant GPU sharing in cloud environments.

## Thermal and Power

| Variant | TDP | Cooling |
|---|---|---|
| SXM5 | 700 W | Passive heatsink, system-level airflow or liquid cooling |
| PCIe Gen5 | 350 W | Dual-slot passive heatsink |

The SXM5 form factor requires purpose-built server platforms (DGX H100, HGX H100 baseboards in partner systems). The high TDP of 700 W per GPU typically requires liquid cooling for sustained operation at full utilization in 8-GPU configurations.

## Form Factors and System Integration

- **SXM5**: Used in DGX H100 (8x H100 SXM5, 640 GB total GPU memory) and HGX H100 baseboards from OEM partners. Full NVLink/NVSwitch connectivity.
- **PCIe Gen5**: Standard dual-slot card for deployment in mainstream servers. Lower power (350 W), no NVLink in most configurations. Suitable for inference and smaller training jobs.

## Use Cases

- **Large language model training**: The primary target workload. NVLink-connected 8-GPU nodes scale to thousands of GPUs for frontier model training.
- **HPC and scientific simulation**: FP64 Tensor Core performance (67 TFLOPS) is a 3x improvement over A100 for double-precision workloads.
- **Inference**: FP8 Tensor Core support and the Transformer Engine deliver high-throughput inference for transformer models. MIG allows efficient sharing across multiple inference workloads.
- **Distributed training infrastructure**: HGX H100 baseboards with NVSwitch form the building block for DGX SuperPOD clusters, connected via InfiniBand NDR (400 Gb/s) networking.

## Comparison to Predecessor (A100)

| Metric | A100 80GB SXM | H100 80GB SXM5 |
|---|---|---|
| Process | TSMC 7nm | TSMC 4N |
| CUDA Cores | 6912 | 16896 |
| Tensor Cores | 432 (3rd gen) | 528 (4th gen) |
| FP8 Support | No | Yes |
| Memory | 80 GB HBM2e | 80 GB HBM3 |
| Memory BW | 2 TB/s | 3.35 TB/s |
| NVLink BW | 600 GB/s | 900 GB/s |
| TDP | 400 W | 700 W |

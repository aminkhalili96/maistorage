# NVIDIA A100 Tensor Core GPU -- Technical Reference

The NVIDIA A100 is a data center GPU based on the Ampere architecture, introduced in 2020 as the successor to the V100 (Volta). It was the first GPU to implement Multi-Instance GPU (MIG) technology and structural sparsity support. The A100 remains widely deployed in enterprise and cloud environments for both training and inference workloads.

## Architecture

The A100 is built on the GA100 GPU die, fabricated on TSMC 7nm (N7) process technology with approximately 54 billion transistors. The full die contains 128 SMs; the product ships with 108 SMs enabled. Each SM contains 64 FP32 CUDA cores and 4 third-generation Tensor Cores, yielding 6912 CUDA cores and 432 Tensor Cores.

The Ampere architecture introduced third-generation Tensor Cores with TF32 (TensorFloat-32) support, enabling drop-in acceleration of FP32 matrix operations without code changes. Ampere also introduced 2:4 structural sparsity, where the hardware skips zero-valued operands to double throughput for pruned models.

## Compute Performance

| Precision | TFLOPS (Dense) | TFLOPS (Sparse) |
|---|---|---|
| TF32 Tensor Core | 156 | 312 |
| FP16 / BF16 Tensor Core | 312 | 624 |
| FP64 Tensor Core | 19.5 | -- |
| FP32 (non-Tensor) | 19.5 | -- |
| INT8 Tensor Core | 624 | 1248 |
| INT4 Tensor Core | 1248 | 2496 |

The A100 does not support FP8 precision; that was introduced with the H100 (Hopper).

## Memory Subsystem

The A100 is available in two memory configurations:

| Specification | A100 40GB | A100 80GB |
|---|---|---|
| Memory Type | HBM2e | HBM2e |
| Capacity | 40 GB | 80 GB |
| Bandwidth | 1.6 TB/s | 2 TB/s |
| Memory Interface | 5120-bit | 5120-bit |

The 80 GB variant is the standard configuration for current deployments. The A100 includes 40 MB of L2 cache (80 GB variant; 20 MB on the 40 GB variant), reducing HBM traffic for workloads with data reuse patterns.

## Interconnect

NVLink 3.0 provides 600 GB/s bidirectional bandwidth through 12 links per GPU. Within a DGX A100 or HGX A100 baseboard, all eight GPUs are connected via NVSwitch 2.0 in a fully non-blocking topology.

The SXM4 variant uses the SXM4 socket. The PCIe variant connects via PCIe Gen4 x16, providing 64 GB/s bidirectional bandwidth (32 GB/s per direction).

GPUDirect RDMA is supported for direct data transfers between the GPU and InfiniBand network adapters. GPUDirect Storage is supported for direct transfers between GPU memory and NVMe storage.

## Multi-Instance GPU (MIG)

The A100 was the first GPU to support MIG, which partitions a single GPU into up to 7 isolated instances. Each MIG instance has dedicated SMs, memory controllers, L2 cache slices, and HBM memory. Instances are hardware-isolated, meaning a fault or workload spike in one instance does not affect others.

MIG instance profiles on the A100 80GB range from 1g.10gb (14 SMs, 10 GB) to 7g.80gb (98 SMs, 80 GB). MIG is particularly useful for inference workloads where a full A100 would be underutilized by a single model.

## Thermal and Power

| Variant | TDP | Cooling |
|---|---|---|
| SXM4 | 400 W | Passive heatsink, system-level airflow |
| PCIe Gen4 | 300 W | Dual-slot passive heatsink |

The A100 SXM4's 400 W TDP is significantly lower than the H100 SXM5's 700 W, making it deployable in standard air-cooled server platforms without liquid cooling.

## Form Factors and System Integration

- **SXM4**: Used in DGX A100 (8x A100 SXM4, 640 GB total GPU memory in the 80 GB configuration) and HGX A100 baseboards. Full NVLink/NVSwitch connectivity.
- **PCIe Gen4**: Standard dual-slot card for mainstream servers. Available in both 40 GB and 80 GB variants. Some PCIe configurations support NVLink bridges between pairs of GPUs (2-way NVLink).

## Structural Sparsity (2:4)

The A100 introduced hardware support for 2:4 structured sparsity, where exactly 2 out of every 4 weights are zero. The Tensor Core datapath compresses sparse matrices and computes only on non-zero values, doubling effective throughput. Achieving 2:4 sparsity requires pruning via NVIDIA's ASP (Automatic SParsity) library in Apex.

## Use Cases

- **Training and inference**: The A100 is a general-purpose data center GPU suitable for both training and inference. It was the standard GPU for training GPT-3-class models and remains in widespread production use.
- **HPC and scientific computing**: FP64 Tensor Core support (19.5 TFLOPS) provides strong double-precision performance for numerical simulation.
- **Multi-tenant inference**: MIG allows cloud providers to partition a single A100 across multiple customers or workloads with hardware isolation.
- **Enterprise AI platforms**: Widely supported by all major cloud providers (AWS p4d/p4de, GCP A2, Azure ND A100 v4) and by enterprise AI platforms including NVIDIA AI Enterprise.

## Comparison to Successor (H100)

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

The A100 remains cost-effective for workloads that do not require FP8 precision or the H100's higher memory bandwidth.

# NVIDIA L40S GPU -- Technical Reference

The NVIDIA L40S is a data center GPU based on the Ada Lovelace architecture, designed for inference, visual computing, and video processing workloads. Unlike the H100 and A100, the L40S uses GDDR6 memory instead of HBM and connects via PCIe rather than SXM, making it deployable in standard server platforms without specialized baseboard designs or liquid cooling.

## Architecture

The L40S is built on the AD102 GPU die, the largest die in the Ada Lovelace family. It contains 142 streaming multiprocessors (SMs), each with 128 CUDA cores and 4 fourth-generation Tensor Cores, yielding 18176 CUDA cores and 568 Tensor Cores. The AD102 die is fabricated on TSMC 4N process technology.

The L40S shares the same fourth-generation Tensor Core architecture as the H100, including FP8 precision and 2:4 structural sparsity support. The Ada Lovelace architecture also includes third-generation RT cores and NVENC/NVDEC media engines for professional visualization, video transcoding, and VDI workloads.

## Compute Performance

| Precision | TFLOPS (Dense) | TFLOPS (Sparse) |
|---|---|---|
| FP8 Tensor Core | 733 | 1466 |
| FP16 / BF16 Tensor Core | 366.4 | 733 |
| TF32 Tensor Core | 183.2 | 366 |
| FP32 (non-Tensor) | 91.6 | -- |
| INT8 Tensor Core | 733 | 1466 |

The H100 SXM5 delivers 3958 FP8 sparse TFLOPS, approximately 2.7x the L40S, but the L40S operates at half the power (350 W vs 700 W) and a lower price point.

## Memory Subsystem

| Specification | Value |
|---|---|
| Memory Type | GDDR6 with ECC |
| Capacity | 48 GB |
| Bandwidth | 864 GB/s |
| Memory Interface | 384-bit |

The L40S uses GDDR6 rather than HBM. GDDR6 has lower bandwidth per unit compared to HBM3 (the H100's 3.35 TB/s is approximately 3.9x the L40S's 864 GB/s), but it is significantly cheaper to implement. The 48 GB capacity is sufficient for serving most production inference models up to approximately 25B parameters in FP16, or larger models with quantization (INT8, INT4, FP8).

ECC is enabled on the GDDR6 memory, providing data integrity for enterprise and data center deployments.

## Interconnect

The L40S connects via **PCIe Gen4 x16**, providing 64 GB/s bidirectional bandwidth (32 GB/s per direction).

The L40S does **not** support NVLink. Multi-GPU communication goes through PCIe, making it less suitable for distributed training with large model parallelism. For inference workloads, the lack of NVLink is typically not a limitation because each GPU serves independent requests.

## Media Engine

The L40S includes hardware encode and decode units:

- **NVENC**: 3 encode engines, supporting H.264, H.265 (HEVC), and AV1 encoding
- **NVDEC**: 3 decode engines, supporting H.264, H.265, VP9, and AV1 decoding

These hardware engines operate independently of the CUDA cores, allowing simultaneous AI inference and video transcoding.

## Thermal and Power

| Specification | Value |
|---|---|
| TDP | 350 W |
| Form Factor | Dual-slot, full-height, full-length PCIe card |
| Cooling | Passive heatsink (requires system airflow) |

The 350 W TDP is compatible with standard air-cooled 2U and 4U server chassis without special cooling infrastructure. A 4U server can accommodate up to 8 L40S GPUs, providing 384 GB of total GPU memory.

## Form Factor and Deployment

The L40S is a standard PCIe add-in card that fits in any server with PCIe Gen4 x16 slots and adequate power delivery and cooling. No specialized SXM baseboard or NVSwitch fabric is required. It is deployable in existing enterprise server infrastructure from Dell, HPE, Supermicro, Lenovo, and other OEMs.

## Use Cases

- **Inference**: The primary target workload. The L40S provides strong FP8 and INT8 inference throughput at a lower total cost of ownership than the H100. Ideal for deploying transformer models (7B-25B parameters) in production.
- **Video processing and transcoding**: The triple NVENC/NVDEC engines support high-density video transcoding pipelines. AI-augmented video processing (super-resolution, denoising, content analysis) can run on the CUDA/Tensor Cores simultaneously with hardware transcode.
- **Virtual desktop infrastructure (VDI)**: The combination of graphics rendering (RT cores), video encode (NVENC), and AI inference makes the L40S suitable for GPU-accelerated virtual desktops with AI features.
- **Graphics rendering**: RT cores support real-time ray tracing for professional visualization and design workloads (Omniverse, CAD rendering).
- **Edge and on-premises inference**: The standard PCIe form factor and moderate power requirements make the L40S deployable in edge data centers and on-premises server rooms without specialized infrastructure.

## Comparison to H100

| Metric | L40S | H100 SXM5 |
|---|---|---|
| Architecture | Ada Lovelace | Hopper |
| CUDA Cores | 18176 | 16896 |
| Tensor Cores | 568 | 528 |
| FP8 Sparse TFLOPS | 1466 | 3958 |
| Memory | 48 GB GDDR6 | 80 GB HBM3 |
| Memory BW | 864 GB/s | 3.35 TB/s |
| NVLink | No | 900 GB/s |
| TDP | 350 W | 700 W |
| Form Factor | PCIe dual-slot | SXM5 |

The L40S has more CUDA and Tensor Cores than the H100 by count, but the H100's higher clock speeds and 3.9x memory bandwidth result in roughly 2.7x more FP8 TFLOPS. The L40S is the cost-effective choice for inference-only deployments in standard enterprise server environments without the infrastructure investment required for HGX/DGX platforms.

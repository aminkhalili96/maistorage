# NVIDIA B200 Tensor Core GPU -- Technical Reference

The NVIDIA B200 is the flagship data center GPU built on the **Blackwell architecture** (2024), succeeding the Hopper-based H100 and H200. Designed for the era of trillion-parameter foundation models and real-time large language model inference, the B200 delivers a generational leap in AI compute density through its revolutionary two-die design, 5th-generation Tensor Cores with native FP4 precision, and NVLink 5.0 interconnect. It serves as the primary compute building block in NVIDIA's GB200 NVL72 rack-scale AI supercomputer platform.

## Architecture

The B200 is fabricated on TSMC's **4NP process node** and contains approximately **208 billion transistors** across a **two-die design**. The two GPU dies are connected via a **10 TB/s chip-to-chip interconnect**, presenting a single unified GPU to software. This dual-die approach enables NVIDIA to push transistor counts and compute density beyond the reticle limits of current lithography.

The GPU features an increased Streaming Multiprocessor (SM) count compared to Hopper, with a substantial uplift in both CUDA cores and 5th-generation Tensor Cores. The 5th-gen Tensor Cores introduce native **FP4 (4-bit floating point)** support, enabling a new tier of inference throughput for quantized large language models. The second-generation **Transformer Engine** automatically selects between FP4, FP8, FP16, and BF16 precision on a per-layer basis, maximizing throughput while preserving model accuracy.

Additional architectural innovations include a hardware **decompression engine** that accelerates database query processing by decompressing data at GPU memory bandwidth speeds, and an enhanced **RAS (Reliability, Availability, Serviceability) engine** with dedicated diagnostics hardware for proactive fault detection in large-scale deployments. A new **Secure AI** feature provides hardware-rooted confidential computing for multi-tenant GPU infrastructure.

## Compute Performance

| Precision | TFLOPS (Dense) | TFLOPS (Sparse) |
|-----------|---------------|-----------------|
| FP4       | 9,000         | 18,000          |
| FP8       | 4,500         | 9,000           |
| FP16 / BF16 | 2,250      | 4,500           |
| TF32      | 1,125         | 2,250           |
| FP64      | 90            | --              |
| INT8      | 4,500         | 9,000           |

FP4 sparse performance of 18 petaFLOPS per GPU represents a **5x improvement** over the H100's peak FP8 sparse throughput, reflecting both the architectural uplift and the new precision tier. The second-generation Transformer Engine orchestrates mixed-precision execution across Tensor Core operations, dynamically selecting the narrowest safe precision for each layer of a transformer model.

## Memory Subsystem

The B200 is equipped with **192 GB of HBM3e** (High Bandwidth Memory 3e), delivering **8 TB/s** of memory bandwidth. This represents a 2.7x bandwidth improvement over the H100's 3.35 TB/s and a significant capacity increase over the H100's 80 GB. The expanded memory capacity and bandwidth are critical for serving trillion-parameter models, enabling larger batch sizes during inference and reducing the number of GPUs needed for model-parallel training of frontier-scale models. Compared to the H200 (141 GB HBM3e at 4.8 TB/s), the B200 provides 36% more capacity and 67% more bandwidth.

## Interconnect

The B200 introduces **NVLink 5.0**, providing **1,800 GB/s** of bidirectional bandwidth per GPU through **18 NVLink links** at 100 GB/s each. This doubles the per-GPU NVLink bandwidth of the H100 (900 GB/s via NVLink 4.0) and enables high-speed all-to-all communication across large GPU clusters.

**NVSwitch 4.0** connects all GPUs within a node and across nodes in NVLink domain configurations, enabling flat memory access across up to 576 GPUs at full NVLink bandwidth. The B200 also supports **PCIe Gen6** for host connectivity, doubling the host interface bandwidth over Gen5.

**GPUDirect RDMA** enables direct data transfer between the GPU and third-party network adapters (InfiniBand, Ethernet) without CPU involvement. **GPUDirect Storage** provides a direct path between GPU memory and NVMe/NFS storage, bypassing the CPU bounce buffer and enabling up to 12.5 GB/s per GPU of direct storage bandwidth for checkpoint and data loading workloads.

## Multi-Instance GPU (MIG)

The B200 supports **Multi-Instance GPU** partitioning with up to **7 isolated GPU instances**, each with dedicated compute, memory, and cache resources. MIG enables secure multi-tenant sharing of a single B200 for inference workloads, with each instance providing hardware-level isolation, independent error containment, and separate performance monitoring. Each MIG instance can run its own CUDA application, container, or virtual machine with guaranteed quality of service.

## Thermal and Power

The B200 has a **Thermal Design Power (TDP) of approximately 1,000 W** in the SXM form factor, reflecting the increased transistor count and compute density of the two-die design. An air-cooled variant is available at a reduced TDP of approximately **700 W** with correspondingly lower sustained boost clocks.

For rack-scale deployments such as the GB200 NVL72, **liquid cooling is required** and is the primary thermal solution. NVIDIA's reference liquid cooling design uses direct-to-chip cold plates with facility water loops, enabling sustained operation at full TDP in dense rack configurations. Data center infrastructure planning should account for 1,000 W per GPU plus associated networking and CPU power overhead.

## Form Factors and System Integration

The B200 is available in the **SXM6 form factor**, designed for maximum-bandwidth NVLink mesh configurations. Key system-level integration points include:

- **GB200 NVL72**: NVIDIA's flagship rack-scale AI supercomputer, integrating **72 B200 GPUs** and **36 Grace CPUs** (each Grace paired with two B200s as a GB200 Superchip) in a single liquid-cooled rack. The NVL72 provides **13.5 TB of aggregate HBM3e** memory and 576 TB/s of total NVLink bandwidth across the rack, operating as a single logical GPU domain for model parallelism. The system targets training runs for trillion-parameter foundation models and real-time inference at massive concurrency.
- **DGX B200**: NVIDIA's 8-GPU server node based on the B200, providing a self-contained AI training and inference platform with integrated networking, storage, and system management software.
- **HGX B200**: The OEM-facing 8-GPU baseboard for partner system builders, enabling custom server designs around the B200 SXM module.

## Use Cases

- **Frontier LLM training**: The B200's compute density and NVLink 5.0 bandwidth enable efficient scaling of training runs for models exceeding one trillion parameters, reducing time-to-train and cluster size requirements versus H100-based systems.
- **Real-time LLM inference**: FP4 Tensor Core support and 192 GB HBM3e enable serving large models (70B+ parameters) at high throughput and low latency on fewer GPUs. The second-generation Transformer Engine automates FP4 quantization for inference without manual calibration.
- **High-performance computing**: 90 TFLOPS of FP64 compute supports traditional HPC workloads in computational fluid dynamics, molecular dynamics, and climate modeling.
- **Scientific AI and digital twins**: Combined FP64 precision for simulation with AI acceleration for surrogate models and physics-informed neural networks.
- **Recommender systems and database acceleration**: The hardware decompression engine accelerates analytics queries, while large HBM3e capacity supports embedding tables for trillion-row recommendation models.

## Comparison to Predecessor (H100)

| Specification           | H100 SXM         | B200 SXM          |
|------------------------|-------------------|--------------------|
| Architecture           | Hopper            | Blackwell          |
| Process node           | TSMC 4N           | TSMC 4NP           |
| Transistors            | 80 billion        | 208 billion        |
| Die design             | Monolithic         | Two-die (10 TB/s link) |
| Tensor Core generation | 4th gen           | 5th gen            |
| FP4 support            | No                | Yes                |
| FP8 sparse (TFLOPS)    | 3,958             | 9,000              |
| FP16/BF16 sparse (TFLOPS) | 1,979          | 4,500              |
| FP64 dense (TFLOPS)    | 67                | 90                 |
| Memory                 | 80 GB HBM3        | 192 GB HBM3e       |
| Memory bandwidth       | 3.35 TB/s         | 8 TB/s             |
| NVLink bandwidth       | 900 GB/s          | 1,800 GB/s         |
| NVLink generation      | 4.0 (18x 50 GB/s) | 5.0 (18x 100 GB/s) |
| PCIe generation        | Gen5              | Gen6               |
| TDP                    | 700 W             | 1,000 W            |

The B200 delivers approximately **2.3x the FP8 sparse throughput**, **2.4x the memory capacity**, **2.4x the memory bandwidth**, and **2x the NVLink bandwidth** versus the H100, representing a full-generational improvement in AI training and inference capability per GPU.

# NVIDIA Grace Hopper Superchip (GH200) -- Technical Reference

The NVIDIA Grace Hopper Superchip (GH200) is a CPU-GPU co-designed module that integrates the NVIDIA Grace ARM CPU and the NVIDIA Hopper H100 GPU onto a single unified platform. By combining a high-performance ARM Neoverse V2 processor with a full Hopper-architecture GPU die connected via NVLink Chip-to-Chip (NVLink-C2C), the GH200 eliminates the traditional PCIe bottleneck between CPU and GPU. This architecture delivers up to 7x more CPU-GPU bandwidth than PCIe Gen5, enabling coherent access to a combined 576 GB memory pool. The GH200 is purpose-built for large-scale AI inference, high-performance computing, recommender systems, and workloads with large working sets that span CPU and GPU memory.

## Architecture

The GH200 pairs a 72-core NVIDIA Grace CPU based on ARM Neoverse V2 with a full GH100 GPU die -- the same Hopper silicon used in the H100 SXM5. The two processors are connected by NVIDIA NVLink Chip-to-Chip (NVLink-C2C), a custom high-bandwidth coherent interconnect that provides 900 GB/s bidirectional bandwidth between CPU and GPU.

NVLink-C2C enables a coherent unified memory architecture: the Grace CPU can directly access GPU HBM3 memory at NVLink speeds, and the GPU can access CPU LPDDR5X memory without explicit data copies or staging through PCIe. This coherent memory model simplifies programming for workloads where data structures exceed GPU memory capacity, as the hardware handles page migration and cache coherence transparently.

## CPU Specifications

The Grace CPU features 72 ARM Neoverse V2 cores with support for ARMv9 instructions, SVE2 vector extensions, and Confidential Compute Architecture (CCA). Key specifications:

- **Cores**: 72 ARM Neoverse V2
- **ISA**: ARMv9-A with SVE2
- **Memory**: 480 GB LPDDR5X with ECC
- **Memory bandwidth**: 546 GB/s
- **I/O**: PCIe Gen5 x16 lanes
- **L3 cache**: 117 MB shared
- **Process**: TSMC 4nm

The Grace CPU delivers up to 740 SPECrate2017_int_base, competitive with high-end x86 server processors while consuming significantly less power for memory-intensive workloads due to the LPDDR5X memory subsystem.

## GPU Specifications

The GH200 uses the same GH100 GPU die as the H100 SXM5, providing identical GPU compute capability:

- **GPU architecture**: NVIDIA Hopper (GH100)
- **Streaming multiprocessors (SMs)**: 132
- **CUDA cores**: 16,896
- **Tensor Cores**: 528 (4th generation)
- **Tensor Core precisions**: FP8, FP16, BF16, TF32, FP64, INT8
- **Transformer Engine**: Yes (automatic mixed-precision FP8/FP16)
- **GPU memory**: 96 GB HBM3 at 4 TB/s (standard) or 144 GB HBM3e (extended variant)
- **Process**: TSMC 4nm

The 4th-generation Tensor Cores include native FP8 support and the Transformer Engine, which dynamically selects between FP8 and FP16 precision on a per-layer basis to accelerate transformer model training and inference without accuracy loss.

## Compute Performance

| Precision | Tensor Core (TFLOPS) | Tensor Core + Sparsity (TFLOPS) |
|-----------|---------------------:|--------------------------------:|
| FP8       | 1,979                | 3,958                           |
| FP16 / BF16 | 990               | 1,979                           |
| TF32      | 495                  | 989                             |
| FP64      | 67                   | --                              |
| FP64 Tensor Core | 67            | --                              |
| INT8      | 1,979                | 3,958                           |

GPU compute performance is identical to the H100 SXM5. The GH200 advantage is not in raw GPU FLOPS but in the unified memory architecture and CPU-GPU interconnect bandwidth.

## Memory Architecture

The GH200 implements a unified memory model that combines CPU and GPU memory into a single addressable pool:

| Component       | Capacity | Bandwidth   | Technology |
|-----------------|----------|-------------|------------|
| GPU memory      | 96 GB    | 4,000 GB/s  | HBM3       |
| CPU memory      | 480 GB   | 546 GB/s    | LPDDR5X    |
| **Total**       | **576 GB** | --        | --         |

NVLink-C2C provides 900 GB/s bidirectional coherent bandwidth between the CPU and GPU memory domains. The Grace CPU can access GPU HBM3 at NVLink speeds, and the GPU can access CPU LPDDR5X memory transparently. This eliminates the need for explicit cudaMemcpy operations for many workloads and enables out-of-core GPU computing where model parameters or datasets exceed GPU memory.

The 144 GB HBM3e variant increases total addressable memory to 624 GB, further expanding the working set capacity for large language model inference and graph analytics.

## Interconnect

| Interconnect         | Bandwidth              | Purpose                          |
|----------------------|------------------------|----------------------------------|
| NVLink-C2C           | 900 GB/s bidirectional | CPU-GPU coherent link            |
| NVLink 4.0           | 900 GB/s bidirectional | GPU-to-GPU (multi-GPU systems)   |
| PCIe Gen5            | 128 GB/s bidirectional | I/O, storage, networking         |
| ConnectX-7 InfiniBand| 400 Gb/s               | Inter-node networking            |

The NVLink-C2C interconnect is the key differentiator of the GH200 architecture. At 900 GB/s bidirectional, it delivers approximately 7x the bandwidth of a PCIe Gen5 x16 link (128 GB/s bidirectional). Unlike PCIe, NVLink-C2C is a coherent interconnect: both CPU and GPU maintain cache coherence across the link, enabling fine-grained shared data structures without software-managed synchronization.

## Thermal and Power

| Configuration            | Total Module Power |
|--------------------------|--------------------|
| Air-cooled GH200         | ~500 W             |
| Liquid-cooled GH200      | ~700 W             |

In the air-cooled configuration, the Grace CPU consumes approximately 200 W and the GPU approximately 300 W. The liquid-cooled variant allows the GPU to operate at higher sustained clock frequencies, increasing total module power to approximately 700 W. Compared to a discrete system with a separate x86 CPU (250-350 W) and H100 SXM5 GPU (700 W), the air-cooled GH200 achieves comparable GPU performance at roughly half the total system power, delivering a significantly better performance-per-watt ratio.

## Form Factors

- **GH200 module**: Single superchip module for OEM server integration
- **NVIDIA MGX reference design**: Modular server reference architecture supporting GH200 modules with standardized mechanical and thermal interfaces
- **DGX GH200**: Large-scale system connecting 256 GH200 superchip modules via the NVLink Switch System, creating a shared memory domain of up to 144 TB across all modules

## Use Cases

The GH200 is optimized for workloads that benefit from large unified memory pools and high CPU-GPU bandwidth:

- **LLM inference**: The 576 GB unified memory pool enables serving large language models (70B+ parameters) that exceed GPU HBM capacity, with NVLink-C2C providing high-bandwidth access to model weights stored in CPU memory
- **Recommender systems**: Embedding tables spanning hundreds of gigabytes can reside in the unified memory space without manual partitioning between CPU and GPU
- **Graph analytics**: Large-scale graph processing with irregular memory access patterns benefits from coherent CPU-GPU memory
- **Digital twins and simulation**: HPC workloads with large state spaces leverage the combined memory capacity
- **HPC applications**: Scientific computing codes with mixed CPU and GPU phases avoid PCIe transfer overhead
- **Edge AI inference**: Power-efficient ARM CPU combined with full Hopper GPU for high-throughput inference at the edge

## Comparison: GH200 vs Discrete H100 + x86 System

| Feature                 | GH200 Superchip         | H100 SXM5 + x86 CPU           |
|-------------------------|--------------------------|--------------------------------|
| Total memory            | 576 GB (480 + 96)        | ~288 GB (256 DDR5 + 32 L2/system) + 80 GB HBM3 |
| GPU memory              | 96 GB HBM3               | 80 GB HBM3                     |
| CPU-GPU bandwidth       | 900 GB/s (NVLink-C2C)    | 128 GB/s (PCIe Gen5 x16)       |
| Memory coherence        | Hardware-coherent         | Software-managed (cudaMemcpy)  |
| Total system power      | ~500 W (air-cooled)       | ~950-1050 W (CPU + GPU)        |
| GPU compute (FP8)       | 3,958 TFLOPS (sparse)    | 3,958 TFLOPS (sparse)          |
| CPU architecture        | ARM Neoverse V2 (72c)    | x86-64 (varies)                |

The GH200 delivers identical GPU compute performance to the discrete H100 SXM5 while providing 7x higher CPU-GPU interconnect bandwidth, hardware memory coherence, and approximately 50% lower total system power consumption. The unified memory architecture is particularly advantageous for inference workloads where model parameters exceed GPU memory capacity.

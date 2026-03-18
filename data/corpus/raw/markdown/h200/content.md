# NVIDIA H200 Tensor Core GPU -- Technical Reference

The NVIDIA H200 is a memory-optimized variant of the H100, sharing the same Hopper architecture and compute configuration but equipped with HBM3e memory. The H200 delivers 141 GB of memory capacity and 4.8 TB/s of bandwidth, making it the highest-memory-bandwidth data center GPU in the Hopper family. It is positioned primarily for large language model inference workloads where memory capacity and bandwidth are the primary bottlenecks.

## Architecture

The H200 uses the same GH100 GPU die as the H100. All compute specifications are identical: 132 SMs, 16896 CUDA cores, 528 fourth-generation Tensor Cores, and the Hopper Transformer Engine with FP8 support. The architectural differentiation is entirely in the memory subsystem.

## Compute Performance

Compute throughput is identical to the H100 SXM5:

| Precision | TFLOPS (Dense) | TFLOPS (Sparse) |
|---|---|---|
| FP8 Tensor Core | 1979 | 3958 |
| FP16 / BF16 Tensor Core | 990 | 1979 |
| TF32 Tensor Core | 495 | 989 |
| FP64 Tensor Core | 67 | -- |
| INT8 Tensor Core | 1979 | 3958 |

The performance advantage of the H200 over the H100 comes not from higher TFLOPS but from the ability to keep the Tensor Cores fed with data at a higher rate due to the increased memory bandwidth, and from the ability to hold larger models or larger batch sizes in memory.

## Memory Subsystem

This is the defining difference between the H200 and H100.

| Specification | H100 SXM5 | H200 SXM |
|---|---|---|
| Memory Type | HBM3 | HBM3e |
| Capacity | 80 GB | 141 GB |
| Bandwidth | 3.35 TB/s | 4.8 TB/s |

The H200 provides 76% more memory capacity and 43% more memory bandwidth than the H100. HBM3e achieves higher bandwidth per pin than HBM3 through improved signaling, while the capacity increase comes from using higher-density memory stacks.

For inference workloads, memory capacity determines the maximum model size that can be served without tensor parallelism across multiple GPUs. The 141 GB capacity allows the H200 to serve a 70B-parameter model (approximately 140 GB in FP16) on a single GPU, whereas the H100 would require at least two GPUs for the same model.

Memory bandwidth directly determines tokens-per-second throughput during autoregressive decoding. The 43% bandwidth increase translates directly to higher inference throughput for memory-bound workloads.

## Interconnect

NVLink 4.0: 900 GB/s bidirectional bandwidth through 18 links, identical to the H100 SXM5. Full NVSwitch 3.0 connectivity is available in HGX H200 8-GPU baseboards.

PCIe Gen5 x16: 128 GB/s bidirectional, identical to the H100.

GPUDirect RDMA and GPUDirect Storage are supported.

## Multi-Instance GPU (MIG)

MIG support is identical to the H100: up to 7 hardware-isolated instances. Each instance receives a proportional share of the larger 141 GB memory pool, making each MIG slice more capable of serving medium-sized models independently.

## Thermal and Power

The H200 is available in the SXM form factor with a 700 W TDP, identical to the H100 SXM5. Cooling requirements are the same: passive heatsink with system-level liquid or air cooling in HGX-based server platforms.

## Form Factors and System Integration

The H200 is currently available in the SXM form factor for integration into HGX H200 baseboards and DGX H200 systems. It is a drop-in replacement for the H100 SXM5 in existing HGX baseboard designs, requiring no mechanical or electrical changes to the server platform.

DGX H200 systems contain 8x H200 GPUs for a total of 1128 GB (1.1 TB) of HBM3e memory per node, compared to 640 GB in the DGX H100.

## Inference Performance

NVIDIA-published benchmarks show the following inference speedups for the H200 over the H100 on representative LLM workloads:

| Model | Speedup vs H100 |
|---|---|
| Llama 2 70B | ~1.9x |
| GPT-3 175B | ~1.9x |
| Mixtral 8x7B (MoE) | ~1.6x |

These gains come from two sources: (1) larger batch sizes fit in memory due to 141 GB capacity, improving GPU utilization, and (2) higher memory bandwidth sustains faster token generation during autoregressive decoding.

Mixture-of-experts (MoE) models benefit particularly from the larger memory capacity because MoE architectures have large total parameter counts but activate only a subset of experts per token. The full model must reside in memory even though only a fraction is computed on each forward pass.

## Use Cases

- **LLM inference at scale**: The primary target workload. Memory capacity and bandwidth are the bottlenecks for autoregressive decoding throughput.
- **Large KV-cache workloads**: Long-context inference (32K+ token sequences) generates large key-value caches. The 141 GB capacity allows serving longer contexts or larger batch sizes before running out of memory.
- **Mixture-of-experts models**: MoE models like Mixtral have high memory footprints relative to their active compute. The H200's memory capacity makes single-GPU or fewer-GPU serving feasible.
- **Training**: While the H200 can train models, the compute is identical to the H100. The memory capacity advantage matters for training only when model states exceed 80 GB per GPU, which is relevant for very large models with limited tensor parallelism.

## Positioning Relative to H100

The H200 is not a replacement for the H100 in all scenarios. For compute-bound training workloads, the H100 and H200 deliver identical performance. The H200's value is in memory-bandwidth-bound inference and in reducing GPU count to serve large models.

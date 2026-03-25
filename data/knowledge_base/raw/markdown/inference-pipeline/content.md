# End-to-End AI Inference Pipeline Guide

Deploying a trained PyTorch model to a production inference endpoint requires a multi-stage pipeline: model export, graph optimization, serving infrastructure configuration, and ongoing monitoring. Each stage introduces tradeoffs between latency, throughput, cost, and operational complexity. This guide covers the complete workflow from a trained checkpoint to a production-ready endpoint on NVIDIA GPU infrastructure, with emphasis on the optimization and serving decisions that determine real-world performance.

## Model Export and Conversion

The first step is exporting the trained PyTorch model to ONNX (Open Neural Network Exchange), an intermediate representation that decouples the model from the training framework. PyTorch provides `torch.onnx.export()` for this conversion, which traces the model's forward pass and serializes the computation graph.

Key considerations during export:

- **Operator support**: Most standard PyTorch operators have ONNX equivalents, but custom operations (custom CUDA kernels, non-standard activations) may require registering custom op converters or rewriting layers with supported primitives.
- **Dynamic axes**: Production models typically handle variable batch sizes and sequence lengths. Specify dynamic axes during export (e.g., `dynamic_axes={"input": {0: "batch", 1: "sequence"}}`) to avoid baking fixed dimensions into the graph.
- **Opset version**: Use the latest supported opset version (typically opset 17+) for maximum operator coverage and optimization potential downstream.
- **Numerical validation**: After export, run identical inputs through both the PyTorch model and the ONNX model (via `onnxruntime`) and compare outputs. Acceptable tolerance is typically `atol=1e-5` for FP32. Discrepancies beyond this threshold indicate export errors or unsupported operator semantics.

## TensorRT Optimization

TensorRT converts the ONNX graph into a hardware-optimized inference engine. The `trtexec` command-line tool handles the build process: `trtexec --onnx=model.onnx --saveEngine=model.plan --fp16`.

Core optimizations that TensorRT applies:

- **Layer fusion**: Combines adjacent operations (Conv + BatchNorm + ReLU) into single kernels, reducing memory bandwidth and kernel launch overhead.
- **Kernel auto-tuning**: Profiles multiple kernel implementations for each operation on the target GPU and selects the fastest.
- **Precision calibration**: Supports FP16 (nearly lossless, 2x speedup on Tensor Cores), INT8 (requires a calibration dataset of 500-1000 representative samples to determine quantization scales), and FP4 on Blackwell architecture GPUs for maximum throughput.

TensorRT engines are hardware-specific: an engine built on an H100 will not run on an L40S. Build engines on the target deployment GPU, and version them alongside the model artifacts. Build times range from minutes (small CNNs) to hours (large transformers).

For large language models, **TensorRT-LLM** provides transformer-specific optimizations: KV cache management, paged attention (reducing memory fragmentation), in-flight batching, and tensor parallelism across multiple GPUs. These optimizations are critical for achieving acceptable token throughput on models with billions of parameters.

## Batching Strategies

Batching is the single largest lever for inference throughput:

- **Static batching**: Fixed batch size, padded to uniform length. Simple to implement but wastes GPU cycles on padding tokens, especially with variable-length sequences. Suitable for fixed-size inputs (image classification).
- **Dynamic batching**: Triton Inference Server groups incoming requests into batches up to `max_batch_size`, waiting up to `max_queue_delay_microseconds` for a full batch. Balances latency and throughput automatically.
- **Continuous (in-flight) batching**: Used by TensorRT-LLM and vLLM. Batches at the token level rather than the request level. When one sequence in a batch completes, a new request immediately fills that slot. Essential for LLM serving where sequences complete at vastly different times.
- **Sequence bucketing**: Groups sequences into length buckets (e.g., 128, 256, 512, 1024 tokens) and pads only within each bucket. Reduces wasted computation from padding by 30-60% compared to padding all sequences to the maximum length.

## Triton Model Repository

Triton Inference Server loads models from a structured repository:

```
model_repository/
  my_model/
    config.pbtxt
    1/
      model.plan      # TensorRT engine
    2/
      model.plan      # Updated version
```

The `config.pbtxt` file specifies the model's serving configuration: platform (e.g., `tensorrt_plan`), `max_batch_size`, input and output tensor names with shapes and data types, instance groups (how many model copies per GPU), and dynamic batching parameters.

**Model warm-up**: Configure warm-up requests in the model config to pre-allocate GPU memory and JIT-compile any remaining kernels before serving live traffic. This eliminates cold-start latency spikes.

**Ensemble models**: Triton supports ensemble pipelines that chain preprocessing (tokenization, image normalization), inference, and postprocessing into a single request. This avoids round-trip overhead between client and server for multi-step pipelines.

## Serving Patterns

- **Single-model serving**: One Triton or NIM instance serves a single model. Simplest operationally, best for large models that consume an entire GPU.
- **Multi-model serving**: Multiple models share a GPU via Triton instance groups. Each model gets dedicated CUDA streams. Suitable when individual models do not saturate GPU compute.
- **Model versioning**: Triton natively supports multiple model versions. Use Kubernetes traffic splitting or Istio service mesh to route a percentage of traffic to a new version (canary deployment) before full rollout.
- **A/B testing**: Deploy model variants behind a service mesh and route traffic based on headers or weighted rules. Collect per-version metrics to compare accuracy and latency before promoting.
- **Autoscaling**: Configure Kubernetes HPA (Horizontal Pod Autoscaler) on custom metrics: GPU utilization (`DCGM_FI_DEV_GPU_UTIL`), request queue depth, or inference latency percentiles. Scale-to-zero with KEDA for cost savings on bursty workloads.

## Cost and Latency Tradeoffs

| Serving Option | p99 Latency | Throughput | GPU Utilization | Operational Complexity |
|----------------|-------------|------------|-----------------|------------------------|
| NVIDIA NIM | Low | High | High | Low (managed container) |
| Triton + TensorRT | Low | Very High | Very High | Medium (config tuning) |
| vLLM | Medium | High (LLMs) | High | Low (Python-native) |
| Custom (FastAPI + torch) | High | Low | Low | High (no batching/optimization) |

**GPU selection for inference workloads**:
- **H200**: 141 GB HBM3e memory. Ideal for memory-bound LLM inference (70B+ parameter models) where KV cache size dominates.
- **L40S**: 48 GB GDDR6 with Ada Lovelace architecture. Cost-effective for medium models (7B-30B parameters) and multi-model serving.
- **T4**: 16 GB GDDR6, budget-friendly. Suitable for small models, INT8 inference, and edge deployments.

**Right-sizing**: Match model memory footprint (weights + KV cache + activation memory) to GPU memory. A 7B FP16 model requires approximately 14 GB, fitting comfortably on an L40S with room for large batch KV caches. Deploying it on an H200 wastes 90% of available memory.

**Batch size vs latency**: Increasing batch size improves throughput (tokens/second) but increases per-request latency. Profile the batch-size-latency curve for your model to find the optimal operating point given your SLA requirements.

## Multi-Model Scheduling

Running multiple models on a single GPU maximizes hardware utilization:

- **MIG (Multi-Instance GPU)**: Available on A100 and H100. Partitions a GPU into isolated instances with dedicated compute, memory, and cache. Each partition runs an independent model with hardware-level isolation. An A100 80 GB can be split into up to 7 MIG instances (1g.10gb each).
- **MPS (Multi-Process Service)**: Software-level GPU sharing that allows concurrent kernels from multiple processes. Lower isolation than MIG but supports any GPU. Useful for small models that individually underutilize the GPU.
- **Triton rate limiter**: Controls concurrent model execution on shared GPUs. Assign resource budgets per model to prevent one high-traffic model from starving others. Configure `rate_limiter { resources [{name: "R1" count: 1}] }` in each model's `config.pbtxt`.
- **Priority scheduling**: Assign higher priority to latency-sensitive interactive models and lower priority to batch workloads (embedding generation, offline scoring). Triton's priority levels (1-5) determine scheduling order when the GPU is contended.

## Monitoring Production Inference

Effective production monitoring requires tracking both serving and hardware metrics:

**Serving metrics** (exposed by Triton and NIM via Prometheus endpoints):
- **Tokens per second**: Aggregate throughput across all active requests.
- **Time-to-first-token (TTFT)**: Latency from request receipt to first generated token. Critical for interactive applications. Target: <200ms for real-time chat.
- **Inter-token latency (ITL)**: Time between consecutive generated tokens. Determines perceived streaming speed. Target: <50ms for smooth UX.
- **Queue depth**: Number of requests waiting for GPU resources. Sustained queue buildup indicates insufficient capacity.

**Hardware metrics** (collected via NVIDIA DCGM):
- GPU utilization, memory utilization, power consumption, and thermal state.
- **KV cache utilization**: Percentage of allocated KV cache memory in use. High utilization (>90%) causes request queuing or eviction.
- PCIe/NVLink bandwidth utilization for multi-GPU deployments.

**Dashboarding**: Export metrics to Prometheus and visualize in Grafana. Set alert thresholds for SLA compliance: p99 TTFT > 500ms, GPU memory > 95%, sustained queue depth > 100 for more than 60 seconds. Correlate DCGM hardware health metrics (ECC errors, thermal throttling, NVLink errors) with serving performance degradation to diagnose infrastructure-level issues before they cause outages.

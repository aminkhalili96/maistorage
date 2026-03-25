# NVIDIA TensorRT -- Inference Optimization Engine

TensorRT is NVIDIA's high-performance deep learning inference optimizer and runtime engine. It takes trained models from frameworks such as PyTorch, TensorFlow, and ONNX and produces highly optimized inference engines tuned for NVIDIA GPU architectures. TensorRT applies graph optimizations, precision calibration, layer fusion, and kernel auto-tuning to deliver the lowest possible latency and highest throughput for production inference workloads.

TensorRT is a core component of the NVIDIA AI inference stack and is used as a backend by Triton Inference Server, NVIDIA NIM, and TensorRT-LLM for large language model serving.

## Optimization Pipeline

TensorRT processes a trained model through a multi-stage optimization pipeline before producing a deployable inference engine.

**1. Model Import.** TensorRT accepts models via several parsers. The ONNX parser is the primary import path and supports the widest range of models exported from PyTorch (`torch.onnx.export`), TensorFlow, and other frameworks. Legacy parsers include UFF (deprecated) and Caffe. TF-TRT provides in-graph integration that replaces supported subgraphs within a TensorFlow graph with TensorRT-optimized nodes. Torch-TensorRT provides similar integration for PyTorch models via `torch.compile` or ahead-of-time compilation.

**2. Layer and Tensor Fusion.** TensorRT analyzes the computation graph and fuses multiple operations into single GPU kernels. Common fusion patterns include Conv+BatchNorm+ReLU, which eliminates intermediate memory reads and writes. Pointwise operations (bias addition, activation functions, residual connections) are fused into upstream kernels. These fusions can reduce kernel launch overhead by 40-60% and dramatically cut memory bandwidth consumption.

**3. Precision Calibration.** TensorRT converts model weights and activations from FP32 to lower precision formats. FP16 inference provides roughly 2x throughput improvement with minimal accuracy degradation and requires no calibration data. INT8 quantization delivers up to 4x throughput but requires a representative calibration dataset to determine per-tensor quantization scales. TensorRT supports entropy calibration (minimizes KL divergence between FP32 and INT8 distributions) and min-max calibration (uses observed tensor ranges). On Hopper GPUs and later, FP8 quantization offers INT8-class performance with simplified calibration via per-tensor scaling factors.

**4. Kernel Auto-Tuning.** TensorRT profiles hundreds of kernel implementations for each layer on the target GPU and selects the fastest. This includes choosing between different tiling strategies, thread block configurations, and algorithm variants (e.g., implicit GEMM vs. Winograd for convolutions). The tuning results are GPU-architecture-specific -- an engine optimized for A100 will not run on H100.

**5. Dynamic Tensor Memory.** TensorRT performs liveness analysis on intermediate tensors and reuses memory allocations across layers whose lifetimes do not overlap. This reduces the GPU memory footprint by 30-50% compared to framework-level inference, enabling larger batch sizes or deployment on smaller GPUs.

**6. Engine Serialization.** The final optimized engine is saved as a plan file (`.engine` or `.plan`). This file contains the fused graph, selected kernels, weight data, and memory allocation plan. Deserialization at runtime is near-instantaneous compared to the minutes-long build process.

## Quantization Deep Dive

Precision selection is the single most impactful optimization for inference performance.

**FP16** halves memory bandwidth requirements and doubles Tensor Core throughput. On A100, FP16 Tensor Core performance is 312 TFLOPS versus 156 TFLOPS for FP32 (with sparsity). Most models tolerate FP16 with no measurable accuracy loss. No calibration dataset is needed -- TensorRT simply converts weights and uses FP16 accumulation where safe.

**INT8** reduces precision to 8-bit integers, achieving up to 624 INT8 TOPS on A100. Calibration requires running 500-1000 representative input samples through the network to determine quantization scales. Entropy calibration generally produces better accuracy than min-max for models with non-uniform activation distributions. Typical accuracy degradation is less than 1% top-1 for image classification models when calibrated properly.

**FP8** (available on Hopper H100, H200, and later) provides similar throughput to INT8 (roughly 4x over FP32) but uses a floating-point representation that handles outlier values more gracefully. FP8 calibration is simpler than INT8 because the format's dynamic range reduces sensitivity to scale selection. H100 delivers 3,958 TFLOPS of FP8 Tensor Core performance.

**Mixed Precision** allows per-layer precision selection. Sensitivity analysis identifies layers where reduced precision causes unacceptable accuracy loss (typically the first and last layers of a network). TensorRT can be configured to keep sensitive layers in FP16 while running the rest in INT8, balancing throughput and accuracy.

## Key Features

**Dynamic Shapes.** TensorRT supports variable-dimension inputs via optimization profiles. Each profile specifies minimum, optimum, and maximum values for each dynamic dimension (e.g., batch size 1/8/32, sequence length 16/128/512). The builder optimizes kernels for the optimum shape while ensuring correctness across the full range. Multiple profiles can be defined for different operating regimes.

**Builder Optimization Profiles.** Users can create multiple optimization profiles within a single engine to handle different input shape ranges efficiently. This avoids rebuilding engines for different deployment scenarios.

**DLA Support.** On NVIDIA Jetson and DRIVE platforms, TensorRT can offload supported layers to the Deep Learning Accelerator (DLA), freeing the GPU for other tasks. DLA-compatible layers include convolution, deconvolution, pooling, and activation functions.

**Plugin API.** Custom layers not natively supported by TensorRT can be implemented as plugins using the C++ or Python plugin API. Plugins integrate into the TensorRT optimization pipeline and participate in fusion and precision calibration.

**Timing Cache.** TensorRT can save kernel auto-tuning results to a timing cache file, reducing rebuild time from minutes to seconds when the network architecture and GPU remain unchanged. This is critical in CI/CD pipelines.

**Strongly Typed Networks.** TensorRT 9.0+ supports strongly typed mode where users explicitly control precision at each layer, providing deterministic precision behavior instead of relying on the builder's heuristic precision selection.

## Integration Patterns

**trtexec CLI.** The `trtexec` command-line tool benchmarks and converts models without writing code:

```bash
# Convert ONNX model to TensorRT engine with FP16
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

# Benchmark with dynamic shapes
trtexec --onnx=model.onnx --minShapes=input:1x3x224x224 \
  --optShapes=input:8x3x224x224 --maxShapes=input:32x3x224x224 \
  --fp16 --saveEngine=model.engine

# INT8 with calibration cache
trtexec --onnx=model.onnx --int8 --calib=calibration.cache \
  --saveEngine=model_int8.engine
```

**C++ and Python APIs.** TensorRT provides full builder, runtime, and plugin APIs in both C++ and Python. The Python API (`import tensorrt as trt`) is the most common entry point for model conversion scripts.

**Torch-TensorRT.** Integrates TensorRT optimization into the PyTorch ecosystem via `torch.compile(model, backend="torch_tensorrt")` or ahead-of-time compilation. Unsupported operators fall back to PyTorch execution automatically.

**ONNX Runtime with TensorRT EP.** ONNX Runtime can use TensorRT as an execution provider, automatically offloading supported subgraphs to TensorRT while running the rest on the default CPU/CUDA provider.

**Triton Inference Server.** TensorRT `.plan` files are a first-class model format in Triton, offering the highest-performance backend for production serving.

## Performance Characteristics

TensorRT typically delivers 2-6x speedup over native framework inference, depending on model architecture, precision, and batch size. Convolution-heavy models (ResNet, EfficientNet) benefit most from layer fusion and INT8. Transformer models see significant gains from FP16/FP8 and attention kernel optimization.

Batch size has a major impact on throughput: increasing from batch 1 to batch 8 can improve GPU utilization 3-5x on A100, as Tensor Cores require sufficient parallelism to reach peak performance. For latency-sensitive applications, TensorRT's `BuilderFlag.PREFER_PRECISION_CONSTRAINTS` mode minimizes per-request latency at the cost of some throughput.

## Limitations and Considerations

**GPU-specific engines.** A TensorRT engine compiled for A100 (SM 8.0) cannot run on H100 (SM 9.0) or vice versa. Engines must be rebuilt for each target GPU architecture. The timing cache can accelerate rebuilds but does not eliminate them.

**Build time.** Engine compilation can take 10-60 minutes for large models, especially with INT8 calibration and multiple optimization profiles. This is a one-time cost per model version and GPU type.

**Operator coverage.** Not all ONNX operators are supported by TensorRT. Unsupported operators cause subgraph partitioning, where TensorRT optimizes supported portions and the remainder falls back to ONNX Runtime or the source framework. Common unsupported operations include certain dynamic control flow and custom attention patterns.

**LLM inference.** For large language models, TensorRT-LLM is the dedicated product. It handles KV-cache management, in-flight batching, tensor parallelism, and other LLM-specific optimizations that go beyond the scope of base TensorRT.

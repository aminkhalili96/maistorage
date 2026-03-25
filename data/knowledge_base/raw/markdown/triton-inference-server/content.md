# NVIDIA Triton Inference Server -- Model Serving Platform

Triton Inference Server is NVIDIA's open-source inference serving software that enables teams to deploy AI models from any framework on any GPU or CPU-based infrastructure. Triton supports models from TensorRT, ONNX Runtime, PyTorch, TensorFlow, vLLM, TensorRT-LLM, OpenVINO, and custom backends via a pluggable backend architecture. It provides dynamic batching, model ensembles, concurrent model execution, and GPU resource management through a unified gRPC and HTTP/REST API.

Triton is deployed widely in production environments ranging from single-GPU edge servers to multi-node GPU clusters managed by Kubernetes. It is available as a container image from NVIDIA NGC (`nvcr.io/nvidia/tritonserver`) and is the inference runtime underlying many NVIDIA NIM microservices.

## Architecture

Triton's architecture separates model management, request scheduling, and inference execution into distinct subsystems.

**Model Repository.** Models are stored in a structured directory hierarchy on local disk, S3, GCS, or Azure Blob Storage. Each model has a directory containing a `config.pbtxt` configuration file and one or more versioned subdirectories with the model artifacts.

**Backend System.** Each model is served by a pluggable backend that handles framework-specific loading and execution. The backend system abstracts framework differences so that clients interact with all models through the same API regardless of the underlying framework.

**Request Scheduler.** Incoming inference requests pass through Triton's scheduler, which applies dynamic batching, sequence batching, or direct scheduling based on the model configuration. The scheduler manages request queues, enforces priority levels, and handles request cancellation and timeouts.

**Endpoints.** Triton exposes gRPC (port 8001), HTTP/REST (port 8000), and Prometheus metrics (port 8002) endpoints. The gRPC endpoint delivers higher throughput for high-volume workloads, while the HTTP endpoint provides broader client compatibility.

## Model Repository Structure

A typical model repository follows this layout:

```
model_repository/
  resnet50/
    config.pbtxt
    1/
      model.plan
    2/
      model.plan
  text_encoder/
    config.pbtxt
    1/
      model.onnx
  preprocessing/
    config.pbtxt
    1/
      model.py
```

Each numbered subdirectory (1/, 2/) represents a model version. Triton's version policy controls which versions are loaded:

- **Latest** (default): loads the highest-numbered version only.
- **All**: loads every version, allowing clients to target specific versions.
- **Specific**: loads an explicit list of version numbers.

The `config.pbtxt` file defines the model's input/output tensor specifications, maximum batch size, instance groups, and scheduling policy:

```protobuf
name: "resnet50"
platform: "tensorrt_plan"
max_batch_size: 32
input [
  {
    name: "input"
    data_type: TYPE_FP16
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP16
    dims: [ 1000 ]
  }
]
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0, 1 ]
  }
]
dynamic_batching {
  preferred_batch_size: [ 8, 16 ]
  max_queue_delay_microseconds: 100
}
```

## Supported Backends

**TensorRT** (`.plan` / `.engine`). The highest-performance backend. TensorRT plan files are compiled for a specific GPU architecture and contain fully optimized kernels. Set `platform: "tensorrt_plan"` in config.

**ONNX Runtime** (`.onnx`). Provides broad framework compatibility. Models exported from PyTorch, TensorFlow, or scikit-learn via ONNX can be served directly. Supports TensorRT and CUDA execution providers for GPU acceleration. Set `platform: "onnxruntime_onnx"`.

**PyTorch** (TorchScript `.pt`). Serves models saved via `torch.jit.save()`. Supports CUDA execution and custom TorchScript operators. Set `platform: "pytorch_libtorch"`.

**TensorFlow** (SavedModel). Serves TensorFlow 2.x SavedModel format. Set `platform: "tensorflow_savedmodel"`.

**OpenVINO**. Optimizes and serves models on Intel CPUs and accelerators.

**Python Backend**. Executes arbitrary Python code, commonly used for preprocessing, postprocessing, tokenization, and custom inference logic. The Python backend enables building complex pipelines without writing C++ code. Model code lives in a `model.py` file implementing `initialize()`, `execute()`, and `finalize()` methods.

**vLLM Backend**. Serves large language models with continuous batching and PagedAttention memory management. Suitable for LLM workloads that require high concurrency and efficient KV-cache utilization.

**TensorRT-LLM Backend**. Serves models optimized by TensorRT-LLM with in-flight batching, KV-cache management, tensor parallelism, and pipeline parallelism. This is the highest-performance option for LLM serving on NVIDIA GPUs.

## Dynamic Batching

Dynamic batching is Triton's primary mechanism for maximizing GPU throughput. When enabled, Triton accumulates individual inference requests and combines them into a single batched execution.

Key configuration parameters in `config.pbtxt`:

- `max_batch_size`: the maximum batch dimension the model supports.
- `preferred_batch_size`: batch sizes at which Triton should execute immediately rather than waiting for more requests (e.g., `[8, 16, 32]`).
- `max_queue_delay_microseconds`: maximum time to wait for additional requests before executing an incomplete batch. Setting this to 100-500 microseconds provides a good balance between latency and throughput for most workloads.

**Sequence Batching** is used for stateful models such as recurrent neural networks or streaming ASR models. Triton correlates requests belonging to the same sequence via a correlation ID and routes them to the same model instance to preserve state.

**Continuous Batching** (also called in-flight batching) is used by the vLLM and TensorRT-LLM backends for LLM serving. Unlike static batching, new requests can join a running batch as earlier requests complete their token generation, maximizing GPU utilization during autoregressive decoding.

## Model Ensembles and Business Logic Scripting

**Ensemble Models** define multi-model pipelines as a directed acyclic graph. An ensemble chains preprocessing, one or more inference models, and postprocessing into a single logical model. Clients send one request and receive the final output without managing intermediate data transfers. Ensemble configuration is defined in `config.pbtxt` with an `ensemble_scheduling` block that specifies input-output mappings between pipeline steps.

**Business Logic Scripting (BLS)** enables more complex multi-model pipelines with conditional logic, loops, and error handling. BLS pipelines are implemented as Python backend models that programmatically invoke other models loaded in the same Triton instance via `pb_utils.InferenceRequest`. BLS is preferred over ensembles when the pipeline requires branching logic, dynamic model selection, or aggregation across multiple model outputs.

## GPU Resource Management

**Instance Groups** control how many copies of a model are loaded and on which GPUs. Setting `count: 2` on `gpus: [0, 1]` places one instance on each GPU. Setting `count: 4` on `gpus: [0]` places four instances on GPU 0, enabling concurrent execution if the GPU has sufficient memory.

**Rate Limiting and Model Priority.** Triton supports rate-limiting model instances to prevent a high-throughput model from starving latency-sensitive models of GPU resources. Priority levels (via `priority` and `priority_levels` in config) allow fine-grained resource allocation.

**Multi-GPU Serving.** For models that fit on a single GPU, Triton distributes instances across available GPUs for throughput scaling. For models requiring multiple GPUs (e.g., large LLMs), the TensorRT-LLM backend handles tensor parallelism across GPUs automatically.

**Concurrent Model Execution.** Multiple different models can share a single GPU. Triton's scheduler manages GPU memory and compute resources to allow concurrent execution, although users must ensure total GPU memory consumption across all loaded models stays within device limits.

## Deployment Patterns

**Docker.** The most common deployment method:

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models
```

**Kubernetes.** NVIDIA provides Helm charts for deploying Triton with horizontal pod autoscaling based on GPU utilization or request queue depth. Readiness probes use the `/v2/health/ready` endpoint, and liveness probes use `/v2/health/live`. The NVIDIA GPU Operator handles GPU driver and device plugin management on Kubernetes nodes.

**Model Hot-Loading.** Triton can monitor the model repository for changes and automatically load, reload, or unload models without server restart. This is enabled with `--model-control-mode=poll` and `--repository-poll-secs=30`. For explicit control, `--model-control-mode=explicit` allows loading and unloading via the model control API.

**Health Checks.** Triton exposes `/v2/health/live` (server is running) and `/v2/health/ready` (server is ready to accept inference requests, all models loaded) endpoints for orchestrator integration.

## Monitoring and Observability

**Prometheus Metrics.** Triton exposes metrics on port 8002 at `/metrics`:

- `nv_inference_request_success` / `nv_inference_request_failure`: request counts per model.
- `nv_inference_request_duration_us`: end-to-end inference latency histogram.
- `nv_inference_queue_duration_us`: time spent in the scheduling queue.
- `nv_inference_compute_infer_duration_us`: GPU compute time.
- `nv_gpu_utilization`: per-GPU utilization percentage.
- `nv_gpu_memory_used_bytes` / `nv_gpu_memory_total_bytes`: GPU memory tracking.

**Triton Model Analyzer.** A standalone tool that profiles model performance across different batch sizes, instance counts, and concurrency levels. Model Analyzer searches the configuration space and recommends optimal settings that meet latency constraints while maximizing throughput:

```bash
model-analyzer profile --model-repository /models \
  --profile-models resnet50 \
  --triton-launch-mode=docker \
  --output-model-repository-path /output
```

**Perf Analyzer.** A client-side load testing tool that measures inference performance:

```bash
perf_analyzer -m resnet50 -u localhost:8001 -i grpc \
  --concurrency-range 1:16 --measurement-interval 10000
```

Perf Analyzer reports throughput (inferences/sec), latency percentiles (p50, p90, p99), and queue time, enabling users to characterize model performance under realistic load.

## Comparison with NVIDIA NIM

Triton Inference Server and NVIDIA NIM serve different points on the ease-of-use vs. flexibility spectrum. Triton is a general-purpose serving platform: it supports any model, any framework, and any deployment topology but requires users to configure model repositories, backends, batching policies, and resource allocation. NIM provides turnkey, pre-optimized containers for specific models (Llama, Mistral, Stable Diffusion, etc.) that start serving with a single `docker run` command. NIM containers often use Triton internally as the serving runtime but hide the configuration complexity behind a simplified API. Choose Triton when you need multi-model serving, custom pipelines, or fine-grained resource control. Choose NIM when deploying a supported model with minimal configuration overhead.

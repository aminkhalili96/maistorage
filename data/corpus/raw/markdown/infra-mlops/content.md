# Containerization and Delivery Pipeline — Technical Reference

Containerization and CI/CD pipelines transform experimental AI models into repeatable, deployable artifacts. Without disciplined delivery practices, GPU infrastructure becomes a collection of snowflake environments that are difficult to reproduce, scale, or debug.

## Container Architecture for AI

AI container images follow a layered architecture. The base layer is typically an Ubuntu or UBI image. On top of that sits the CUDA runtime (libcudart, cuBLAS, cuDNN, cuFFT, NPTL), which must be version-compatible with the host GPU driver. The framework layer adds PyTorch, TensorFlow, or JAX with their specific CUDA and cuDNN dependencies. The application layer adds custom training or inference code, configuration files, and model-loading logic. The model weights themselves may be baked into the image for inference or mounted at runtime from external storage. This layered approach enables layer caching — rebuilding the application layer does not require re-downloading the 8+ GB CUDA runtime layer.

## NGC — NVIDIA GPU Cloud Containers

NVIDIA NGC provides pre-built, optimized container images for major frameworks (PyTorch, TensorFlow, Triton Inference Server, RAPIDS, NeMo). These images are tested against specific driver versions and include performance-tuned libraries (cuDNN, NCCL, TensorRT). NGC images follow a monthly release cadence with version tags like `24.03-py3`. Version pinning is essential for reproducibility — always reference a specific tag, never `latest`. NGC images serve as reliable base images for custom containers, eliminating the need to manually resolve CUDA/cuDNN/framework version compatibility.

## NVIDIA Container Toolkit

The NVIDIA Container Toolkit enables GPU access inside containers. Its architecture consists of several components: `libnvidia-container` provides the low-level library for GPU container setup, `nvidia-container-cli` is the command-line interface for configuring GPU access, and the runtime hook (OCI prestart hook or CDI — Container Device Interface) injects GPU devices, driver libraries, and CUDA user-space components into the container at startup. Two runtime configuration approaches exist: the legacy `nvidia-container-runtime` (a shim around runc that intercepts container creation) and the newer CDI approach where GPU devices are declared as CDI resources and requested via standard OCI runtime spec annotations. CDI is the forward-looking standard and required for rootless container support. GPU device selection is controlled via the `NVIDIA_VISIBLE_DEVICES` environment variable or CDI device names. MIG (Multi-Instance GPU) devices are exposed as individual GPU instances, allowing containers to access specific MIG slices via device UUIDs.

## Image Optimization

AI container images are notoriously large — naive images can exceed 50 GB. Multi-stage builds are essential: use a full build image with compilers and development headers for the build stage, then copy only the compiled artifacts and runtime libraries into a slim runtime image. BuildKit's cache mounts (`--mount=type=cache`) accelerate rebuilds by caching pip and apt downloads across builds. Minimize CUDA toolkit components in runtime images — include only the runtime libraries (`libcudart`, `libcublas`, `libcudnn`), not the full toolkit with compilers and headers. Docker layer ordering matters: place infrequently changing layers (OS, CUDA) early and frequently changing layers (application code) last to maximize cache hits.

## CI/CD for AI Models

A production CI/CD pipeline for AI has distinct stages. The build stage constructs the Docker image using BuildKit with dependency caching and multi-stage builds. The test stage runs unit tests on CPU (fast, no GPU required), then integration tests on GPU CI runners (validates CUDA operations, model loading, inference correctness). Model accuracy regression tests compare inference outputs against a golden reference set to catch performance degradation. The promotion stage tags images through environments — `dev` to `staging` to `prod` — using immutable digest-based tags rather than mutable labels. Image signing with `cosign` (Sigstore) provides supply chain security and attestation. The deployment stage uses Kubernetes rolling updates, Triton Inference Server model repository versioning (model version directories), or canary deployments that route a percentage of inference traffic to the new model version while monitoring accuracy and latency metrics.

## MLOps Patterns

Model registries (MLflow Model Registry, Weights and Biases, or custom S3-based registries with metadata databases) track model versions, training lineage, performance metrics, and deployment status. Feature stores such as Feast or Tecton ensure that the same feature transformation logic used during training is applied during inference — feature skew between training and serving is a common source of production accuracy degradation. Experiment tracking captures hyperparameters, training curves, hardware utilization, and artifact locations for every training run, enabling reproducibility and comparison. Infrastructure as Code tools are essential for GPU infrastructure: Terraform manages cloud GPU instances, quotas, and networking; Helm charts package Kubernetes deployments of the GPU Operator, Triton, and monitoring stacks with parameterized values files.

## Monitoring Deployed Models

Production inference monitoring tracks request latency (p50, p95, p99), throughput (requests per second), GPU utilization and memory usage, error rates, and batch queue depth. Model drift detection compares inference input distributions and output distributions against training-time baselines — statistical tests (KS test, PSI) or learned drift detectors flag when the model's operating environment has shifted. A/B testing infrastructure routes traffic between model versions and measures business metrics alongside technical metrics.

## Common Anti-Patterns

Fat images exceeding 50 GB cause slow pull times and painful scaling during rolling updates. Running containers as root complicates shared cluster environments — use non-root users and `runAsNonRoot` security contexts. Hardcoded model paths prevent updating models without image rebuilds — mount weights from external storage instead. Containers without health check endpoints (`/health`, `/ready`) prevent Kubernetes from detecting unhealthy inference pods. Missing resource requests and limits cause scheduling failures and GPU oversubscription.

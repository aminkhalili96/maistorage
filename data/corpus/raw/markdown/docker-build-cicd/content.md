# Docker Build and CI/CD for AI Model Deployment

Docker containers are the standard deployment unit for AI inference services, training jobs, and development environments. This document covers multi-stage build patterns, NVIDIA GPU container runtime integration, NGC base images, and CI/CD pipeline design for GPU-accelerated workloads.

## Multi-Stage Builds

Multi-stage builds separate the build environment (compilers, CUDA toolkit headers, pip dependencies) from the runtime environment (CUDA runtime libs, model artifacts, inference server), reducing images from 15-20 GB to 3-8 GB.

Stage 1 (build) uses the full PyTorch NGC container with CUDA toolkit, cuDNN headers, and build tools. Stage 2 (runtime) uses only `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, copying installed packages from the builder via `COPY --from=builder /install /usr/local`. Model weights are added in the runtime stage.

## NVIDIA Container Runtime

The `nvidia-container-toolkit` exposes host GPU drivers inside containers. After installation, configure Docker with `nvidia-ctk runtime configure --runtime=docker` and restart the daemon. Run GPU containers with `docker run --gpus all` or `docker run --gpus '"device=0,1"'` for specific GPUs. The `--gpus` flag uses CDI (Container Device Interface) internally since Docker 25.0. Verify with `nvidia-smi` inside the container. The container's CUDA minor version must be compatible with the host driver's supported CUDA version.

## NGC Base Images

NVIDIA GPU Cloud provides pre-built images at `nvcr.io/nvidia/`:

| Image | Use Case | Size |
|---|---|---|
| `pytorch:24.03-py3` | Training and development | 15 GB |
| `tritonserver:24.03-py3` | Multi-framework inference | 12 GB |
| `cuda:12.4.1-runtime-ubuntu22.04` | Minimal CUDA runtime | 2.5 GB |
| `cuda:12.4.1-devel-ubuntu22.04` | CUDA development (nvcc) | 5.5 GB |

Pin tags to specific monthly releases (e.g., `24.03-py3`, not `latest`) for reproducible builds. Authenticate with `docker login nvcr.io` using your NGC API key.

## BuildKit Features

**Cache mounts** preserve pip/conda caches across builds: `RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt`. **Secret mounts** inject credentials without embedding in layers: `RUN --mount=type=secret,id=ngc_key NGC_API_KEY=$(cat /run/secrets/ngc_key) python download_model.py`. Build with `docker build --secret id=ngc_key,src=$HOME/.ngc/api_key .`. BuildKit also executes independent stages in parallel automatically.

## CI/CD Pipeline Patterns

### Build

Lint (`ruff check src/`), unit test (`pytest tests/unit/`), build image (`docker build -t $REGISTRY/$IMAGE:$SHA .`), push to registry. Tag with Git SHA for traceability. Unit tests run on CPU-only runners with mocked GPU operations.

### Test

GPU integration tests require self-hosted runners with physical GPUs. Run: `docker run --gpus all --rm $REGISTRY/$IMAGE:$SHA pytest tests/integration/ -v`. Validate model loading, inference correctness against golden references, throughput benchmarks (tokens/sec), and GPU memory consumption.

### Promote

Promotion gates: all integration tests pass, image vulnerability scan (Trivy/Grype) reports no critical CVEs, model accuracy exceeds baselines, image size within bounds. Promote by retagging: `docker tag $IMAGE:$SHA $IMAGE:staging`, then `$IMAGE:production` after canary validation.

### Deploy

**Kubernetes rolling update**: Update Deployment image tag with `maxUnavailable: 0`, `maxSurge: 1`. GPU requests (`nvidia.com/gpu: 1`) ensure scheduling on GPU nodes. **Blue-green**: Two Deployments, switch Service selector after health checks. **Triton model update**: Update the model repository and call `/v2/repository/models/{model}/load` -- no container restart required.

## Model Image Patterns

**Baking models in**: Copy weights during build. Large images (10-30 GB) but code and model versioned together. Suitable for models under 5 GB. **Mounting from storage**: Download weights at startup from MinIO/S3 via init container or entrypoint script. Adds 30-120 seconds startup but keeps images small. Preferred for models over 5 GB. **Versioning**: Tag images as `$IMAGE:code-v1.3-model-v2.1` or use labels: `LABEL model.version=2.1`.

## Security

**Image scanning**: `trivy image --severity HIGH,CRITICAL $IMAGE:$SHA` in CI; fail builds on critical findings. NGC images are pre-scanned by NVIDIA. **Non-root execution**: `USER 1000:1000` in Dockerfile; NVIDIA runtime does not require root inside the container. **Read-only filesystem**: `docker run --read-only --tmpfs /tmp` prevents modification of application code.

## Docker Compose for Local GPU Development

```yaml
services:
  inference:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "8000:8000"
```

Use `count: 1` to limit to a single GPU. Requires Compose V2 (`docker compose` plugin, not legacy `docker-compose`).

## Registry Management

Implement retention policies to delete untagged images and images older than 90 days. Enable automatic scanning on push (Harbor, ECR, GCR support built-in Trivy/Clair). Sign images with `cosign sign $IMAGE:$SHA` and verify with admission controllers (Kyverno, OPA Gatekeeper) in Kubernetes.

## Common Pitfalls

**CUDA version mismatches**: Container CUDA must be compatible with host driver. Check with `nvidia-smi`. Symptom: `CUDA error: no kernel image is available for execution on the device`. **Large images**: Mitigate with multi-stage builds, `.dockerignore` (exclude `.git/`, `__pycache__/`, raw datasets), and `-runtime-` instead of `-devel-` base images. **Driver compatibility**: The GPU Operator manages driver installation across Kubernetes clusters. **Layer caching**: Place `COPY requirements.txt` and `RUN pip install` before `COPY src/` so code changes do not invalidate the dependency layer.

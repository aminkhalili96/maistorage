"""Performance benchmark tests — measures latency and throughput of local operations.

No LLM API calls. All operations are local (keyword index, chunking, classification).
"""
from __future__ import annotations

import statistics
import time

from app.config import get_settings
from app.corpus import load_demo_chunks, load_normalized_chunks, load_sources
from app.services.chunking import chunk_markdown_document
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import KeywordEmbedder
from app.services.retrieval import classify_assistant_mode
from app.models import DocumentSource


def test_ingestion_speed():
    """Time how long it takes to load all JSONL files from normalized_doc_root. Assert < 10 seconds."""
    settings = get_settings()
    start = time.perf_counter()
    chunks = load_normalized_chunks(settings.normalized_doc_root)
    elapsed = time.perf_counter() - start

    assert len(chunks) > 0, "Expected at least some normalized chunks to load"
    assert elapsed < 10.0, f"JSONL loading took {elapsed:.2f}s, expected < 10s"


def test_search_latency_p95():
    """Build a full index with demo chunks, run 50 keyword searches. Assert P95 < 500ms."""
    settings = get_settings()
    demo_chunks = load_demo_chunks(settings.demo_corpus_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    index.upsert(demo_chunks)

    queries = [
        "NCCL tuning parameters for bandwidth",
        "H100 memory capacity",
        "multi-GPU training scaling",
        "mixed precision training FP16",
        "CUDA installation guide",
        "NVLink topology",
        "container toolkit deployment",
        "GPU operator Kubernetes",
        "Megatron-LM pipeline parallelism",
        "tensor core utilization",
        "DGX BasePOD architecture",
        "RDMA over Converged Ethernet",
        "GPUDirect Storage configuration",
        "Triton Inference Server setup",
        "BeeGFS parallel file system",
        "Lustre client mount options",
        "A100 SXM4 specifications",
        "DCGM health monitoring",
        "Fabric Manager NVSwitch",
        "Slurm workload scheduler",
        "model parallelism vs data parallelism",
        "all-reduce communication overhead",
        "NCCL_ALGO ring tree",
        "transformer engine FP8",
        "NSight Compute profiling",
        "inference pipeline optimization",
        "TensorRT model conversion",
        "NIM deployment guide",
        "L40S inference performance",
        "H200 HBM3e memory",
        "PCIe Gen5 bandwidth",
        "NVMe over Fabrics",
        "SHARP collective offload",
        "MIG vGPU partitioning",
        "RAPIDS cuDF performance",
        "speculative decoding latency",
        "Docker build CI/CD pipeline",
        "MinIO object storage S3",
        "Linux mdraid configuration",
        "GPU monitoring best practices",
        "enterprise reference architecture",
        "advanced Kubernetes GPU scheduling",
        "base command manager",
        "GH200 Grace Hopper",
        "B200 Blackwell architecture",
        "cluster networking topology",
        "memory bandwidth roofline",
        "input pipeline bottleneck",
        "storage throughput optimization",
        "NVIDIA AI Enterprise licensing",
    ]

    timings = []
    for query in queries:
        start = time.perf_counter()
        index.search(query, top_k=5)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    p95 = statistics.quantiles(timings, n=20)[18]  # 95th percentile
    assert p95 < 0.5, f"P95 search latency is {p95*1000:.1f}ms, expected < 500ms"


def test_chunking_throughput():
    """Generate a 10KB markdown string. Time chunk_markdown_document(). Assert < 1 second."""
    # Build a realistic ~10KB markdown document
    sections = []
    for i in range(20):
        section = f"## Section {i}: NVIDIA GPU Infrastructure Topic {i}\n\n"
        for j in range(5):
            section += (
                f"This is paragraph {j} in section {i} about NVIDIA GPU infrastructure. "
                f"The H100 GPU features 80GB of HBM3 memory with 3.35 TB/s memory bandwidth. "
                f"NVLink 4.0 provides 900 GB/s bidirectional bandwidth for multi-GPU scaling. "
                f"NCCL handles collective communication operations such as all-reduce and broadcast. "
                f"Mixed precision training uses FP16 compute with FP32 accumulation for efficiency.\n\n"
            )
        sections.append(section)
    markdown_text = "\n".join(sections)
    assert len(markdown_text) >= 10000, f"Generated markdown is only {len(markdown_text)} bytes"

    source = DocumentSource(
        id="perf-test",
        title="Performance Test Document",
        url="https://example.com/perf-test",
        doc_family="core",
        doc_type="markdown",
        crawl_prefix="https://example.com/",
    )

    start = time.perf_counter()
    chunks = chunk_markdown_document(source, source.url, markdown_text)
    elapsed = time.perf_counter() - start

    assert len(chunks) > 0, "Expected at least one chunk from markdown"
    assert elapsed < 1.0, f"Chunking took {elapsed:.2f}s, expected < 1s"


def test_index_build_time():
    """Time building an InMemoryHybridIndex and ingesting all demo chunks. Assert < 5 seconds."""
    settings = get_settings()
    demo_chunks = load_demo_chunks(settings.demo_corpus_path)

    start = time.perf_counter()
    index = InMemoryHybridIndex(KeywordEmbedder())
    index.upsert(demo_chunks)
    elapsed = time.perf_counter() - start

    assert index.count() > 0, "Expected chunks in the index"
    assert elapsed < 5.0, f"Index build took {elapsed:.2f}s, expected < 5s"


def test_classification_speed():
    """Time 100 calls to classify_assistant_mode() with varied queries. Assert average < 10ms."""
    queries = [
        "Why is 4-GPU training scaling poorly?",
        "What are the key tuning parameters for NCCL?",
        "Hello, how are you?",
        "What's the weather in San Francisco?",
        "What is NVIDIA stock price today?",
        "How do I install CUDA on Ubuntu?",
        "Compare H100 and A100 memory bandwidth",
        "What is NVLink?",
        "Tell me about the DGX BasePOD",
        "What changed in the latest NVIDIA Container Toolkit release?",
        "How do I configure BeeGFS for GPU clusters?",
        "What is the recipe for pasta?",
        "Who founded NVIDIA?",
        "Explain mixed precision training",
        "What are the DCGM health check commands?",
        "How does GPUDirect Storage work?",
        "What is the latest NBA score?",
        "How do I deploy Triton Inference Server on Kubernetes?",
        "What GPU should I use for inference?",
        "Tell me about Lustre parallel file system",
    ]

    total_time = 0.0
    iterations = 100
    for i in range(iterations):
        query = queries[i % len(queries)]
        start = time.perf_counter()
        classify_assistant_mode(query)
        elapsed = time.perf_counter() - start
        total_time += elapsed

    avg_ms = (total_time / iterations) * 1000
    assert avg_ms < 10.0, f"Average classification time is {avg_ms:.2f}ms, expected < 10ms"

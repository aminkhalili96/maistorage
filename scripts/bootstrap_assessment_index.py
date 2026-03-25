from __future__ import annotations

import argparse
from itertools import islice

from app.config import get_settings
from app.knowledge_base import load_demo_chunks, load_normalized_chunks
from app.services.indexes import PineconeHybridIndex
from app.services.providers import build_embedder


DEFAULT_SOURCE_IDS = [
    "a100",
    "advanced-k8s-gpu",
    "b200",
    "base-command",
    "beegfs",
    "container-toolkit",
    "cublas",
    "cuda-best-practices",
    "cuda-install",
    "cuda-programming-guide",
    "cudnn",
    "dcgm",
    "dgx-basepod",
    "dl-performance",
    "docker-build-cicd",
    "enterprise-reference-architecture",
    "fabric-manager",
    "gh200",
    "gpu-monitoring",
    "gpu-operator",
    "gpudirect-storage",
    "h100",
    "h200",
    "infra-cluster-ops",
    "infra-mlops",
    "infra-platforms",
    "infra-storage",
    "inference-pipeline",
    "kubernetes-workloads",
    "l40s",
    "linux-mdraid",
    "lustre",
    "megatron-core",
    "mig-vgpu",
    "minio-object-storage",
    "mlnx-ofed",
    "nccl",
    "nemo-performance",
    "nsight-compute",
    "nvidia-ai-enterprise",
    "nvidia-nim",
    "nvme-of",
    "rapids",
    "sharp",
    "slurm-workload-manager",
    "speculative-decoding",
    "tensorrt",
    "transformer-engine",
    "triton-inference-server",
]


def batched(items, size: int):
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate Pinecone with the interview-critical NVIDIA knowledge base subset.")
    parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        help="Specific source id to include. Can be passed multiple times. Defaults to the interview subset.",
    )
    parser.add_argument(
        "--demo-knowledge-base",
        action="store_true",
        help="Populate Pinecone from the bundled demo chunk set instead of the larger normalized knowledge base.",
    )
    parser.add_argument("--batch-size", type=int, default=25, help="Number of chunks to upsert per batch.")
    args = parser.parse_args()

    settings = get_settings()
    source_ids = args.source_ids or DEFAULT_SOURCE_IDS

    if args.demo_knowledge_base:
        chunks = load_demo_chunks(settings.demo_knowledge_base_path)
    else:
        chunks = [chunk for chunk in load_normalized_chunks(settings.normalized_doc_root) if chunk.source_id in set(source_ids)]
    if not chunks:
        raise RuntimeError("No normalized chunks found for the requested source ids.")

    embedder = build_embedder(settings)
    index = PineconeHybridIndex(settings, embedder)
    before = index.count()

    total = len(chunks)
    print(f"Target namespace: {settings.pinecone_namespace}")
    if args.demo_knowledge_base:
        print("Selected sources: bundled demo knowledge base")
    else:
        print(f"Selected sources: {', '.join(source_ids)}")
    print(f"Selected chunks: {total}")
    print(f"Record count before upsert: {before}")

    processed = 0
    for batch in batched(chunks, args.batch_size):
        index.upsert(batch)
        processed += len(batch)
        print(f"Upserted {processed}/{total} chunks")

    after = index.count()
    print(f"Record count after upsert: {after}")


if __name__ == "__main__":
    main()

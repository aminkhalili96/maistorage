from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "data" / "evals"


def load_rows(name: str) -> list[dict]:
    return json.loads((EVAL_ROOT / name).read_text())


def slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def build() -> list[dict]:
    core = load_rows("golden_questions.json")
    job = load_rows("job_requirement_questions.json")
    ext = load_rows("jd_offline_extension_questions.json")
    harder = load_rows("harder_questions.json")
    deploy = load_rows("deployment_eval_questions.json")

    # Questions removed as redundant (covered by more specific/analytical alternatives)
    redundant_core_questions = {
        "What should a repeatable Linux GPU deployment path include?",
        "Why does the NVIDIA container runtime matter for GPU workloads?",
        "What does the GPU Operator simplify in a Kubernetes environment?",
    }
    redundant_job_questions = {
        "Why would an AI team use a parallel file system like Lustre or BeeGFS?",
        "What RAID levels should I compare for AI server redundancy and performance?",
        "How should I compare H100, A100, and L40S for AI workloads?",
    }

    core = [row for row in core if row["question"] not in redundant_core_questions]
    job = [row for row in job if row["question"] not in redundant_job_questions]

    selected_extension_questions = {
        "Which Kubernetes workload primitives matter for AI infrastructure components?",
        "Why would an AI platform choose BeeGFS for shared training data?",
        "What does Lustre add for large training clusters?",
        "How should I think about Linux mdraid and RAID levels on AI servers?",
    }
    selected_harder_questions = {
        "Which NVIDIA document gives exact BIOS settings for every EPYC motherboard model?",
        "What changed in the latest CUDA release this week?",
    }
    selected_deploy_questions = {
        "Describe the end-to-end pipeline from training a model to serving it in production on an NVIDIA GPU cluster.",
        "When should you use NVIDIA NIM versus Triton for deploying LLM inference?",
        "How do you monitor GPU health and inference performance in a production deployment?",
        "What is NVIDIA AI Enterprise and how does it relate to NGC containers?",
        "What storage architecture would you recommend for serving multiple large language models with fast model loading?",
    }

    extension_rows = [row for row in ext if row["question"] in selected_extension_questions]
    harder_rows = [row for row in harder if row["question"] in selected_harder_questions]
    deploy_rows = [row for row in deploy if row["question"] in selected_deploy_questions]

    direct_chat_rows = [
        {
            "question": "What is NVIDIA?",
            "category": "direct_chat",
            "assistant_mode_expected": "direct_chat",
            "response_mode_expected": "direct-chat",
            "expected_sources": [],
            "expected_terms": ["technology company", "gpu", "ai"],
            "reference_answer": "NVIDIA is a technology company best known for GPUs and accelerated computing products used in gaming, AI, and data center systems.",
            "expected_tool_path": ["direct_chat"],
            "max_retries": 0,
            "should_require_citations": False,
            "should_use_fallback": False,
        },
        {
            "question": "Who founded NVIDIA?",
            "category": "direct_chat",
            "assistant_mode_expected": "direct_chat",
            "response_mode_expected": "direct-chat",
            "expected_sources": [],
            "expected_terms": ["jensen huang", "co-founded", "1993"],
            "reference_answer": "NVIDIA was co-founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem.",
            "expected_tool_path": ["direct_chat"],
            "max_retries": 0,
            "should_require_citations": False,
            "should_use_fallback": False,
        },
    ]

    def normalize_core(row: dict) -> dict:
        return {
            "question": row["question"],
            "category": row["query_class"],
            "assistant_mode_expected": "doc_rag",
            "response_mode_expected": "corpus-backed",
            "expected_sources": row["expected_sources"],
            "expected_terms": row["expected_terms"],
            "reference_answer": row["reference_answer"],
            "expected_tool_path": ["nvidia_docs"],
            "max_retries": 2,
            "should_require_citations": True,
            "should_use_fallback": False,
        }

    def normalize_job(row: dict) -> dict:
        lowered = row["question"].lower()
        if "networking" in lowered or "throughput" in lowered:
            category = "distributed_multi_gpu"
        elif "scheduler" in lowered or "kubernetes" in lowered:
            category = "cluster_operations"
        elif "linux" in lowered or "docker" in lowered:
            category = "deployment_runtime"
        elif "file system" in lowered or "object storage" in lowered or "raid" in lowered:
            category = "storage_data_path"
        elif "ci/cd" in lowered:
            category = "mlops_delivery"
        else:
            category = "hardware_topology"
        return {
            "question": row["question"],
            "category": category,
            "assistant_mode_expected": "doc_rag",
            "response_mode_expected": "corpus-backed",
            "expected_sources": row["expected_sources"],
            "expected_terms": row["expected_terms"],
            "reference_answer": row["reference_answer"],
            "expected_tool_path": ["nvidia_docs"],
            "max_retries": 2,
            "should_require_citations": True,
            "should_use_fallback": False,
        }

    def normalize_extension(row: dict) -> dict:
        lowered = row["question"].lower()
        if "kubernetes" in lowered:
            category = "cluster_operations"
        elif "raid" in lowered:
            category = "storage_data_path"
        else:
            category = "storage_data_path"
        return {
            "question": row["question"],
            "category": category,
            "assistant_mode_expected": "doc_rag",
            "response_mode_expected": "corpus-backed",
            "expected_sources": row["expected_sources"],
            "expected_terms": row["expected_terms"],
            "reference_answer": row.get("reference_answer")
            or "A grounded answer should stay within the local offline infrastructure guide and summarize only the supported storage or cluster operations details.",
            "expected_tool_path": ["nvidia_docs"],
            "max_retries": 2,
            "should_require_citations": True,
            "should_use_fallback": False,
        }

    def normalize_harder(row: dict) -> dict:
        payload = {
            "question": row["question"],
            "category": "recency_fallback" if "latest cuda release" in row["question"].lower() else "refusal",
            "assistant_mode_expected": row["assistant_mode_expected"],
            "response_mode_expected": "web-backed"
            if "latest cuda release" in row["question"].lower()
            else row["response_mode_expected"],
            "expected_sources": row.get("expected_sources", []),
            "expected_terms": row.get("expected_terms", []),
            "reference_answer": row["reference_answer"],
            "expected_tool_path": ["nvidia_docs", "web_search"]
            if "latest cuda release" in row["question"].lower()
            else ["nvidia_docs"],
            "max_retries": 2,
            "should_require_citations": "latest cuda release" in row["question"].lower(),
            "should_use_fallback": "latest cuda release" in row["question"].lower(),
        }
        return payload

    def normalize_deploy(row: dict) -> dict:
        return {
            "question": row["question"],
            "category": row.get("category", "inference_deployment"),
            "assistant_mode_expected": "doc_rag",
            "response_mode_expected": "corpus-backed",
            "expected_sources": row["expected_sources"],
            "expected_terms": row["expected_terms"],
            "reference_answer": row["reference_answer"],
            "expected_tool_path": ["nvidia_docs"],
            "max_retries": 2,
            "should_require_citations": True,
            "should_use_fallback": False,
        }

    # 5 new custom questions for uncovered sources
    new_source_questions = [
        {
            "question": "What are the key specifications of the H200 GPU and how does it improve on the H100 for large model inference?",
            "category": "hardware_topology",
            "assistant_mode_expected": "doc_rag",
            "response_mode_expected": "corpus-backed",
            "expected_sources": ["h200", "h100"],
            "expected_terms": ["H200", "H100", "HBM3e", "memory bandwidth", "inference", "large model"],
            "reference_answer": "The H200 uses HBM3e memory with significantly higher bandwidth and capacity compared to the H100's HBM3, enabling faster inference for large language models that are memory-bandwidth-bound. The H200 maintains the same Hopper architecture and compute capabilities as the H100 but the increased memory bandwidth reduces time-to-first-token and improves throughput for models that exceed the H100's memory capacity.",
            "expected_tool_path": ["nvidia_docs"],
            "max_retries": 2,
            "should_require_citations": True,
            "should_use_fallback": False,
        },
        {
            "question": "How does NVIDIA Transformer Engine enable FP8 training and what are the performance benefits compared to BF16?",
            "category": "training_optimization",
            "assistant_mode_expected": "doc_rag",
            "response_mode_expected": "corpus-backed",
            "expected_sources": ["transformer-engine"],
            "expected_terms": ["FP8", "BF16", "Transformer Engine", "mixed precision", "Hopper", "throughput"],
            "reference_answer": "NVIDIA Transformer Engine automatically manages FP8 precision on Hopper GPUs by using per-tensor scaling factors to maintain accuracy while reducing memory footprint and increasing arithmetic throughput. Compared to BF16 training, FP8 can roughly double throughput for transformer layers by leveraging the Hopper FP8 Tensor Cores, while the automatic scaling prevents the accuracy degradation that naive 8-bit quantization would cause.",
            "expected_tool_path": ["nvidia_docs"],
            "max_retries": 2,
            "should_require_citations": True,
            "should_use_fallback": False,
        },
        {
            "question": "What is a DGX BasePOD and when would you deploy one versus a custom GPU cluster?",
            "category": "hardware_topology",
            "assistant_mode_expected": "doc_rag",
            "response_mode_expected": "corpus-backed",
            "expected_sources": ["dgx-basepod"],
            "expected_terms": ["DGX BasePOD", "reference architecture", "networking", "storage", "validated", "cluster"],
            "reference_answer": "A DGX BasePOD is NVIDIA's validated reference architecture that combines DGX systems with prescribed networking (InfiniBand or Ethernet) and storage configurations, tested and certified as a unit. You would deploy a BasePOD when you need a turnkey, vendor-supported cluster with guaranteed performance characteristics, whereas a custom GPU cluster offers more flexibility in component selection and cost optimization but requires more integration and validation effort.",
            "expected_tool_path": ["nvidia_docs"],
            "max_retries": 2,
            "should_require_citations": True,
            "should_use_fallback": False,
        },
        {
            "question": "How does Nsight Compute help identify performance bottlenecks in CUDA kernels?",
            "category": "training_optimization",
            "assistant_mode_expected": "doc_rag",
            "response_mode_expected": "corpus-backed",
            "expected_sources": ["nsight-compute"],
            "expected_terms": ["Nsight Compute", "kernel", "profiling", "occupancy", "memory", "bottleneck"],
            "reference_answer": "Nsight Compute is NVIDIA's interactive CUDA kernel profiler that collects detailed hardware performance counters per kernel launch. It identifies bottlenecks by analyzing compute throughput, memory throughput, occupancy, warp scheduling efficiency, and instruction mix. The guided analysis compares achieved versus theoretical peak performance across the compute, memory, and latency domains, pinpointing whether a kernel is compute-bound, memory-bound, or latency-bound.",
            "expected_tool_path": ["nvidia_docs"],
            "max_retries": 2,
            "should_require_citations": True,
            "should_use_fallback": False,
        },
        {
            "question": "How does Slurm handle GPU-aware batch scheduling for distributed AI training jobs?",
            "category": "cluster_operations",
            "assistant_mode_expected": "doc_rag",
            "response_mode_expected": "corpus-backed",
            "expected_sources": ["slurm-workload-manager"],
            "expected_terms": ["Slurm", "GPU", "scheduling", "GRES", "batch", "distributed"],
            "reference_answer": "Slurm handles GPU-aware scheduling through its GRES (Generic RESource) plugin, which tracks GPU availability per node and allocates specific GPU devices to jobs. For distributed AI training, Slurm can allocate multi-node GPU resources, set up the necessary environment variables for NCCL communication, and manage job queues with GPU-aware fair-share scheduling. The srun launcher coordinates multi-process execution across allocated nodes.",
            "expected_tool_path": ["nvidia_docs"],
            "max_retries": 2,
            "should_require_citations": True,
            "should_use_fallback": False,
        },
    ]

    rows: list[dict] = []
    for row in core:
        rows.append(normalize_core(row))
    for row in job:
        rows.append(normalize_job(row))
    for row in extension_rows:
        rows.append(normalize_extension(row))
    for row in deploy_rows:
        rows.append(normalize_deploy(row))
    rows.extend(new_source_questions)

    # Fix expected_sources on questions that reference generic meta-sources
    source_fixes = {
        "When should I use an HPC scheduler like Slurm versus Kubernetes for AI workloads?": "slurm-workload-manager",
        "When is S3-compatible object storage useful for AI datasets and checkpoints?": "minio-object-storage",
        "Why do CI/CD pipelines matter for model deployment and MLOps?": "docker-build-cicd",
    }
    for row in rows:
        fix_source = source_fixes.get(row["question"])
        if fix_source and fix_source not in row["expected_sources"]:
            row["expected_sources"].append(fix_source)

    rows.extend(direct_chat_rows)
    for row in harder_rows:
        rows.append(normalize_harder(row))

    assert len(rows) == 54, len(rows)

    enriched: list[dict] = []
    for index, row in enumerate(rows, start=1):
        enriched.append(
            {
                "id": f"agentic-{index:03d}-{slugify(row['question'])[:48]}",
                **row,
            }
        )
    return enriched


def main() -> None:
    payload = build()
    output_path = EVAL_ROOT / "agentic_golden_questions.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()

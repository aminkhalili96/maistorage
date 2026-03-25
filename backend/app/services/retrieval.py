from __future__ import annotations

import json
import logging
import re
import threading

from app.config import RerankConfig, Settings
from app.models import ChatTurn, DocumentSource, QueryClass, QueryPlan, RetrieverResult, SearchDebugResponse, TraceEvent
from app.services.indexes import SearchIndex
from app.services.providers import tokenize

_DEFAULT_RERANK_CONFIG = RerankConfig()

_log = logging.getLogger("maistorage.retrieval")

# If confidence exceeds this threshold after the first query, skip remaining expansion queries.
EARLY_STOP_CONFIDENCE = 0.75


QUERY_CLASS_RULES: list[tuple[QueryClass, tuple[str, ...]]] = [
    (
        QueryClass.distributed_multi_gpu,
        (
            "multi-gpu",
            "multi-gpu configuration",
            "multi-gpu configurations",
            "distributed",
            "nccl",
            "all-reduce",
            "nvlink",
            "nvswitch",
            "interconnect",
            "cluster networking",
            "networking",
            "4-gpu",
            "scaling",
            "parallelism",
            "pipeline parallelism",
            "tensor parallelism",
            "data parallelism",
            "large model",
            "model size",
            "memory limits",
            "communication cost",
            "collective performance",
            "synchronization",
            "gpu count",
        ),
    ),
    (
        QueryClass.hardware_topology,
        (
            "h100",
            "h200",
            "a100",
            "l40s",
            "hardware",
            "topology",
            "sizing",
            "on-prem",
            "server design",
            "motherboard",
            "motherboards",
            "epyc",
            "xeon",
            "ram",
            "high-speed ram",
            "memory channels",
            "pcie",
            "pcie lanes",
            "enterprise cpu",
            "server components",
            "host-to-device",
            "hbm3e",
            "hbm2e",
            "nvlink 4",
            "nvlink 3",
            "nvswitch 3",
            "memory bandwidth",
            "bandwidth gb",
        ),
    ),
    (
        QueryClass.deployment_runtime,
        (
            "deploy",
            "deployment",
            "runtime",
            "linux",
            "driver",
            "cuda",
            "container",
            "kubernetes",
            "operator",
            "slurm",
            "scheduler",
            "scheduling",
            "parallel file system",
            "lustre",
            "beegfs",
            "gpfs",
            "object storage",
            "s3",
            "raid",
            "ci/cd",
            "pipeline",
            "pipelines",
            "mlops",
            "compatibility",
            "bottleneck",
            "bottlenecks",
            "cluster management",
            "health check",
            "health checks",
            "diagnostics",
            "dcgm",
            "dcgmi",
            "monitoring",
            "pre-job",
            "preflight",
        ),
    ),
    (QueryClass.training_optimization, ("mixed precision", "throughput", "utilization", "training", "tensor core", "memory-bound", "profiling", "input pipeline", "storage path", "storage throughput")),
]


CLASS_FAMILY_MAP: dict[QueryClass, list[str]] = {
    QueryClass.training_optimization: ["core", "advanced", "infrastructure"],
    QueryClass.distributed_multi_gpu: ["distributed", "core", "advanced"],
    QueryClass.deployment_runtime: ["infrastructure", "core"],
    QueryClass.hardware_topology: ["hardware", "infrastructure", "advanced"],
    QueryClass.general: ["core", "distributed", "infrastructure", "advanced", "hardware"],
}


EXPANSION_TERMS: dict[QueryClass, list[str]] = {
    QueryClass.training_optimization: ["tensor cores", "roofline", "sm efficiency", "compute throughput", "memory bandwidth", "input pipeline", "storage path"],
    QueryClass.distributed_multi_gpu: ["nccl", "NCCL_ALGO", "ring algorithm", "all-reduce", "parallelism", "tensor parallel", "pipeline parallel", "micro-batch", "NVLink bandwidth"],
    QueryClass.deployment_runtime: ["drivers", "cuda toolkit", "container runtime", "dcgm diagnostics", "health monitoring", "nvidia-container-runtime", "device plugin", "ClusterPolicy", "BeeGFS stripe", "Lustre OST"],
    QueryClass.hardware_topology: ["h100", "a100", "l40s", "HBM3e", "HBM2e", "NVLink 4.0", "NVLink 3.0", "NVSwitch", "server design"],
    QueryClass.general: ["nvidia training infrastructure", "retrieval optimization", "ai infrastructure"],
}


QUERY_CLASS_SOURCE_HINTS: dict[QueryClass, dict[str, float]] = {
    QueryClass.training_optimization: {
        "dl-performance": 0.42,
        "cuda-best-practices": 0.24,
        "nsight-compute": 0.18,
        "gpudirect-storage": 0.26,
    },
    QueryClass.distributed_multi_gpu: {
        "nccl": 0.48,
        "fabric-manager": 0.32,
        "megatron-core": 0.26,
        "nemo-performance": 0.12,
    },
    QueryClass.deployment_runtime: {
        "cuda-install": 0.42,
        "container-toolkit": 0.34,
        "gpu-operator": 0.28,
        "dgx-basepod": 0.12,
        "infra-cluster-ops": 0.22,
        "infra-storage": 0.18,
        "infra-mlops": 0.18,
        "slurm-workload-manager": 0.34,
        "kubernetes-workloads": 0.32,
        "beegfs": 0.38,
        "lustre": 0.38,
        "minio-object-storage": 0.36,
        "linux-mdraid": 0.38,
        "docker-build-cicd": 0.34,
        "dcgm": 0.52,
    },
    QueryClass.hardware_topology: {
        "h100": 0.56,
        "h200": 0.48,
        "a100": 0.5,
        "l40s": 0.44,
        "dgx-basepod": 0.12,
        "infra-platforms": 0.3,
    },
    QueryClass.general: {},
}

RECENCY_TERMS = ("latest", "current", "recent", "today", "yesterday", "newest", "release notes", "changed", "stock price", "share price", "market cap", "stock market")
FINANCIAL_TERMS = ("stock price", "share price", "market cap", "stock market", "shares outstanding", "earnings", "valuation", "ticker", "stock")
# Topics that need live web data — route to Tavily directly instead of corpus retrieval.
LIVE_QUERY_TERMS = (
    "weather", "forecast", "temperature outside", "raining", "sunny",
    "stock price", "share price", "market cap", "stock market",
    "shares outstanding", "earnings", "valuation", "ticker",
    "news today", "latest news", "current events",
)
DOC_RAG_INFRA_TERMS = (
    "cuda",
    "cudnn",
    "cublas",
    "cufft",
    "curand",
    "nccl",
    "nvlink",
    "nvswitch",
    "nvshmem",
    "nsight",
    "tensorrt",
    "triton",
    "gpudirect",
    "gpu operator",
    "fabric manager",
    "container toolkit",
    "transformer engine",
    "fp8",
    "bf16",
    "fp16",
    "gemm",
    "training",
    "precision",
    "dcgm",
    "health check",
    "health checks",
    "diagnostics",
    "driver",
    "drivers",
    "deployment",
    "runtime",
    "kubernetes",
    "docker",
    "linux",
    "h100",
    "h200",
    "a100",
    "l40s",
    "dgx",
    "megatron",
    "nemo",
    "mixed precision",
    "profiling",
    "throughput",
    "scaling",
    "tensor core",
    "parallelism",
    "pipeline parallelism",
    "tensor parallelism",
    "data parallelism",
    "large model",
    "model size",
    "memory limits",
    "communication cost",
    "collective performance",
    "cluster networking",
    "storage throughput",
    "slurm",
    "scheduler",
    "scheduling",
    "motherboard",
    "motherboards",
    "epyc",
    "xeon",
    "pcie",
    "ram",
    "parallel file system",
    "lustre",
    "beegfs",
    "gpfs",
    "object storage",
    "s3",
    "raid",
    "ci/cd",
    "pipeline",
    "pipelines",
    "mlops",
    "official docs",
    "official documentation",
    # Hardware health / thermal / power
    "temperature",
    "thermal",
    "power",
    "watt",
    "tdp",
    "cooling",
    "gpu memory",
    "oom",
    "out of memory",
    "vram",
    # Networking
    "infiniband",
    "rdma",
    "roce",
    # Server management
    "bmc",
    "ipmi",
    "redfish",
    "baseboard",
)
GENERIC_DOC_TERMS = ("docs", "documentation", "manual", "manuals", "guide", "guides")
NVIDIA_ENTITY_TERMS = ("nvidia", "geforce", "rtx")
NVIDIA_DOC_CONTEXT_TERMS = (
    "official",
    "stack",
    "deploy",
    "deployment",
    "runtime",
    "driver",
    "drivers",
    "container",
    "containers",
    "kubernetes",
    "gpu",
    "gpus",
    "linux",
    "install",
    "setup",
    "operator",
    "training",
    "mixed precision",
    "profiling",
    "throughput",
    "scaling",
    "cluster",
    "scheduler",
    "slurm",
    "motherboard",
    "pcie",
    "epyc",
    "xeon",
    "raid",
    "s3",
    "object storage",
    "parallel file system",
    "ci/cd",
    "mlops",
)
CASUAL_CHAT_TERMS = (
    "hi",
    "hello",
    "hey",
    "hellaur",
    "yo",
    "sup",
    "thanks",
    "thank you",
    "good morning",
    "good afternoon",
    "good evening",
)
# Topics that are clearly out-of-scope for an NVIDIA infra assistant — route to direct-chat
# instead of going to Tavily (which would return garbage non-NVIDIA results).
# NOTE: weather/forecast/stock terms moved to LIVE_QUERY_TERMS — they now route through Tavily.
OUT_OF_SCOPE_TERMS = (
    "sports",
    "football",
    "basketball",
    "soccer",
    "cricket",
    "baseball",
    "nba",
    "nfl",
    "recipe",
    "cooking",
    "restaurant",
    "flight",
    "airline",
    "hotel",
    "vacation",
    "horoscope",
    "zodiac",
    "lottery",
    "political",
    "election",
    "president",
    "prime minister",
)
DOC_ACTION_TERMS = (
    "install",
    "setup",
    "configure",
    "deploy",
    "deployment",
    "runtime",
    "driver",
    "drivers",
    "container",
    "containers",
    "kubernetes",
    "operator",
    "cluster",
    "stack",
    "profile",
    "profiling",
    "optimize",
    "optimization",
    "throughput",
    "latency",
    "performance",
    "debug",
    "debugging",
    "troubleshoot",
    "troubleshooting",
    "scale",
    "scaling",
    "mixed precision",
    "tensor core",
    "compatibility",
    "requirements",
    "networking",
    "storage",
    "filesystem",
    "file systems",
    "object storage",
    "raid",
    "scheduler",
    "scheduling",
    "ci/cd",
    "pipeline",
    "pipelines",
)
FOLLOW_UP_DOC_RAG_TERMS = ("where", "which one", "which guide", "how", "steps", "show me", "link", "source", "sources")
GENERAL_KNOWLEDGE_PATTERNS = (
    re.compile(r"^(what|who|where|when)\s+(is|are|was|were)\b"),
    re.compile(r"^what does\b"),
    re.compile(r"^who founded\b"),
    re.compile(r"^who makes\b"),
    re.compile(r"^tell me about\b"),
    re.compile(r"^give me (an?\s+)?overview of\b"),
    re.compile(r"^overview of\b"),
    re.compile(r"^history of\b"),
    re.compile(r"^define\b"),
)
KNOWN_DIRECT_CHAT_FACTOIDS = (
    "what is nvidia",
    "who founded nvidia",
    "where is nvidia headquartered",
    "what does nvidia do",
    "tell me about nvidia",
    "how big is nvidia",
    "what is nvidia known for",
    "what is an h100",
    "what is cuda",
    "what does cuda stand for",
    "tell me about h100",
    "what is h100",
    "what docs do you have",
    "what sources do you have",
    "overview of all docs",
    "overview of the docs",
    "what is rag",
    "what is agentic rag",
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _is_live_query(question: str) -> bool:
    lowered = question.strip().lower()
    if _contains_any(lowered, LIVE_QUERY_TERMS) and not _contains_any(lowered, DOC_RAG_INFRA_TERMS):
        return True
    return False


def _is_casual_chat(question: str) -> bool:
    lowered = question.strip().lower()
    if lowered in CASUAL_CHAT_TERMS:
        return True
    if lowered in {"what day is it", "what date is it", "how are you", "who are you"}:
        return True
    if lowered.startswith("how are you") or lowered.startswith("how's it going") or lowered.startswith("how do you do"):
        return True
    # Gratitude / acknowledgement phrases — don't let these enter the RAG pipeline
    if lowered.startswith("thanks") or lowered.startswith("thank you") or lowered.startswith("thx"):
        return True
    if lowered.startswith("great,") or lowered.startswith("awesome,") or lowered.startswith("perfect,") or lowered.startswith("got it"):
        return True
    # Out-of-scope topics that have no NVIDIA-tech relevance
    if _contains_any(lowered, OUT_OF_SCOPE_TERMS) and not _contains_any(lowered, DOC_RAG_INFRA_TERMS):
        return True
    return False


def _is_general_knowledge_prompt(question: str) -> bool:
    lowered = question.strip().lower()
    return any(pattern.search(lowered) for pattern in GENERAL_KNOWLEDGE_PATTERNS)


def _is_known_direct_chat_factoid(lowered: str) -> bool:
    return any(lowered.startswith(prefix) for prefix in KNOWN_DIRECT_CHAT_FACTOIDS)


def _has_doc_request(lowered: str) -> bool:
    return _contains_any(lowered, GENERIC_DOC_TERMS) or "official docs" in lowered or "official documentation" in lowered


def _has_operational_doc_intent(lowered: str) -> bool:
    return _contains_any(lowered, DOC_ACTION_TERMS)


def _previous_user_context_prefers_doc_rag(previous_user_context: str) -> bool:
    if not previous_user_context:
        return False
    return (
        classify_question(previous_user_context) != QueryClass.general
        or _contains_any(previous_user_context, DOC_RAG_INFRA_TERMS)
        or (_contains_any(previous_user_context, NVIDIA_ENTITY_TERMS) and _has_doc_request(previous_user_context))
        or (_contains_any(previous_user_context, NVIDIA_ENTITY_TERMS) and _has_operational_doc_intent(previous_user_context))
    )


def classify_question(question: str) -> QueryClass:
    lowered = question.lower()
    best_class = QueryClass.general
    best_score = 0
    for query_class, terms in QUERY_CLASS_RULES:
        score = 0
        for term in terms:
            if term in lowered:
                score += max(1, len(term.split()))
        if score > best_score:
            best_score = score
            best_class = query_class
    if best_score > 0:
        return best_class
    return QueryClass.general


def is_recency_sensitive(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in RECENCY_TERMS)


_ENTITY_TERMS_FOR_ADAPTIVE = ("h100", "a100", "l40s", "h200", "nccl", "cuda", "nvlink", "dcgm", "gpudirect", "megatron")


def _adaptive_retrieval_params(question: str, query_class: QueryClass) -> tuple[int, float]:
    """Return (top_k, confidence_floor) dynamically based on query complexity.

    Heuristics:
    - Short, single-entity factoid (<=8 tokens, <=1 entity) → small top_k, higher floor
    - Complex analytical (>15 tokens or >=2 entities) → larger top_k, lower floor
    - Default → class-based baseline
    """
    token_count = len(tokenize(question))
    lowered = question.lower()
    entity_count = sum(1 for term in _ENTITY_TERMS_FOR_ADAPTIVE if term in lowered)

    if token_count <= 8 and entity_count <= 1:
        # Simple factoid: retrieve fewer, require higher confidence
        return 3, 0.35
    if token_count > 15 or entity_count >= 2:
        # Complex analytical or multi-entity: retrieve more, relax floor
        return 10, 0.22

    # Default: class-based baseline
    if query_class in {QueryClass.distributed_multi_gpu, QueryClass.hardware_topology}:
        return 7, 0.26
    return 5, 0.30


def build_query_plan(question: str, settings: Settings) -> QueryPlan:
    query_class = classify_question(question)
    expansions = EXPANSION_TERMS[query_class][:2]
    search_queries = [question] + [f"{question} {term}".strip() for term in expansions]
    recency_sensitive = is_recency_sensitive(question)
    top_k, confidence_floor = _adaptive_retrieval_params(question, query_class)
    return QueryPlan(
        query_class=query_class,
        search_queries=search_queries,
        source_families=CLASS_FAMILY_MAP[query_class],
        top_k=top_k,
        use_tavily_fallback=settings.use_tavily_fallback,
        confidence_floor=confidence_floor,
        max_retries=2,
        recency_sensitive=recency_sensitive,
    )


def llm_build_query_plan(
    question: str,
    settings: Settings,
    reasoner,
) -> tuple[QueryPlan, str]:
    """Build a query plan using LLM reasoning about the question.

    Returns (plan, method) where method is 'llm' or 'rule_fallback'.
    """
    if not reasoner.enabled:
        return build_query_plan(question, settings), "rule_fallback"

    prompt = (
        "You are a query planner for an NVIDIA AI infrastructure knowledge base.\n"
        "Given the user's question, produce a retrieval plan as JSON.\n\n"
        "Query classes:\n"
        "- training_optimization: training throughput, mixed precision, profiling, tensor cores, memory optimization\n"
        "- distributed_multi_gpu: multi-GPU scaling, NCCL, NVLink, parallelism strategies, collective communication\n"
        "- deployment_runtime: installing/deploying CUDA, containers, drivers, K8s GPU operator, cluster ops, storage systems\n"
        "- hardware_topology: specific GPU hardware (H100, A100, etc.), NVLink topology, server design, memory specs\n"
        "- general: questions that don't fit other categories\n\n"
        "Source families: core, distributed, infrastructure, advanced, hardware\n\n"
        "Respond with ONLY this JSON (no extra text):\n"
        '{"query_class": "...", "search_queries": ["q1", "q2"], '
        '"source_families": ["fam1", "fam2"], "top_k": N, '
        '"confidence_floor": F, "reasoning": "..."}\n\n'
        "Guidelines:\n"
        "- search_queries: 2-3 queries optimized for retrieval (include the original + reformulated variants)\n"
        "- top_k: number of chunks to retrieve (3-15)\n"
        "- confidence_floor: minimum confidence threshold (0.15-0.50)\n"
        "- source_families: 1-3 most relevant families\n\n"
        f"Question: {question}"
    )

    try:
        raw = reasoner.generate_text(prompt, model=settings.routing_model)

        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)

        # Validate query_class
        valid_classes = {e.value for e in QueryClass}
        qc_value = data.get("query_class", "general")
        if qc_value not in valid_classes:
            qc_value = "general"
        query_class = QueryClass(qc_value)

        # Validate search_queries
        search_queries = data.get("search_queries", [])
        if not isinstance(search_queries, list) or not search_queries:
            search_queries = [question]
        search_queries = [q for q in search_queries if isinstance(q, str) and q.strip()]
        if not search_queries:
            search_queries = [question]

        # Validate source_families
        allowed_families = {"core", "distributed", "infrastructure", "advanced", "hardware"}
        source_families = data.get("source_families", [])
        if isinstance(source_families, list):
            source_families = [f for f in source_families if f in allowed_families]
        if not source_families:
            source_families = CLASS_FAMILY_MAP[query_class]

        # Validate top_k and confidence_floor
        top_k = data.get("top_k", 5)
        if not isinstance(top_k, (int, float)):
            top_k = 5
        top_k = max(3, min(15, int(top_k)))

        confidence_floor = data.get("confidence_floor", 0.28)
        if not isinstance(confidence_floor, (int, float)):
            confidence_floor = 0.28
        confidence_floor = max(0.15, min(0.50, float(confidence_floor)))

        recency_sensitive = is_recency_sensitive(question)

        plan = QueryPlan(
            query_class=query_class,
            search_queries=search_queries,
            source_families=source_families,
            top_k=top_k,
            use_tavily_fallback=settings.use_tavily_fallback,
            confidence_floor=confidence_floor,
            max_retries=2,
            recency_sensitive=recency_sensitive,
        )
        return plan, "llm"

    except Exception as exc:
        _log.warning("LLM query planning failed (%s), falling back to rules", exc)
        return build_query_plan(question, settings), "rule_fallback"


def classify_assistant_mode(question: str, history: list[ChatTurn] | None = None) -> str:
    recent_user_turns = [turn.content for turn in (history or []) if turn.role == "user" and turn.content.strip()][-4:]
    previous_user_context = " ".join(recent_user_turns).lower()
    lowered = question.strip().lower()

    # Prompt injection defense: detect attempts to override system instructions
    _INJECTION_PREFIXES = (
        "ignore previous", "ignore all", "disregard", "forget your instructions",
        "you are now", "new instructions:", "system:", "system prompt:",
        "override:", "admin:", "jailbreak",
    )
    if any(lowered.startswith(prefix) for prefix in _INJECTION_PREFIXES):
        _log.warning("Prompt injection attempt detected: %s", question[:80])
        return "direct_chat"

    # Live data queries (weather, stocks, news) → Tavily search directly
    if _is_live_query(question):
        return "live_query"

    if _is_casual_chat(question):
        return "direct_chat"

    # Financial/market queries need live data — but guard against infra terms
    # so "GPU memory stock configuration" stays doc_rag
    if _contains_any(lowered, FINANCIAL_TERMS) and not _contains_any(lowered, DOC_RAG_INFRA_TERMS):
        return "live_query"

    if _is_known_direct_chat_factoid(lowered):
        return "direct_chat"

    # Recency-sensitive questions (weather, news, latest events) → route through
    # the RAG pipeline so Tavily web search handles them directly.
    if is_recency_sensitive(question) and not _contains_any(lowered, NVIDIA_ENTITY_TERMS):
        return "doc_rag"

    if _previous_user_context_prefers_doc_rag(previous_user_context) and (
        lowered in FOLLOW_UP_DOC_RAG_TERMS or len(tokenize(question)) <= 3
    ):
        return "doc_rag"

    if (
        _is_general_knowledge_prompt(question)
        and not _has_doc_request(lowered)
        and not _has_operational_doc_intent(lowered)
        and not _contains_any(lowered, DOC_RAG_INFRA_TERMS)
    ):
        return "direct_chat"

    if _has_doc_request(lowered):
        if (
            "official" in lowered
            or _contains_any(lowered, NVIDIA_ENTITY_TERMS)
            or _contains_any(lowered, DOC_RAG_INFRA_TERMS)
            or _has_operational_doc_intent(lowered)
            or _previous_user_context_prefers_doc_rag(previous_user_context)
        ):
            return "doc_rag"
        return "direct_chat"

    if classify_question(question) != QueryClass.general:
        return "doc_rag"

    if _contains_any(lowered, DOC_RAG_INFRA_TERMS):
        return "doc_rag"

    if _contains_any(lowered, NVIDIA_ENTITY_TERMS) and (
        _contains_any(lowered, NVIDIA_DOC_CONTEXT_TERMS) or _has_operational_doc_intent(lowered)
    ):
        return "doc_rag"

    if "official" in lowered and _previous_user_context_prefers_doc_rag(previous_user_context):
        return "doc_rag"

    # Last-chance catch: sufficiently long queries mentioning gpu/nvidia that weren't
    # caught by any earlier rule are almost certainly infrastructure questions.
    if ("gpu" in lowered or "nvidia" in lowered) and len(tokenize(question)) > 6:
        return "doc_rag"

    return "direct_chat"


def llm_classify_assistant_mode(
    question: str,
    history: list[ChatTurn] | None,
    reasoner,
    routing_model: str,
) -> tuple[str, str]:
    """Classify the user's question using an LLM for semantic understanding.

    Returns (mode, method) where mode is one of 'doc_rag', 'direct_chat', 'live_query'
    and method is 'llm' or 'rule_fallback'.
    """
    if not reasoner.enabled:
        return classify_assistant_mode(question, history), "rule_fallback"

    # Build context from recent history
    history_context = ""
    if history:
        recent_user_turns = [
            turn.content for turn in history
            if turn.role == "user" and turn.content.strip()
        ][-2:]
        if recent_user_turns:
            history_context = (
                "\nRecent conversation context:\n"
                + "\n".join(f"- {t}" for t in recent_user_turns)
                + "\n"
            )

    prompt = (
        "Classify the user's question into exactly one mode.\n\n"
        "Modes:\n"
        "- doc_rag: Questions about NVIDIA GPU infrastructure, CUDA, drivers, deployment, "
        "training, hardware specs, networking (NCCL, NVLink, InfiniBand), storage (BeeGFS, "
        "Lustre), scheduling (Slurm, K8s), or any technical documentation topic. "
        "Default for technical questions.\n"
        "- direct_chat: Casual greetings, general knowledge not about NVIDIA infrastructure, "
        "acknowledgements, out-of-scope topics (sports, cooking, etc.)\n"
        "- live_query: Questions requiring real-time data — weather, stock prices, current news, "
        "live events\n"
        f"{history_context}\n"
        'Respond with ONLY this JSON (no extra text):\n'
        '{"mode": "doc_rag|direct_chat|live_query", "reasoning": "brief explanation"}\n\n'
        f"Question: {question}"
    )

    try:
        raw = reasoner.generate_text(prompt, model=routing_model)

        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)

        mode = data.get("mode", "")
        valid_modes = {"doc_rag", "direct_chat", "live_query"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}' from LLM")

        return mode, "llm"

    except Exception as exc:
        _log.warning("LLM classification failed (%s), falling back to rules", exc)
        return classify_assistant_mode(question, history), "rule_fallback"


def rewrite_query(question: str, query_class: QueryClass, attempt: int = 1) -> str:
    hint = " ".join(EXPANSION_TERMS[query_class][: 2 + min(attempt, 1)])
    return f"{question} {hint}".strip()


# Map of product name literals → source IDs that should be boosted when the product is named in the query
_PRODUCT_NAME_SOURCE_BOOST: list[tuple[str, str, float]] = [
    ("beegfs", "beegfs", 0.50),
    ("lustre", "lustre", 0.50),
    ("container-toolkit", "container-toolkit", 0.50),
    ("container toolkit", "container-toolkit", 0.50),
    ("nvidia-container-runtime", "container-toolkit", 0.50),
    ("gpu operator", "gpu-operator", 0.50),
    ("gpu-operator", "gpu-operator", 0.50),
    ("gpudirect", "gpudirect-storage", 0.50),
    ("gpudirect storage", "gpudirect-storage", 0.50),
    ("dcgm", "dcgm", 0.50),
    ("nsight compute", "nsight-compute", 0.50),
    ("nsight-compute", "nsight-compute", 0.50),
    ("megatron", "megatron-core", 0.40),
    ("nemo", "nemo-performance", 0.40),
    ("fabric manager", "fabric-manager", 0.40),
    ("transformer engine", "transformer-engine", 0.40),
    ("nccl", "nccl", 0.48),
]


def generate_hyde_query(question: str, reasoner, settings: Settings) -> str:
    """Generate a hypothetical document for HyDE embedding.

    When USE_HYDE is disabled (default), returns the original query unchanged.
    When enabled, asks the LLM to generate a hypothetical answer passage
    that would be embedded instead of the raw question — improving semantic
    retrieval by shifting the query into document space.
    """
    if not settings.use_hyde:
        return question
    if not getattr(reasoner, "enabled", False):
        return question
    try:
        prompt = (
            "Write a short, factual paragraph (3-5 sentences) that would appear in "
            "an NVIDIA technical document and directly answers this question. "
            "Do NOT include any preamble — just the document-style passage.\n\n"
            f"Question: {question}"
        )
        hyde_doc = reasoner.generate_text(prompt, model=getattr(settings, "routing_model", None))
        return hyde_doc.strip() if hyde_doc and hyde_doc.strip() else question
    except Exception:
        _log.warning("HyDE generation failed, falling back to original query")
        return question


class CrossEncoderReranker:
    """Neural reranker using a cross-encoder model. Lazy-loads on first use."""

    _instance: CrossEncoderReranker | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None

    @classmethod
    def get_instance(cls) -> CrossEncoderReranker:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                _log.info("Cross-encoder model loaded: ms-marco-MiniLM-L-6-v2")
            except ImportError:
                _log.warning("sentence-transformers not installed, cross-encoder reranking disabled")
                raise

    def rerank(self, question: str, results: list[RetrieverResult], top_n: int = 10) -> list[RetrieverResult]:
        """Rerank results using cross-encoder scores. Returns top_n results."""
        if not results:
            return results

        try:
            self._ensure_model()
        except ImportError:
            return results

        # Create query-passage pairs
        pairs = [(question, r.chunk.content[:512]) for r in results]

        # Get cross-encoder scores
        scores = self._model.predict(pairs)

        # Combine cross-encoder scores with existing scores
        scored = []
        for result, ce_score in zip(results, scores):
            # Blend: 60% cross-encoder + 40% existing rerank score
            blended = 0.6 * float(ce_score) + 0.4 * result.rerank_score
            scored.append((result, blended))

        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, blended_score in scored[:top_n]:
            reranked.append(result.model_copy(update={"rerank_score": blended_score}))

        return reranked


def rerank_results(
    question: str,
    plan: QueryPlan,
    results: list[RetrieverResult],
    config: RerankConfig | None = None,
) -> list[RetrieverResult]:
    cfg = config or _DEFAULT_RERANK_CONFIG
    query_tokens = set(tokenize(question))
    lowered_question = question.lower()
    reranked: list[RetrieverResult] = []
    source_hints = QUERY_CLASS_SOURCE_HINTS.get(plan.query_class, {})
    # Compute per-query product-name boosts
    active_product_boosts: dict[str, float] = {}
    for phrase, source_id, boost in _PRODUCT_NAME_SOURCE_BOOST:
        if phrase in lowered_question:
            active_product_boosts[source_id] = max(active_product_boosts.get(source_id, 0.0), boost)
    for result in results:
        chunk_tokens = set(result.chunk.sparse_terms or tokenize(result.chunk.content))
        overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
        family_bonus = cfg.family_bonus if result.chunk.doc_family in plan.source_families else 0.0
        if plan.query_class == QueryClass.hardware_topology and result.chunk.doc_family == "hardware":
            family_bonus += cfg.hardware_family_bonus
        metadata_bonus = cfg.metadata_bonus if any(token in result.chunk.section_path.lower() for token in query_tokens) else 0.0
        source_bonus = source_hints.get(result.chunk.source_id, 0.0)
        # Apply product-name boost on top of class-level source hints
        source_bonus = max(source_bonus, active_product_boosts.get(result.chunk.source_id, 0.0))
        tag_bonus = cfg.tag_bonus if any(token in " ".join(result.chunk.product_tags).lower() for token in query_tokens) else 0.0
        rerank_score = result.score + (cfg.lexical_overlap_weight * overlap) + family_bonus + metadata_bonus + source_bonus + tag_bonus
        # R5: Chunk recency weighting — gently favor fresher content
        recency_bonus = 0.0
        retrieved_at = getattr(result.chunk, 'retrieved_at', None)
        if retrieved_at:
            try:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - dt).days
                if age_days <= 30:
                    recency_bonus = 0.02
                elif age_days <= 90:
                    recency_bonus = 0.01
            except (ValueError, TypeError):
                pass
        rerank_score += recency_bonus
        reranked.append(result.model_copy(update={"rerank_score": rerank_score}))
    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    # Per-source diversity cap: adaptive to plan size to avoid premature diversification
    max_per_source = min(cfg.max_per_source, max(plan.top_k - 1, 1))
    source_counts: dict[str, int] = {}
    diverse: list[RetrieverResult] = []
    overflow: list[RetrieverResult] = []
    for result in reranked:
        sid = result.chunk.source_id
        if source_counts.get(sid, 0) < max_per_source:
            diverse.append(result)
            source_counts[sid] = source_counts.get(sid, 0) + 1
        else:
            overflow.append(result)
    return diverse + overflow


# Sources that are considered "noise" when queries explicitly name a different product.
# When a product name boost is active and a chunk comes from one of these high-volume
# generic sources, apply a stricter threshold.
_GENERIC_HIGH_VOLUME_SOURCES = {"cuda-install", "cuda-programming-guide", "cuda-best-practices"}


def grade_results(question: str, plan: QueryPlan, results: list[RetrieverResult]) -> tuple[list[RetrieverResult], list[str]]:
    query_tokens = set(tokenize(question))
    lowered_question = question.lower()
    accepted: list[RetrieverResult] = []
    rejected: list[str] = []
    # Determine if query has an explicit product focus (product-name boost is active)
    has_product_focus = any(phrase in lowered_question for phrase, _, _ in _PRODUCT_NAME_SOURCE_BOOST)
    for result in results:
        chunk_tokens = set(result.chunk.sparse_terms or tokenize(result.chunk.content))
        overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
        family_match = result.chunk.doc_family in plan.source_families
        if plan.query_class == QueryClass.hardware_topology:
            threshold = 0.22
        elif plan.query_class == QueryClass.deployment_runtime:
            threshold = 0.25
        else:
            threshold = 0.18
        # When a specific product is named, apply strict threshold to generic high-volume sources
        if has_product_focus and result.chunk.source_id in _GENERIC_HIGH_VOLUME_SOURCES:
            threshold = max(threshold, 0.35)
        if result.rerank_score >= threshold or (family_match and overlap >= 0.20 and result.rerank_score >= 0.12):
            accepted.append(result)
        else:
            rejected.append(result.chunk.id)
    return accepted, rejected


def estimate_confidence(results: list[RetrieverResult]) -> float:
    if not results:
        return 0.0
    scores = [r.rerank_score for r in results[:3]]
    # Weighted average emphasizing the best result: 50% top-1, 30% top-2, 20% top-3
    weights = (0.5, 0.3, 0.2)
    weighted = sum(s * w for s, w in zip(scores, weights[:len(scores)]))
    total_weight = sum(weights[:len(scores)])
    return round(weighted / total_weight, 4)


def needs_retry(plan: QueryPlan, results: list[RetrieverResult]) -> bool:
    return estimate_confidence(results) < plan.confidence_floor


# R4: Query expansion — expand common abbreviations and fix typos
_ABBREVIATION_MAP: dict[str, str] = {
    "AR": "all-reduce",
    "TP": "tensor parallelism",
    "DP": "data parallelism",
    "PP": "pipeline parallelism",
    "NVL": "NVLink",
    "FSDP": "fully sharded data parallel",
    "MoE": "mixture of experts",
    "GDS": "GPUDirect Storage",
    "RDMA": "GPUDirect RDMA",
    "RoCE": "RDMA over Converged Ethernet",
    "IB": "InfiniBand",
    "OOD": "out of distribution",
    "FP16": "half precision",
    "BF16": "bfloat16",
    "FP8": "FP8 precision",
    "MIG": "Multi-Instance GPU",
    "vGPU": "virtual GPU",
    "MPS": "Multi-Process Service",
    "UCX": "Unified Communication X",
    "SHARP": "Scalable Hierarchical Aggregation",
    "DPU": "data processing unit",
    "BMC": "baseboard management controller",
}


def expand_query_abbreviations(query: str) -> str:
    """Expand known abbreviations in a query to improve retrieval recall.

    Only expands abbreviations that appear as whole words (not substrings).
    Returns the original query with expansions appended if any are found.
    """
    import re as _re
    expansions: list[str] = []
    for abbr, full in _ABBREVIATION_MAP.items():
        # Match whole-word abbreviations (case-sensitive for short ones, insensitive for 3+ chars)
        if len(abbr) <= 2:
            pattern = r'\b' + _re.escape(abbr) + r'\b'
            if _re.search(pattern, query):
                expansions.append(full)
        else:
            pattern = r'\b' + _re.escape(abbr) + r'\b'
            if _re.search(pattern, query, _re.IGNORECASE):
                expansions.append(full)

    if expansions:
        return query + " " + " ".join(expansions)
    return query


class RetrievalService:
    def __init__(self, settings: Settings, sources: list[DocumentSource], index: SearchIndex) -> None:
        self.settings = settings
        self.sources = sources
        self.index = index

    def list_sources(self) -> list[DocumentSource]:
        return self.sources

    def build_plan(self, question: str) -> QueryPlan:
        return build_query_plan(question, self.settings)

    def run_retrieval_pass(self, question: str, plan: QueryPlan, query: str) -> tuple[list[RetrieverResult], list[str], float, list[TraceEvent]]:
        trace: list[TraceEvent] = []
        candidates: dict[str, RetrieverResult] = {}
        queries = [query]
        if query == question:
            queries = plan.search_queries[:3]
        retrieval_top_k = max(plan.top_k * 4, 12)

        total_retrieved = 0
        for query_idx, active_query in enumerate(queries):
            # R4: expand abbreviations in the active query for better recall
            expanded_query = expand_query_abbreviations(active_query)
            try:
                retrieved = self.index.search(expanded_query, top_k=retrieval_top_k, families=plan.source_families)
            except Exception as exc:
                _log.warning("Index search failed for query %r (%s: %s), skipping", active_query[:80], type(exc).__name__, str(exc)[:120])
                retrieved = []
            total_retrieved += len(retrieved)
            trace.append(
                TraceEvent(
                    type="retrieval",
                    message=f"Retrieved {len(retrieved)} candidates from the hybrid index",
                    payload={
                        "stage": "tool_request",
                        "status": "request",
                        "tool": "nvidia_docs",
                        "tool_label": "NVIDIA Docs",
                        "source_kind": "corpus",
                        "brand": "nvidia",
                        "query": active_query,
                        "families": plan.source_families,
                        "top_k": retrieval_top_k,
                    },
                )
            )
            for item in retrieved:
                existing = candidates.get(item.chunk.id)
                if existing is None or item.score > existing.score:
                    candidates[item.chunk.id] = item

            # Early stopping: if first query already yields high confidence, skip expansion queries
            if query_idx < len(queries) - 1 and candidates:
                preliminary = rerank_results(question, plan, list(candidates.values()), self.settings.rerank_config)
                prelim_conf = estimate_confidence(preliminary)
                if prelim_conf > EARLY_STOP_CONFIDENCE:
                    _log.info(
                        "Early stop after query %d/%d with confidence %.3f (threshold %.3f)",
                        query_idx + 1, len(queries), prelim_conf, EARLY_STOP_CONFIDENCE,
                    )
                    break

        reranked = rerank_results(question, plan, list(candidates.values()), self.settings.rerank_config)
        # R1: Optional cross-encoder neural reranking
        if self.settings.use_cross_encoder:
            try:
                ce_reranker = CrossEncoderReranker.get_instance()
                reranked = ce_reranker.rerank(question, reranked, top_n=min(plan.top_k * 2, len(reranked)))
                _log.info("Cross-encoder reranked %d results", len(reranked))
            except Exception as exc:
                _log.warning("Cross-encoder reranking failed (%s), using TF-IDF reranking", type(exc).__name__)
        confidence = estimate_confidence(reranked)
        trace.append(
            TraceEvent(
                type="rerank",
                message="Reranked candidates using lexical overlap, source-family routing, and metadata features",
                payload={
                    "stage": "tool_result",
                    "status": "result",
                    "tool": "nvidia_docs",
                    "tool_label": "NVIDIA Docs",
                    "source_kind": "corpus",
                    "brand": "nvidia",
                    "top_chunk_ids": [item.chunk.id for item in reranked[:3]],
                    "top_docs": [
                        {
                            "chunk_id": item.chunk.id,
                            "title": item.chunk.title,
                            "section_path": item.chunk.section_path,
                            "url": item.chunk.url,
                            "source_kind": item.chunk.source_kind,
                        }
                        for item in reranked[:4]
                    ],
                    "confidence": confidence,
                    "retrieved_total": total_retrieved,
                },
            )
        )
        accepted, rejected = grade_results(question, plan, reranked)
        trace.append(
            TraceEvent(
                type="document_grading",
                message="Filtered weak chunks before synthesis",
                payload={
                    "stage": "evidence_selection",
                    "tool": "nvidia_docs",
                    "tool_label": "NVIDIA Docs",
                    "source_kind": "corpus",
                    "brand": "nvidia",
                    "accepted": [
                        {
                            "chunk_id": item.chunk.id,
                            "title": item.chunk.title,
                            "section_path": item.chunk.section_path,
                            "url": item.chunk.url,
                            "source_kind": item.chunk.source_kind,
                        }
                        for item in accepted[:5]
                    ],
                    "rejected": rejected[:8],
                    "total_count": len(reranked),
                    "kept_count": len(accepted),
                    "grades": ["pass"] * len(accepted) + ["fail"] * len(rejected),
                },
            )
        )
        return accepted[: plan.top_k], rejected, estimate_confidence(accepted), trace

    def merge_results(self, question: str, plan: QueryPlan, left: list[RetrieverResult], right: list[RetrieverResult]) -> tuple[list[RetrieverResult], list[str], float]:
        merged: dict[str, RetrieverResult] = {item.chunk.id: item for item in left}
        for item in right:
            existing = merged.get(item.chunk.id)
            if existing is None or item.score > existing.score:
                merged[item.chunk.id] = item
        reranked = rerank_results(question, plan, list(merged.values()), self.settings.rerank_config)
        accepted, rejected = grade_results(question, plan, reranked)
        return accepted[: plan.top_k], rejected, estimate_confidence(accepted)

    def search(self, question: str) -> SearchDebugResponse:
        plan = self.build_plan(question)
        trace = [
            TraceEvent(
                type="classification",
                message=f"Classified question as {plan.query_class.value}",
                payload={"source_families": plan.source_families, "search_queries": plan.search_queries[:2]},
            )
        ]

        results, rejected, confidence, pass_trace = self.run_retrieval_pass(question, plan, question)
        trace.extend(pass_trace)
        rewritten_query = None
        retry_count = 0

        if needs_retry(plan, results):
            retry_count = 1
            rewritten_query = rewrite_query(question, plan.query_class, retry_count)
            retry_results, retry_rejected, retry_confidence, retry_trace = self.run_retrieval_pass(question, plan, rewritten_query)
            trace.extend(
                [
                    TraceEvent(
                        type="rewrite",
                        message="Ran a second retrieval pass with a rewritten query",
                        payload={"rewritten_query": rewritten_query},
                    )
                ]
            )
            trace.extend(retry_trace)
            results, rejected, confidence = self.merge_results(question, plan, results, retry_results)
            rejected = list(dict.fromkeys(rejected + retry_rejected))
            confidence = max(confidence, retry_confidence)

        return SearchDebugResponse(
            plan=plan,
            results=results,
            rewritten_query=rewritten_query,
            confidence=confidence,
            retry_count=retry_count,
            rejected_chunk_ids=rejected,
            trace=trace,
        )

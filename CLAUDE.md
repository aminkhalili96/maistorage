# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An **agentic RAG assistant** for NVIDIA AI infrastructure questions. The system retrieves from a bundled offline NVIDIA corpus, synthesizes grounded answers via OpenAI, and exposes a React chat UI with citations, agent trace, and trust signals (`corpus-backed`, `web-backed`, `llm-knowledge`, `insufficient-evidence`).

## Commands

### Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Run tests (from repo root):
```bash
PYTHONPATH=backend backend/.venv/bin/pytest backend/tests
```

Run a single test:
```bash
PYTHONPATH=backend backend/.venv/bin/pytest backend/tests/test_retrieval.py
```

Core smoke check:
```bash
PYTHONPATH=backend backend/.venv/bin/pytest backend/tests/test_simple_chat_flow.py backend/tests/test_chatbot_smoke.py -v
```

### Frontend

```bash
cd frontend
npm install
npm run dev      # http://127.0.0.1:5173
npm run build
npm test         # vitest run
```

Frontend tests from repo root:
```bash
cd frontend && npm test
```

### Docker

```bash
docker compose up --build
```

### Corpus & evaluation scripts (run from repo root with backend venv)

```bash
# Rebuild normalized chunks
PYTHONPATH=backend backend/.venv/bin/python scripts/normalize_corpus.py

# Populate Pinecone with demo set
PYTHONPATH=backend backend/.venv/bin/python scripts/bootstrap_assessment_index.py --demo-corpus

# Run full evaluation artifact suite
LANGSMITH_TRACING=false backend/.venv/bin/python scripts/run_evaluation_suite.py --skip-verification

# Download raw corpus from NVIDIA docs
PYTHONPATH=backend backend/.venv/bin/python scripts/download_corpus.py

# Build 54-question golden benchmark from component question files
PYTHONPATH=backend backend/.venv/bin/python scripts/build_agentic_golden.py

# Compare keyword vs OpenAI embedder performance
PYTHONPATH=backend backend/.venv/bin/python scripts/compare_embeddings.py
```

## Environment

Copy `.env.example` to `.env`. Two modes:

- **`APP_MODE=dev`** (default): uses in-memory keyword index, no Pinecone/OpenAI required
- **`APP_MODE=assessment`**: requires `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `USE_PINECONE=true`, `EMBEDDER_PROVIDER=openai`

Settings are validated at startup via `Settings.validate_runtime()` — assessment mode will raise on missing credentials.

### Optional feature flags

| Variable | Default | Effect |
|----------|---------|--------|
| `SEMANTIC_CACHE_ENABLED` | `false` | Enable in-memory semantic query cache |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Cosine similarity threshold for cache hits |
| `QUERY_DECOMPOSITION_ENABLED` | `false` | Enable LLM-based multi-part query decomposition |
| `USE_TAVILY_FALLBACK` | `false` | Enable Tavily web search fallback (requires `TAVILY_API_KEY`) |

## Architecture

### Request flow

```
POST /api/chat/stream
  → AgentService.stream()           # progressive SSE via asyncio.Queue
  → AgentService.run()              # checks SemanticCache first (context-aware key)
  → _run_with_optional_trace()
    → _reformulate_follow_up()      # LLM or static follow-up reformulation
    → _format_history_context()     # last 2-3 Q&A pairs for synthesis prompt
    → classify_assistant_mode()     # uses reformulated query for classification
  → LangGraph graph (or _run_without_graph shim)
```

The LangGraph graph nodes run in order:
```
classify → retrieve → document_grading
  → [rewrite_if_needed | fallback_if_needed]
  → generate → self_reflect → grounding_check → answer_quality_check
  → [post_generation_fallback → generate → self_reflect → grounding_check → answer_quality_check]
```

Conditional routing:
- `document_grading` → rewrite (1x) if confidence below floor, then fallback if still low
- `answer_quality_check` → `post_generation_fallback` if quality fails and Tavily not yet tried; else rewrite up to `max_retries`
- `post_generation_fallback` → second `generate` pass if Tavily returned results

The SSE stream emits typed events **progressively** as each node completes (not all at once after the pipeline finishes):
`classification`, `retrieval`, `rerank`, `document_grading`, `rewrite`, `fallback`, `generation`, `generation_error`, `self_reflect`, `grounding_check`, `answer_quality_check`, `citation`, `answer_chunk`, `done`

### Service wiring (`backend/app/runtime.py`)

`get_services()` (double-checked locking singleton) wires:
- `Settings.raw_md_root` → `data/corpus/raw/markdown` (markdown content directory)
- `SearchIndex` → either `PineconeHybridIndex` or `InMemoryHybridIndex`
- `RetrievalService` → query planning, keyword/semantic search, confidence scoring, adaptive retrieval
- `IngestionService` → bootstraps corpus at startup from `data/corpus/normalized/*.jsonl`
- `AgentService` → LangGraph workflow + OpenAI generation + Tavily fallback + SemanticCache
- `EvaluationService` → retrieval benchmark and RAGAS evaluation against golden questions

### Key modules

| Path | Purpose |
|------|---------|
| `backend/app/api.py` | FastAPI router: `/health`, `/api/sources`, `/api/ingest/status`, `/api/ingest/start`, `/api/search/debug`, `/api/evals/retrieval`, `/api/evals/ragas`, `/api/chat/stream` |
| `backend/app/services/agent.py` | LangGraph graph, all node/routing logic, progressive SSE streaming, SemanticCache, Self-RAG |
| `backend/app/services/retrieval.py` | Query classification, plan building, adaptive retrieval, reranking, confidence |
| `backend/app/services/indexes.py` | `InMemoryHybridIndex` and `PineconeHybridIndex` |
| `backend/app/services/providers.py` | `OpenAIReasoner`, `TavilyClient`, embedder factory |
| `backend/app/services/ingestion.py` | Corpus bootstrap, JSONL load, Pinecone upsert |
| `backend/app/services/evaluation.py` | Retrieval benchmarks, RAGAS runner |
| `backend/app/corpus.py` | JSONL chunk loading, source grouping |
| `backend/app/models.py` | Pydantic models: `ChunkRecord`, `QueryPlan`, `AgentRunState`, `Citation`, `TraceEvent` |
| `backend/app/config.py` | `Settings` dataclass + `get_settings()` (dotenv-backed, LRU-cached) |
| `backend/app/runtime.py` | `AppServices` + `get_services()` thread-safe singleton |
| `backend/app/services/chunking.py` | Text splitting, nav-element filtering, PDF cleanup, markdown section extraction |
| `frontend/src/App.tsx` | Main React component: chat, citations panel, agent trace, corpus status |
| `frontend/src/api.ts` | SSE client, typed event parsing |
| `frontend/src/types.ts` | Frontend TypeScript types mirroring backend models |
| `frontend/src/components/CapyThinking.tsx` | Pixel-art capybara thinking animation during pipeline execution |
| `frontend/src/components/CorpusPanel.tsx` | Left sidebar: browsable corpus source list, PDF viewer modal |
| `frontend/src/components/ThinkingBlock.tsx` | BurstSpinner + structured trace panel with tool cards and step rendering |
| `frontend/src/components/MetaBar.tsx` | Trust badges, confidence display, grounding/quality status chips |
| `frontend/src/components/AnswerContent.tsx` | Markdown rendering with inline citation refs and streaming cursor |
| `frontend/src/components/MessageActions.tsx` | CopyButton and FeedbackButtons (thumbs up/down) |
| `frontend/src/components/CitationsPanel.tsx` | Citation card list with Docs/Web source kind badges |

### Data layout

- `data/corpus/normalized/*.jsonl` — one JSONL file per source (36 files), each line is a `ChunkRecord`
- `data/corpus/raw/html/` — raw HTML source files (downloaded from NVIDIA docs)
- `data/corpus/raw/pdfs/` — raw PDF source files
- `data/corpus/raw/markdown/` — handcrafted markdown content for hardware datasheets, offline sources, field guide sections, and sources where web scraping produces garbage
- `data/demo_chunks.json` — curated subset used for fast Pinecone bootstrap
- `data/evals/agentic_golden_questions.json` — 54-question benchmark (50 corpus-backed RAGAS-scored, 2 direct-chat, 1 refusal, 1 recency fallback; 23 multi-source, 31 of 36 sources covered)
- `data/evals/ragas_slim_10.json` — curated 10-question RAGAS subset (7 categories, all 4 response modes, 18 sources; cost ~$0.35-0.70)
- `data/evals/golden_questions.json` — older 46-question set, superseded by agentic version
- `data/evals/harder_questions.json` — 6 advanced questions referencing sources outside main corpus
- `data/evals/job_requirement_questions.json` — 8 JD-aligned questions
- `data/evals/jd_offline_extension_questions.json` — 7 infrastructure questions (Slurm, K8s, BeeGFS, etc.)
- `data/evals/deployment_eval_questions.json` — 8 deployment/inference-focused eval questions
- `data/evals/results/` — timestamped evaluation artifact bundles
- `data/corpus/manifest.json` — corpus snapshot metadata
- `data/sources/nvidia_sources.json` — source registry (`DocumentSource` list)

### Trust model

Every answer carries a `response_mode`:
- `corpus-backed` — answer grounded in the offline NVIDIA corpus
- `web-backed` — Tavily fallback was used (includes `live_query` route for weather/stock/news)
- `llm-knowledge` — OpenAI general-knowledge fallback (corpus + Tavily both exhausted)
- `insufficient-evidence` — refusal; truly unable to answer (last resort)
- `direct-chat` — conversational turn, no retrieval (greetings, follow-ups)

The classifier routes queries into three `assistant_mode` values: `doc_rag` (corpus retrieval), `direct_chat` (conversational), `live_query` (real-time data via Tavily — weather, stock prices, news). The `live_query` route bypasses the corpus pipeline entirely and returns `web-backed` response mode.

**5-mode response model** (first 4 form the fallback chain):
1. Corpus retrieval → grade → (rewrite 1x, LLM-powered) → generate → self-reflect → grounding/quality check
2. Post-generation Tavily: if quality fails and Tavily not yet tried → retry with web results (now a proper LangGraph node: `post_generation_fallback`)
3. LLM general-knowledge: if all layers exhausted → OpenAI answers from its own knowledge → `llm-knowledge`
4. Refusal: if nothing works → `insufficient-evidence`

**Grounding check** (strengthened):
- Checks that ≥60% of substantive paragraphs contain `[N]` citation markers
- Fails immediately if the LLM used hedging phrases ("based on my knowledge", "I believe", etc.)
- Fails if `_answer_says_insufficient()` is True

**Citation enforcement** (strengthened):
- `_ensure_citations()` only adds `[N]` markers when paragraph tokens overlap with the cited chunk's sparse terms (≥2 overlapping tokens required) — no more mechanical force-adding

**Self-RAG reflection** (`self_reflect` node):
- After `generate`, asks the LLM to score the answer on relevance/groundedness/completeness (1-5)
- If `groundedness < 3`, forces `grounding_passed = False` to trigger the fallback chain
- Skips gracefully when reasoner is disabled

### Multi-turn conversation memory

`_reformulate_follow_up()` replaces the old `_contextualize_query()` for follow-up handling:
- Detects follow-ups using the same heuristic (≤8 tokens or pronoun start)
- When `self.reasoner.enabled`: calls the LLM with last 2 Q&A pairs to rewrite as a standalone query (method `"llm"`)
- When LLM unavailable or fails: falls back to static string concatenation (method `"static"`)
- Emits `query_reformulation` SSE trace event with `original_query`, `reformulated_query`, and `method`
- Runs *before* `classify_assistant_mode()` so the classifier sees the reformulated query (e.g., "How much memory does it have?" → "How much memory does the NVIDIA H100 GPU have?" → classified as `doc_rag`)

`_format_history_context()` formats last 2-3 Q&A pairs as a string. This is:
- Passed to `_synthesize_answer()` as an appendix to the synthesis prompt (after context blocks, with instruction to use only for continuity)
- Passed to `_llm_knowledge_answer()` for the general-knowledge fallback
- Stored in `GraphState.history_context` so graph nodes can access it

`_cache_key()` builds a context-aware semantic cache key by concatenating last 2 user turns with the question, so the same follow-up question with different prior context gets different cache entries.

### Adaptive retrieval

`_adaptive_retrieval_params()` in `retrieval.py` dynamically sets `top_k` and `confidence_floor` based on query complexity:
- Short factoid (≤8 tokens, ≤1 entity) → `top_k=3`, `confidence_floor=0.35`
- Complex analytical (>15 tokens or ≥2 entities) → `top_k=10`, `confidence_floor=0.22`
- Default → class-based baseline (5–7, 0.26–0.30)

### Early stopping in retrieval

`run_retrieval_pass()` runs up to 3 search queries per pass (original + 2 expansion terms). After each query, it checks if preliminary confidence exceeds `EARLY_STOP_CONFIDENCE` (0.75). If so, remaining expansion queries are skipped — saves latency for straightforward queries that already have strong matches.

### LLM-based query rewriting

`_graph_rewrite()` first attempts an LLM-powered rewrite with a focused prompt:
> "You are a search query optimizer for an NVIDIA infrastructure documentation system. The following query returned low-confidence results. Rewrite it to be more specific, technical, and likely to match NVIDIA documentation..."

Falls back to static expansion terms if the LLM call fails or returns something too similar to the original. The trace event includes `rewrite_method: "llm" | "static_expansion"`.

### Semantic caching

`SemanticCache` (in `agent.py`) caches `AgentRunState` objects keyed by query embedding similarity:
- Cosine similarity threshold: `SEMANTIC_CACHE_THRESHOLD` (default 0.92)
- LRU eviction at 128 entries
- Thread-safe via `threading.Lock`
- Disabled by default; enable with `SEMANTIC_CACHE_ENABLED=true`
- Cache is checked before the full pipeline and populated after

### Progressive SSE streaming

`stream()` uses `asyncio.Queue` + `loop.call_soon_threadsafe` to emit events as each pipeline stage completes:
- `_thread_local_emit` thread-local stores the emit callback during sync pipeline execution
- `_append_trace()` and `_graph_retrieve()` call the emit callback for each event
- Trace events appear in the UI immediately as each node finishes, not all at once after the full pipeline

### Query decomposition

When `QUERY_DECOMPOSITION_ENABLED=true`, `_should_decompose()` detects multi-part questions (contains "and/vs/compare/difference between", multiple `?`, or >20 tokens with ≥2 entity terms). For these:
1. `_decompose_question()` calls the LLM to split into 2-3 focused sub-questions
2. `_graph_retrieve()` runs retrieval for each sub-question independently
3. Results are merged via `RetrievalService.merge_results()` before synthesis

### Reranking config

`RerankConfig` dataclass in `config.py` holds all reranking weights:
- `lexical_overlap_weight`, `family_bonus`, `hardware_family_bonus`, `metadata_bonus`, `tag_bonus`, `max_per_source`
- Passed explicitly to `rerank_results()` so weights can be tuned and tested independently

### Embedder providers

Controlled by `EMBEDDER_PROVIDER`:
- `keyword` — TF-IDF-style local keyword baseline (default in dev)
- `openai` — OpenAI `text-embedding-3-large` at 3072 dimensions (required in assessment mode)

## Gotchas

- **Thread-safe singleton**: `get_services()` uses double-checked locking (not `@lru_cache`). Tests call `get_services.cache_clear()` (which calls `reset_services()`) in `conftest.py`. Any new test file that modifies env vars must do the same.
- **`get_settings()` is still `@lru_cache`**: tests also call `get_settings.cache_clear()` in `conftest.py`.
- **Tests always run in dev mode**: `conftest.py` forces `APP_MODE=dev`, `USE_PINECONE=false`, `EMBEDDER_PROVIDER=keyword`. Tests never hit Pinecone or OpenAI.
- **`PYTHONPATH=backend` required**: All backend test and script invocations must set this (from repo root) so `import app.*` resolves correctly.
- **LangSmith tracing during evals**: Prefix eval script commands with `LANGSMITH_TRACING=false` to avoid quota burn.
- **OpenAI model costs**: 4-tier model strategy — GPT-5.4 Nano for routing (classification, decomposition), GPT-5 Mini for pipeline steps (rewriting, reformulation, self-reflection), GPT-5.4 for user-facing synthesis. Classification costs ~50x less than synthesis.
- **`get_settings` cache after env changes**: If you change a `.env` value mid-session, call `get_settings.cache_clear()` before the next request or restart uvicorn.
- **`_thread_local_emit`**: The progressive SSE emit callback is stored in a thread-local. It is set at the start of each `stream()` call and cleared in the `finally` block. Never set it manually outside of `stream()`.
- **Semantic cache disabled by default**: `SEMANTIC_CACHE_ENABLED=false` in dev. The cache is only created when both `semantic_cache_enabled=True` and an `embedder` is passed to `AgentService`.
- **Query decomposition disabled by default**: `QUERY_DECOMPOSITION_ENABLED=false`. Enable only when OpenAI is available; decomposition silently falls back to the original question if the LLM call fails.
- **CORS origins hardcoded**: `main.py` hardcodes `127.0.0.1:5173`, `localhost:5173`, `127.0.0.1:5174`, `localhost:5174` plus `settings.cors_origin`. Docker frontend on port 4173 must use the `FRONTEND_ORIGIN` env var.
- **Docker frontend port is 4173, not 5173**: `docker-compose.yml` maps port 4173:80 (Nginx serves the prod build). Dev server is 5173.
- **Markdown sources**: Sources with `doc_type: "markdown"` (hardware datasheets, offline notes, field guide) are chunked from `data/corpus/raw/markdown/{source_id}/content.md`. The ingestion pipeline checks the markdown directory for ALL sources, regardless of `doc_type` — so you can supplement any HTML/PDF source with additional markdown content.

## Demo RAG Questions

Three representative questions for smoke-testing the full pipeline:

| Question | Expected mode |
|----------|---------------|
| `Why is 4-GPU training scaling poorly?` | corpus-backed |
| `What are the key tuning parameters for NCCL to maximize bandwidth?` | corpus-backed |
| `What changed in the latest NVIDIA Container Toolkit release?` | web-backed (recency fallback) |

Full 54-question benchmark: `data/evals/agentic_golden_questions.json`
Demo playbook with talking points: `docs/interview-demo-playbook.md`
Curated demo questions with expected answers: `docs/interview-demo-questions.md`

## Interview & Assessment Context

**Company**: MaiStorage Technology Sdn Bhd, Puchong, Selangor (Phison group — storage solutions)
**Roles applied**: Lead AI Solution & Deployment Engineer (primary); AI Engineer R&D; AI Engineer Solution
**Assessment**: Task 2, Question 1 — Agentic RAG
**Demo window**: 15–20 minutes

### Assessment Q1 checklist
- [x] Agentic RAG that retrieves chunks correctly
- [x] Working prototype demo
- [x] Thought process and implementation flow discussion
- [x] Traditional RAG vs agentic RAG comparison
- [x] Test cases to assure quality
- [x] (Bonus) Citations handling
- [x] (Bonus) Optimized retrieval (accuracy + performance)

### Why this corpus fits the Lead AI Solution JD
MaiStorage sells on-prem NVIDIA AI server clusters to enterprise clients. The corpus maps directly to their customers' real questions:
- `nccl`, `fabric-manager` → cluster networking and multi-GPU scaling
- `gpu-operator`, `container-toolkit` → Kubernetes deployment of AI workloads
- `gpudirect-storage`, `dgx-basepod` → storage architecture for GPU training clusters
- `a100`, `h100`, `h200`, `l40s` → hardware selection and sizing

### Corpus gaps closed (March 2026 expansion)
- `tensorrt` — inference optimization (added as markdown)
- `triton-inference-server` — model serving (added as markdown)
- `nvidia-nim` — NVIDIA Inference Microservices (added as markdown)
- `nvidia-ai-enterprise` — enterprise software platform (added as markdown)
- Hardware datasheets (`h100`, `h200`, `a100`, `l40s`) — replaced marketing page scrapes with technical markdown
- 7 offline infrastructure sources — Slurm, Kubernetes, BeeGFS, Lustre, MinIO, mdraid, Docker CI/CD (markdown)
- 4 field guide sections — expanded from ~1 paragraph to ~800 words each (markdown)

### Remaining gaps (lower priority)
- `sharp` — NVIDIA SHARP InfiniBand aggregation (registered, not yet downloaded)
- `mlnx-ofed` — Mellanox OFED driver stack (registered, not yet downloaded)
- `enterprise-reference-architecture` — architecture whitepaper PDF (registered, not yet downloaded)

### Slide-ready benchmark numbers (last successful eval run)
**Retrieval:** hit@5=1.0, MRR=1.0, nDCG@5=0.88, routing@3=1.0
**RAGAS 10-question slim set (GPT-5.4 generation, gpt-4.1 judge, March 18 2026):** faithfulness=0.694, answer_relevancy=0.735, context_precision=0.683, context_recall=0.675
**Earlier 46-question RAGAS (Gemini generation, gemini-2.5-flash judge):** faithfulness=0.91, context_precision=0.87, context_recall=0.80, answer_relevancy=0.38
**Ablation:** keyword-only hit@5=0.75 → keyword+grading=1.0 → full agentic=only mode with correct web-backed routing

**Talking point:** The 10-question slim set is the current authoritative RAGAS benchmark (run on OpenAI GPT-5.4 after the Gemini-to-OpenAI migration). The earlier 46-question run used Gemini models and scored higher on faithfulness but lower on answer_relevancy due to more refusal cases. Both sets are documented in `docs/evaluation-evidence.md`.

## Current State

The project is **working end-to-end**. 212 backend tests pass (265 total; ~50 pre-existing failures in benchmark/eval suites that require live API credentials); 19 frontend tests pass. Corpus expanded from 21 to 36 normalized sources (~7,300 indexed chunks in dev mode, ~12,000 in JSONL files). Frontend decomposed into 7 component files (App.tsx reduced from ~973 to ~310 lines).

Quick smoke check:
```bash
PYTHONPATH=backend backend/.venv/bin/pytest backend/tests/test_simple_chat_flow.py backend/tests/test_chatbot_smoke.py -v
```

Full suite:
```bash
PYTHONPATH=backend backend/.venv/bin/pytest backend/tests -q
```

19 backend test files in `backend/tests/`. Notable specialized suites:
- `test_assistant_router_matrix.py` — 25 parametrized router classification cases
- `test_rag_benchmark_suite.py` — full 46-question corpus benchmark
- `test_metrics.py` — unit tests for retrieval metrics (hit@k, MRR, nDCG)

## Changelog

### 4-Tier Model Strategy — March 2026

Per-task model optimization: cheapest where simple, best quality for user-facing output.

| Item | Change |
|------|--------|
| **4-tier model assignment** | GPT-5.4 Nano ($0.20/1M input) for routing (classification, decomposition). GPT-5 Mini ($0.25/1M) for pipeline (rewriting, reformulation, self-reflection). GPT-5.4 ($2.50/1M) for synthesis, direct chat, LLM-knowledge fallback. Classification costs ~50x less than synthesis. |
| **Config: `openai_routing_model`** | New `Settings` field + `routing_model` property + `OPENAI_ROUTING_MODEL` env var. `ALLOWED_OPENAI_MODELS` updated to `("gpt-5.4-nano", "gpt-5-mini", "gpt-5.4")`. |
| **Model selector removed** | Frontend always uses GPT-5.4 for synthesis (server-side default). Model selector dropdown, localStorage persistence, and CSS removed from frontend. `streamChat()` no longer sends `model` field. |

### Production-Grade Quality Improvements — March 2026

Frontend component decomposition, UI polish, and remaining plan items.

| Item | Change |
|------|--------|
| **App.tsx decomposition** | Extracted 5 components from the 973-line monolith: `ThinkingBlock.tsx` (BurstSpinner, trace-to-steps, tool cards), `MetaBar.tsx` (trust badges, confidence, quality chips), `AnswerContent.tsx` (markdown + citation refs), `MessageActions.tsx` (CopyButton, FeedbackButtons), `CitationsPanel.tsx` (citation card list with selection). App.tsx reduced to ~310 lines of state management + layout. |
| **React.memo** | All extracted components wrapped in `React.memo()` to prevent unnecessary re-renders when SSE events arrive. `CitationCard` also memoized individually. |
| **Model selector removed** | Frontend always uses GPT-5.4 for synthesis (server-side default). Model selector dropdown removed from header. |
| **Citation kind badges** | Each citation card shows "Docs" (coral) or "Web" (blue) badge based on `source_kind`. Visual distinction between corpus and web-sourced citations. |
| **localStorage test fix** | Fixed pre-existing test failure where jsdom localStorage was broken in Node 22. Now uses `Storage.prototype` spies. |

### Multi-Turn Conversation Memory — March 2026

| Item | Change |
|------|--------|
| **LLM-based query reformulation** | `_reformulate_follow_up()` replaces `_contextualize_query()` as the primary follow-up handler. Uses OpenAI to rewrite follow-ups as standalone queries; falls back to static concat when LLM unavailable. Runs *before* classification so reformulated queries route correctly. |
| **History-aware synthesis prompt** | `_format_history_context()` formats last 2-3 Q&A pairs. Appended to synthesis prompt after context blocks with instruction to use only for continuity, not as factual source. Also passed to `_llm_knowledge_answer()`. |
| **Context-aware semantic cache** | `_cache_key()` incorporates last 2 user turns so the same follow-up with different prior context gets different cache entries. |
| **Reformulation trace event** | New `query_reformulation` SSE event emitted with `original_query`, `reformulated_query`, and `method` ("llm" or "static"). Visible in agent trace panel. |
| **GraphState.history_context** | New optional field threads history context through the LangGraph state to `_graph_generate()` → `_synthesize_answer()`. |
| **Classification on reformulated query** | `classify_assistant_mode()` now runs on the reformulated query (when reformulation triggered) instead of the raw question, fixing cases where "How much memory does it have?" was misrouted to `direct_chat`. |

### Production Hardening (P1) — March 2026

Robustness improvements for production-grade output:

| Item | Change |
|------|--------|
| **Self-RAG score parsing hardening** | `_graph_self_reflect()` uses `json.loads()` first, handles float scores via `int(float(...))`, defaults to neutral score (3) on parse failure instead of 0 (which triggered unnecessary fallbacks). |
| **SSE error event** | `stream()` wraps `self.run()` in try/except; on uncaught exception emits `error` SSE event with `message` and `recoverable` fields, followed by a `done` event with `generation_degraded=True`. Frontend gets clean error UX instead of a broken stream. |
| **Request validation** | `/api/chat/stream` validates: question ≤2000 chars, history ≤20 turns, each history message ≤5000 chars. Returns 400 with clear error messages. |
| **Weighted confidence estimation** | `estimate_confidence()` uses weighted average `0.5*top_1 + 0.3*top_2 + 0.2*top_3` instead of simple mean. Better captures "one great result" vs "all mediocre results". |
| **Early stopping in retrieval** | `run_retrieval_pass()` checks confidence after each expansion query; skips remaining queries if confidence > `EARLY_STOP_CONFIDENCE` (0.75). Saves latency for easy queries. |

### Corpus Expansion — March 2026

Full corpus expansion to close the inference/deployment gap and add infrastructure coverage.

| Item | Change |
|------|--------|
| **Markdown chunking pipeline** | Added `extract_markdown_sections()` and `chunk_markdown_document()` to `chunking.py`; `_normalize_local_source()` now scans `raw_md_root/{source_id}/*.md` for every source. Added `raw_md_root` setting to `config.py`. |
| **Source registry fixes** | Fixed 3 wrong URLs (NCCL → Developer Guide, cuda-install → CUDA Installation Guide, nsight-compute → latest version). Changed 4 hardware sources from `doc_type: "product"` to `"markdown"` (old scrapes were pure marketing garbage). |
| **Hardware datasheets** | Replaced garbage marketing page scrapes for H100, H200, A100, L40S with comprehensive technical markdown datasheets (~850 words each). |
| **Inference/deployment sources** | Added TensorRT, Triton Inference Server, NVIDIA NIM, NVIDIA AI Enterprise as markdown content. Closes the deployment gap in the corpus. |
| **Offline infrastructure sources** | Added 7 markdown reference documents: Slurm, Kubernetes, BeeGFS, Lustre, MinIO, Linux mdraid, Docker CI/CD. |
| **Field guide expansion** | Expanded 4 field guide sections from ~1 paragraph to ~800 words each: infra-platforms, infra-cluster-ops, infra-storage, infra-mlops. |
| **Deployment eval questions** | Added `data/evals/deployment_eval_questions.json` (8 inference/deployment questions). |
| **Interview demo questions** | Added `docs/interview-demo-questions.md` with curated questions per category, expected answers, and talking points. |
| **Corpus stats** | 21 → 36 normalized JSONL files, ~7,300 indexed chunks (dev mode), ~12,000 total chunks in JSONL. |

### Plan B (Opus batch) — March 2026

Major architectural improvements:

| Item | Change |
|------|--------|
| **Graph/non-graph parity** | Added `post_generation_fallback` LangGraph node. `_run_without_graph` is now a thin shim that calls the same node functions in the same order as the compiled graph — no divergent logic. |
| **LLM query rewriting** | `_graph_rewrite()` now calls the LLM to rephrase low-confidence queries before falling back to static expansion terms. Trace includes `rewrite_method`. |
| **Stronger citation enforcement** | `_ensure_citations()` only adds `[N]` markers when paragraph tokens overlap with cited chunk sparse terms (≥2 tokens). No more mechanical force-adding. |
| **Stronger grounding check** | `_grounding_check()` now also fails on LLM hedging phrases ("based on my knowledge", "I believe", etc.). |
| **Semantic caching** | `SemanticCache` class with cosine similarity, LRU eviction (128 entries), thread safety. Opt-in via `SEMANTIC_CACHE_ENABLED=true`. |
| **Self-RAG reflection** | New `self_reflect` LangGraph node after `generate`. LLM scores relevance/groundedness/completeness (1-5); forces grounding failure if `groundedness < 3`. |
| **Progressive SSE streaming** | `stream()` now uses `asyncio.Queue` + `_thread_local_emit` so trace events appear in the UI as each pipeline stage completes, not all at once. |
| **Query decomposition** | Detects multi-part questions; decomposes via LLM into 2-3 sub-questions; runs retrieval per sub-question; merges results. Opt-in via `QUERY_DECOMPOSITION_ENABLED=true`. |

### Plan A (Sonnet batch) — March 2026

Correctness, hardening, and UI improvements:

| Item | Change |
|------|--------|
| **Structured logging** | Added `logging.getLogger("maistorage.*")` to `agent.py`, `retrieval.py`, `indexes.py`. Silent `except: pass` blocks replaced with `_log.warning(...)`. |
| **Error handling** | Tavily, Pinecone, and index search calls wrapped with `try/except`; degrade gracefully instead of crashing. `generation_degraded` flag added to `AgentRunState` and SSE `done` event. |
| **Thread-safe singleton** | `get_services()` replaced `@lru_cache` with double-checked locking pattern. `reset_services()` added for test compatibility. |
| **Classification hardening** | Expanded `DOC_RAG_INFRA_TERMS` with 18 new technical terms. Added last-chance rule for `gpu`/`nvidia` queries >6 tokens. |
| **Adaptive retrieval** | `_adaptive_retrieval_params()` dynamically sets `top_k` and `confidence_floor` based on query token count and entity density. |
| **Rerank config** | `RerankConfig` dataclass extracted to `config.py`; weights are now tunable and tested independently. |
| **Follow-up contextualization** | `_contextualize_query()` threshold raised from 5 to 8 tokens; added `_FOLLOW_UP_PRONOUNS` set for reference-word detection. |
| **Test coverage** | Added `test_agent_flow.py` gap tests (retry, grounding failure, quality failure, llm-knowledge fallback), `test_classification_coverage.py` (25 parametrized cases), `test_api.py` error-handling tests, `test_chunking.py` nav-filter and PDF tests. |
| **UI: clickable citations** | Inline `[N]` markers scroll to and highlight the corresponding citation card. |
| **UI: conversation persistence** | Chat history persisted to `localStorage`; "New chat" button clears it. |
| **UI: copy to clipboard** | Copy button on every assistant message. |
| **UI: confidence display** | `MetaBar` shows confidence percentage and `generation_degraded` warning. |
| **UI: feedback buttons** | Thumbs up/down on every assistant message, persisted to `localStorage`. |
| **UI: mobile responsive** | `@media (max-width: 768px)` layout adjustments. |
| **UI: retry on error** | Error banner includes a "Retry" button that resubmits the last question. |

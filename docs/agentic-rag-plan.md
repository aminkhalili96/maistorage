# NVIDIA AI Infrastructure Agentic RAG for Q1

## Project Objective

Build an assessment-ready agentic RAG assistant for NVIDIA AI infrastructure and training optimization. The system answers from an offline snapshot of official NVIDIA documentation with explicit citations, optimized retrieval, a visible agent trace, and evaluation artifacts suitable for the MaiStorage Question 1 presentation.

## Requirement Mapping

| Q1 Requirement | Implementation |
| --- | --- |
| Build an agentic RAG that retrieves the correct chunks | Query classification, source-family routing, hybrid retrieval, reranking, document grading, rewrite loop, fallback gating |
| Demo a working prototype | React + FastAPI app with streaming answers, citations, trace, corpus status, and benchmark view |
| Discuss thought process and implementation flow | This plan, `docs/corpus-refresh.md`, `docs/testing-strategy.md`, and the visible trace panel |
| Investigate agentic RAG as a whole | Traditional vs agentic RAG comparison, corpus strategy, evaluation, and refresh design |
| Explain test cases to assure quality | Unit, integration, retrieval, RAGAS, citation, workflow, regression, and refresh validation |
| Optional: citations | Every factual paragraph must have citation-backed evidence |
| Optional: optimized retrieval | Dynamic top-k, metadata routing, reranking, document grading, rewrite loop, Pinecone hybrid search |

## Assessment Mode vs Dev Mode

- `APP_MODE=dev`
  - allows the deterministic in-memory fallback path
  - allows demo corpus bootstrap if the offline bundle is absent
  - intended for quick local iteration
- `APP_MODE=assessment`
  - defaults to `gemini-2.5-flash` for app generation
  - uses `gemini-3.1-pro-preview` only as the preferred live RAGAS judge when quota allows
  - requires `gemini-embedding-001`
  - requires `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY`
  - requires Pinecone configuration
  - requires the bundled local corpus and manifest
  - fails fast on missing assessment-critical configuration

## Architecture and Component Flow

### Frontend

- React single-page app
- Assistant tab
  - chat panel
  - one-click demo prompts
  - visible response mode and retry count
  - citations panel
  - trace panel
- Corpus & Evaluation tab
  - source-family coverage
  - snapshot metadata
  - ingestion status
  - retrieval benchmark results

### Backend

- FastAPI API
- Service container in `backend/app/runtime.py`
- Local corpus bundle under `data/corpus`
- Ingestion service that prefers bundled raw HTML/PDF snapshots and normalized JSONL chunks
- Search index abstraction
  - Pinecone hybrid index for assessment mode
  - in-memory hybrid index for dev fallback
- LangGraph-based agent workflow

## Corpus Strategy

### Offline-first bundle

The project now ships with:

- raw HTML snapshots in `data/corpus/raw/html`
- raw PDFs in `data/corpus/raw/pdfs`
- normalized JSONL chunks in `data/corpus/normalized`
- snapshot metadata in `data/corpus/manifest.json`

This keeps the live demo stable and reproducible even if NVIDIA changes its docs later.

For live assessment-mode validation in this workspace:

- the full offline NVIDIA bundle remained under `data/corpus`
- the live Pinecone population used the bundled demo corpus subset to keep Gemini re-embedding cost and latency reasonable
- the Pinecone namespace was created automatically on first upsert

### Source families

- Core
  - CUDA Programming Guide
  - CUDA Best Practices
  - Deep Learning Performance
  - Nsight Compute
- Distributed
  - NCCL
  - Fabric Manager
- Infrastructure
  - CUDA install
  - Container Toolkit
  - GPU Operator
  - GPUDirect Storage
  - DCGM
  - DGX BasePOD
- Advanced
  - cuDNN
  - cuBLAS
  - Megatron Core
  - NeMo Performance
  - Transformer Engine
- Hardware
  - H100
  - H200
  - A100
  - L40S

## Retrieval and Agent Workflow

### Query classes

- training optimization
- distributed and multi-GPU troubleshooting
- deployment and runtime setup
- hardware and topology recommendation
- general

### Agent graph

1. classify (+ optional query decomposition into sub-questions)
2. retrieve (per sub-question if decomposed, then merge)
3. document grading
4. rewrite if needed (LLM-powered, falls back to static expansion)
5. fallback if needed (Tavily)
6. generate
7. self_reflect (Self-RAG: LLM scores relevance/groundedness/completeness 1-5)
8. grounding check (citation markers + LLM hedging phrase detection)
9. answer quality check
10. post_generation_fallback (Tavily, if quality failed and Tavily not yet tried)
11. retry or finish

### Retrieval policy

- primary search: bundled corpus via Pinecone hybrid retrieval
- metadata routing by source family
- reranking by lexical overlap, family fit, and metadata hints
- document grading before synthesis
- rewrite loop when confidence is below threshold
- Tavily fallback only on corpus insufficiency or recency-sensitive questions

### Citation contract

- every factual paragraph must include inline citations
- citations expose title, section path, URL, snippet, and source kind
- corpus-backed and web-backed citations are visually distinct
- unsupported answers are retried or downgraded to insufficient-evidence

## Traditional RAG vs Agentic RAG

| Traditional RAG | Agentic RAG in this project |
| --- | --- |
| single retrieval pass | multi-step graph with retries |
| fixed top-k | adaptive top-k and confidence floor by query complexity |
| no document vetting | document grading before synthesis |
| no query rewrite | LLM-powered rewrite on low-confidence retrieval |
| no explicit fallback policy | Tavily only on insufficient corpus evidence (proper LangGraph node) |
| no self-evaluation | Self-RAG reflection node scores answer quality before grounding check |
| opaque answer generation | visible trace for classification, retrieval, grading, rewrite, self-reflect, grounding, answer quality |
| all events at end | progressive SSE streaming — trace events appear as each stage completes |
| single question only | optional query decomposition for multi-part questions |
| no caching | optional semantic cache for repeated queries |

## Trust Model

The system implements a 4-layer answer mode hierarchy:

- `corpus-backed`
  - answer stayed within the offline NVIDIA corpus
  - primary mode for the assessment demo
- `web-backed`
  - answer needed controlled Tavily fallback because the question was time-sensitive or out of corpus
  - should be labeled clearly and treated differently from corpus answers
- `llm-knowledge`
  - corpus and Tavily both exhausted, or question is a general AI/ML concept not specific to NVIDIA
  - Gemini answers from its own general knowledge
  - labeled with a purple badge in the UI to indicate the answer is not corpus- or web-grounded
- `insufficient-evidence`
  - the system refused to guess because it could not find grounded evidence after all layers were tried
  - last resort; avoids hallucination

**4-layer fallback architecture:**

1. Corpus retrieval → grade → rewrite (up to 1x, LLM-powered) → generate → self-reflect → grounding/quality check
2. Post-generation Tavily: if quality fails and Tavily has not yet been tried → retry with web results → `web-backed` (implemented as `post_generation_fallback` LangGraph node)
3. LLM general knowledge: if all retrieval layers are exhausted → Gemini answers from its own knowledge → `llm-knowledge`
4. Refusal: if the answer cannot be grounded at any layer → `insufficient-evidence`

**Multi-turn query contextualization:** short follow-up questions (≤8 tokens) or questions starting with reference pronouns ("it", "this", "that", "why", "how", etc.) are automatically expanded by prepending the prior user turn to the retrieval query, so the corpus search has enough context to find relevant chunks even when the follow-up is implicit (e.g., "What about latency?" after a question about NVLink).

Trust signals exposed in the UI:

- response mode
- query class and routed source families
- citation count
- rejected chunk count
- grounding check result
- answer-quality check result

## Design Decisions and Tradeoffs

| Decision | Alternatives | Why This Choice |
| --- | --- | --- |
| Offline corpus snapshot | live crawl only | reproducible and safe for an interview demo |
| Pinecone in assessment mode | in-memory only | closer to enterprise deployment and scalable indexing |
| Agentic retrieval | one-shot RAG | better observability, control, and failure handling |
| `gemini-2.5-flash` + `gemini-embedding-001` | heavier Gemini models or local embeddings | better quota resilience for a live demo while preserving the same retrieval embedding path |
| Controlled Tavily fallback | corpus-only or always-on web search | keeps the main story corpus-grounded while still handling recency |

## MaiStorage Business Framing

The assistant should be explained as a tool for:

- solution architects sizing and validating NVIDIA-based AI infrastructure
- deployment teams troubleshooting drivers, CUDA, containers, and Kubernetes
- AI engineers understanding multi-GPU scaling, throughput, and precision tradeoffs
- support or enablement teams searching large technical documentation sets with citations

This is a better fit for MaiStorage than presenting the system as a generic document chatbot.

## Freshness and Versioning

- Local corpus snapshot is versioned by `snapshot_id`
- Each source stores:
  - canonical URL
  - retrieval timestamp
  - local HTML/PDF files
  - content hash
  - optional PDF hash
- UI shows snapshot and refresh timestamps
- Refresh and promotion strategy is documented in `docs/corpus-refresh.md`

## Testing Strategy Overview

The full QA taxonomy lives in `docs/testing-strategy.md`. The short version:

- unit tests for chunking, routing, citation formatting, config validation
- integration tests for normalization, indexing, SSE, ingestion, and API behavior
- retrieval metrics: `hit@k`, `MRR`, `nDCG`, `routing@3`
- RAGAS: faithfulness, answer relevancy, context precision, context recall
- citation and grounding checks
- agent workflow tests for grading, rewrite gating, retry limits, and fallback behavior
- refresh/regression checks after corpus updates
- end-to-end demo checks for 3 presentation queries

## Current Live Validation Status

Verified in assessment mode with live credentials:

- Gemini embedding path using `gemini-embedding-001` with 3072-dimensional vectors
- Pinecone connectivity and namespace population for the demo corpus subset
- Tavily fallback reachability
- 3 primary demo queries returning `corpus-backed` cited answers
- 1 recency-sensitive query returning `web-backed`
- live RAGAS execution with authored reference answers
- slide-ready evaluation artifacts under:
  - [20260315-181658](/Users/amin/dev/maistorage/data/evals/results/20260315-181658)
  - [20260316-030114](/Users/amin/dev/maistorage/data/evals/results/20260316-030114)

Most useful measured results from that artifact bundle:

- retrieval benchmark on the capped benchmark subset:
  - `hit@5 = 1.0`
  - `MRR = 1.0`
  - `nDCG@5 = 0.8757`
  - `routing@3 = 1.0`
- pipeline ablation:
  - retrieval-only keyword-hit rate: `0.75`
  - retrieval + grading keyword-hit rate: `1.0`
  - full agentic flow was the only mode that produced the intended `web-backed` recency behavior
- latency:
  - average end-to-end latency: `14838.51 ms`
  - average retrieval latency: `12.57 ms`
  - average generation latency: `11273.43 ms`
- RAGAS:
  - `faithfulness = 0.9103`
  - `answer_relevancy = 0.3804`
  - `context_precision = 0.8667`
  - `context_recall = 0.8`
  - the refreshed run uses authored reference answers
  - the latest successful run used `gemini-2.5-flash` for the eval path after Gemini 3.1 Pro hit its daily quota during regeneration
  - a later direct rerun on `gemini-3.1-pro-preview` failed with `429 ResourceExhausted`, so the saved successful RAGAS evidence is still the 2.5 Flash run

Not yet live-verified in this workspace:

- Docker end-to-end startup
- LangSmith trace upload, because the current LangSmith token returned `401 Invalid token`

## Release Gates

Use these as the interview-quality bar:

- retrieval should keep expected evidence in the top results before answer metrics are trusted
- every factual paragraph should be citation-backed
- grounding and answer-quality checks should pass on the rehearsed demo questions
- refreshed snapshots should not be promoted if retrieval metrics regress
- web fallback should remain the exception, not the primary answer path

## Slide Wording

Use this wording in the presentation instead of the older router language:

- `Router classifies training, distributed, deployment, and hardware queries correctly`
- `Fallback policy sends only low-evidence or recency-sensitive queries to web search`

For embedding experiments, present this as methodology unless you have executed the harness and collected results:

> In practice, I would compare multiple embedding models on this NVIDIA corpus and choose the one with the best retrieval and RAGAS context precision on my own data.

Updated guidance now that the harness has actually been run:

- say **3 embedding models** only because the current harness now compares:
  - `gemini-embedding-001 @ 3072`
  - `multilingual-e5-large @ 1024`
  - `llama-text-embed-v2 @ 1024`
- mention the separate Gemini dimension ablation:
  - `gemini-embedding-001 @ 3072`
  - `gemini-embedding-001 @ 1536`
- the current capped benchmark subset still did not separate the models or the two Gemini dimensions, which is a lesson in benchmark difficulty rather than proof that all embeddings are equivalent

## Demo and Presentation Flow

1. Start on the corpus tab and show the bundled snapshot, source families, and indexed chunk count.
2. Ask: `Why is 4-GPU training scaling poorly?`
3. Walk through classification, retrieval, document grading, and rewrite events.
4. Show the grounded answer and citations from distributed/core docs.
5. Ask: `When should I use mixed precision training?`
6. Show the answer drawing from core and advanced sources.
7. Ask: `What NVIDIA stack is needed to deploy training workloads on Linux or Kubernetes?`
8. Show deployment/runtime routing and infrastructure citations.
9. Ask a controlled recency or out-of-corpus question and show either web-backed fallback or insufficient-evidence behavior.
10. Finish on the evaluation view and explain retrieval metrics, RAGAS, citation checks, and refresh gates.

The full interview script and wording live in [interview-demo-playbook.md](/Users/amin/dev/maistorage/docs/interview-demo-playbook.md).

## Definition of Done

- Clean code structure and consistent data flow.
- Assessment mode enforces:
  - `gemini-3.1-pro-preview`
  - `gemini-embedding-001`
  - Pinecone
  - local offline corpus bundle
- Bundled 21-source corpus snapshot exists in `data/corpus`.
- Normalized chunks exist and can bootstrap the app locally.
- LangGraph trace is visible in the UI.
- Citations render correctly.
- Three demo queries work reliably.
- One recency-sensitive query demonstrates `web-backed` behavior.
- Retrieval metrics run on the benchmark set.
- RAGAS endpoint is available when Gemini credentials are configured.
- Docker artifacts exist for a follow-up deployment pass.

## Risks and Limitations

- The current offline bundle is a snapshot, not a live mirror of NVIDIA docs.
- Gemini 3.1 Pro is still a preview model, so account availability may vary.
- Pinecone setup is still required for true assessment mode.
- RAGAS still depends on runtime credentials and optional packages.
- Docker is included as a hardening step, but the main focus remains the assessment workflow.

## Completed Improvements (Post-Initial Build)

### Plan A — Correctness & Hardening
- Structured logging across `agent.py`, `retrieval.py`, `indexes.py`
- Robust error handling: Tavily, Pinecone, index search all degrade gracefully
- Thread-safe `get_services()` singleton (double-checked locking)
- Expanded classification terms and adaptive retrieval parameters
- `RerankConfig` dataclass for tunable reranking weights
- Follow-up contextualization threshold raised to 8 tokens + pronoun detection
- Comprehensive test coverage: agent flow gaps, classification coverage, API error handling, chunking

### Plan A — UI Improvements
- Clickable inline `[N]` citation references scroll to citation card
- Conversation persistence via `localStorage` with "New chat" button
- Copy-to-clipboard on every assistant message
- Confidence percentage and `generation_degraded` warning in MetaBar
- Thumbs up/down feedback buttons with `localStorage` persistence
- Mobile-responsive layout
- Error recovery "Retry" button

### Plan B — Architecture Upgrades
- `post_generation_fallback` LangGraph node (graph/non-graph parity)
- LLM-powered query rewriting with static expansion fallback
- Token-overlap citation enforcement in `_ensure_citations()`
- LLM hedging phrase detection in `_grounding_check()`
- `SemanticCache` with cosine similarity, LRU eviction, thread safety
- `self_reflect` LangGraph node (Self-RAG scoring)
- Progressive SSE streaming via `asyncio.Queue`
- Query decomposition for multi-part questions

## Roadmap

- expand the golden benchmark set and keep refresh evaluation gated
- promote only candidate indices that preserve retrieval quality
- add live observability with LangSmith once runtime credentials are configured
- parallelize sub-question retrieval in query decomposition
- add streaming generation (token-by-token) when Gemini streaming API is available

## Production Roadmap

| Layer | What it would look like |
|-------|------------------------|
| Corpus refresh | Scheduled crawler checks source hashes weekly; only re-chunks changed docs; diff-based upsert to Pinecone |
| Index promotion | New candidate index runs full retrieval benchmark + RAGAS before replacing prod namespace; rollback if metrics regress |
| Observability | LangSmith traces all agent runs; dashboards for latency, fallback rate, grounding pass rate per query class |
| Multi-tenancy | One Pinecone namespace per customer corpus; embedder and index config per-namespace; same agent graph |
| Latency | Cache repeated retrieval plans (question hash → plan); stream generation starts before validation |
| Confidence calibration | Replace binary pass/fail grounding with a 0–1 score; surface to UI as a confidence meter |

**Multi-turn contextualization:** the system already handles short follow-up questions (≤8 tokens) or questions starting with reference pronouns ("it", "this", "that", "why", "how", etc.) by prepending the prior user turn to the retrieval query. In production this would be extended to maintain a sliding context window of the conversation to support richer multi-turn sessions.

# NVIDIA AI Infrastructure Agentic RAG

Assessment-ready React + FastAPI prototype for MaiStorage Question 1.

An agentic RAG assistant for NVIDIA AI infrastructure and training optimization. It uses an offline corpus snapshot, explicit citations, a visible agent trace, retrieval benchmarks, and optional RAGAS evaluation.

## What is included

- React frontend with:
  - chat with conversation persistence and copy-to-clipboard
  - clickable inline citation references
  - agent trace panel with progressive streaming
  - confidence display and generation-degraded warning
  - feedback buttons (thumbs up/down)
  - corpus status and retrieval benchmark view
  - mobile-responsive layout
  - error recovery retry button
- FastAPI backend with:
  - LangGraph workflow (9 nodes including Self-RAG reflection)
  - Progressive SSE streaming via `asyncio.Queue`
  - LLM-powered query rewriting (OpenAI, falls back to static expansion)
  - Semantic caching layer (opt-in, cosine similarity, LRU)
  - Query decomposition for multi-part questions (opt-in)
  - Adaptive retrieval (dynamic `top_k` and `confidence_floor`)
  - OpenAI generation path (4-tier: GPT-5.4 Nano for routing, GPT-5 Mini for pipeline, GPT-5.4 for synthesis)
  - OpenAI embedding path
  - Pinecone hybrid index path
  - Tavily fallback gating (now a proper LangGraph node)
  - Offline corpus normalization and ingestion
  - Structured logging and robust error handling
- Bundled NVIDIA corpus snapshot under `data/corpus`
- Docs:
  - [agentic-rag-plan.md](docs/agentic-rag-plan.md)
  - [evaluation-evidence.md](docs/evaluation-evidence.md)
  - [interview-demo-playbook.md](docs/interview-demo-playbook.md)
  - [testing-strategy.md](docs/testing-strategy.md)
  - [evaluation-matrix.md](docs/evaluation-matrix.md)

## Architecture

```
Browser (React + SSE)
  │
  POST /api/chat/stream ──► FastAPI ──► AgentService.stream()
  │                                       │
  ◄── SSE events (progressive) ───────────┘
                                          │
                             ┌────────────┴────────────┐
                             │   LangGraph Pipeline     │
                             │  classify → retrieve     │
                             │  → grade → [rewrite]     │
                             │  → generate (OpenAI)     │
                             │  → self_reflect          │
                             │  → grounding_check       │
                             │  → quality_check         │
                             │  → [fallback: Tavily]    │
                             └────────────┬────────────┘
                                          │
                             ┌────────────┴────────────┐
                             │  InMemory / Pinecone     │
                             │  36 sources, ~7300 chunks│
                             └─────────────────────────┘
```

## Workspace layout

- `frontend/` — React client
- `backend/` — FastAPI app, agent flow, retrieval, ingestion, evaluation
- `data/corpus/` — offline raw and normalized NVIDIA corpus bundle
- `data/evals/` — golden benchmark questions and evaluation results
- `data/sources/` — source registry
- `docs/` — architecture, testing, and demo playbook docs
- `scripts/` — corpus download, normalization, and evaluation helpers

## Local setup

### Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

- Frontend: [http://127.0.0.1:5173](http://127.0.0.1:5173)
- Backend: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Environment

Copy `.env.example` to `.env`.

**Dev mode** (default — no external credentials needed):
- `APP_MODE=dev`
- Uses in-memory keyword index and keyword embedder
- No Pinecone or OpenAI required

**Assessment mode** (full production-style stack):
- `APP_MODE=assessment`
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-5.4`
- `OPENAI_PIPELINE_MODEL=gpt-5-mini`
- `OPENAI_ROUTING_MODEL=gpt-5.4-nano`
- `OPENAI_EMBEDDING_MODEL=text-embedding-3-large`
- `OPENAI_EMBEDDING_DIMENSIONS=3072`
- `USE_PINECONE=true`
- `PINECONE_API_KEY=...`
- `PINECONE_INDEX_NAME=...`
- `EMBEDDER_PROVIDER=openai`

**Optional feature flags:**

| Variable | Default | Effect |
|----------|---------|--------|
| `SEMANTIC_CACHE_ENABLED` | `false` | In-memory semantic query cache (cosine similarity, LRU 128) |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Similarity threshold for cache hits |
| `QUERY_DECOMPOSITION_ENABLED` | `false` | LLM-based decomposition of multi-part questions |
| `USE_TAVILY_FALLBACK` | `false` | Enable Tavily web search fallback |

## Agent pipeline

```
classify
  → retrieve (per sub-question if decomposition enabled, then merge)
  → document_grading
  → rewrite_if_needed (LLM-powered, falls back to static expansion)
  → fallback_if_needed (Tavily)
  → generate
  → self_reflect (Self-RAG: scores relevance/groundedness/completeness 1-5)
  → grounding_check (citation markers + hedging phrase detection)
  → answer_quality_check
  → post_generation_fallback (Tavily, if quality failed and not yet tried)
  → [second generate → self_reflect → grounding_check → answer_quality_check]
```

**Trust model (5 modes):**
1. `corpus-backed` — grounded in offline NVIDIA corpus
2. `web-backed` — Tavily fallback used
3. `llm-knowledge` — OpenAI general knowledge (all retrieval exhausted)
4. `insufficient-evidence` — refusal (last resort, avoids hallucination)
5. `direct-chat` — conversational turn, no retrieval needed

## Interview framing

This repo demos a **small production-style AI system**, not a generic RAG chatbot.

Key talking points:
- Versioned offline corpus snapshot for reliability
- Explicit agent trace for observability
- Visible trust model with 5 response modes
- Multi-turn conversation memory with LLM-based query reformulation
- Citations with token-overlap grounding enforcement
- Self-RAG reflection loop for answer quality
- Retrieval benchmarks before answer-level claims
- Progressive SSE streaming — trace events appear as each stage completes
- Semantic caching for repeated queries
- Query decomposition for complex multi-part questions

Use the playbook in [interview-demo-playbook.md](docs/interview-demo-playbook.md) for the live demo order, MaiStorage-specific business framing, and design tradeoffs.

**Recommended demo questions:**
- `Why is 4-GPU training scaling poorly?`
- `When should I use mixed precision training and what are the tradeoffs?`
- `What NVIDIA stack is needed to deploy training workloads on Linux or Kubernetes?`
- Controlled fallback: `What changed in the latest NVIDIA Container Toolkit release?`

## Corpus commands

Rebuild normalized chunks from bundled raw corpus:
```bash
PYTHONPATH=backend backend/.venv/bin/python scripts/normalize_corpus.py
```

Populate Pinecone with the demo set:
```bash
PYTHONPATH=backend backend/.venv/bin/python scripts/bootstrap_assessment_index.py --demo-corpus
```

Run the full evaluation artifact suite:
```bash
LANGSMITH_TRACING=false backend/.venv/bin/python scripts/run_evaluation_suite.py --skip-verification
```

## Docker

```bash
docker compose up --build
```

## Verification

Backend tests (from repo root):
```bash
PYTHONPATH=backend backend/.venv/bin/pytest backend/tests -q
```

Smoke check:
```bash
PYTHONPATH=backend backend/.venv/bin/pytest backend/tests/test_simple_chat_flow.py backend/tests/test_chatbot_smoke.py -v
```

Frontend tests:
```bash
cd frontend && npm test
```

Frontend build:
```bash
cd frontend && npm run build
```

## Test status

- **212 backend tests pass** (265 total; ~50 pre-existing failures in benchmark/eval suites that require live API credentials)
- **19 frontend tests pass**

## Benchmark numbers (last eval run)

| Metric | Value |
|--------|-------|
| hit@5 | 1.0 |
| MRR | 1.0 |
| nDCG@5 | 0.88 |
| routing@3 | 1.0 |
| RAGAS faithfulness | 0.694 |
| RAGAS answer_relevancy | 0.735 |
| RAGAS context_precision | 0.683 |
| RAGAS context_recall | 0.675 |

RAGAS scores from 10-question curated slim set (OpenAI GPT-5.4 generation, gpt-4.1 judge, March 2026). Earlier 46-question run with Gemini scored higher on faithfulness (0.91) but used a different model and judge.

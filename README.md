# NVIDIA AI Infrastructure Agentic RAG

Assessment-ready React + FastAPI prototype for MaiStorage Question 1.

The project is an agentic RAG assistant for NVIDIA AI infrastructure and training optimization. It uses an offline corpus snapshot, explicit citations, a visible agent trace, retrieval benchmarks, and optional RAGAS evaluation.

## What is included

- React frontend with:
  - chat
  - citations
  - agent trace
  - corpus status
  - retrieval benchmark view
- FastAPI backend with:
  - LangGraph workflow
  - Gemini generation path
  - Google embedding path
  - Pinecone hybrid index path
  - Tavily fallback gating
  - offline corpus normalization and ingestion
- Bundled NVIDIA corpus snapshot under `data/corpus`
- Docs:
  - [agentic-rag-plan.md](/Users/amin/dev/maistorage/docs/agentic-rag-plan.md)
  - [corpus-refresh.md](/Users/amin/dev/maistorage/docs/corpus-refresh.md)
  - [evaluation-evidence.md](/Users/amin/dev/maistorage/docs/evaluation-evidence.md)
  - [interview-demo-playbook.md](/Users/amin/dev/maistorage/docs/interview-demo-playbook.md)
  - [testing-strategy.md](/Users/amin/dev/maistorage/docs/testing-strategy.md)
  - [evaluation-matrix.md](/Users/amin/dev/maistorage/docs/evaluation-matrix.md)

## Workspace layout

- `frontend/`: React client
- `backend/`: FastAPI app, agent flow, retrieval, ingestion, evaluation
- `data/corpus/`: offline raw and normalized NVIDIA corpus bundle
- `data/evals/`: golden benchmark questions
- `data/sources/`: source registry
- `docs/`: architecture, refresh, and testing docs
- `scripts/`: corpus download and normalization helpers

## Local setup

### Backend

```bash
cd /Users/amin/dev/maistorage/backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd /Users/amin/dev/maistorage/frontend
npm install
npm run dev
```

Frontend default:

- [http://127.0.0.1:5173](http://127.0.0.1:5173)

Backend default:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Environment

Copy `.env.example` to `.env`.

Assessment mode expects:

- `APP_MODE=assessment`
- `GEMINI_MODEL=gemini-3.1-pro-preview`
- `GEMINI_EMBEDDING_MODEL=gemini-embedding-001`
- `GEMINI_DOCUMENT_TASK_TYPE=RETRIEVAL_DOCUMENT`
- `GEMINI_QUERY_TASK_TYPE=RETRIEVAL_QUERY`
- `USE_PINECONE=true`
- valid Gemini and Pinecone credentials

Dev mode can still use the local in-memory fallback path.

## Interview framing

This repo is meant to demo a **small production-style AI system**, not a generic RAG chatbot.

The strongest story to tell is:

- versioned offline corpus snapshot for reliability
- explicit agent trace for observability
- visible trust model: `corpus-backed`, `web-backed`, `insufficient-evidence`
- citations and grounding checks for answer trust
- retrieval benchmarks before answer-level claims
- refresh and promotion design for upstream doc changes

Use the playbook in [interview-demo-playbook.md](/Users/amin/dev/maistorage/docs/interview-demo-playbook.md) for:

- the live demo order
- MaiStorage-specific business framing
- design tradeoffs
- limitations and roadmap
- safe wording for verified vs credential-dependent capabilities

Recommended demo questions:

- `Why is 4-GPU training scaling poorly?`
- `When should I use mixed precision training and what are the tradeoffs?`
- `What NVIDIA stack is needed to deploy training workloads on Linux or Kubernetes?`
- controlled failure/fallback: `What changed in the latest NVIDIA Container Toolkit release?`

## Corpus commands

Download or refresh the bundled NVIDIA snapshot:

```bash
cd /Users/amin/dev/maistorage
backend/.venv/bin/python scripts/download_corpus.py
```

Rebuild normalized chunks from the bundled raw corpus:

```bash
cd /Users/amin/dev/maistorage
PYTHONPATH=backend backend/.venv/bin/python scripts/normalize_corpus.py
```

Compare multiple embedding configurations on the same corpus:

```bash
cd /Users/amin/dev/maistorage
PYTHONPATH=backend backend/.venv/bin/python scripts/compare_embeddings.py --configs data/evals/embedding_experiments.example.json
```

Run the Gemini dimension ablation:

```bash
cd /Users/amin/dev/maistorage
PYTHONPATH=backend backend/.venv/bin/python scripts/compare_embeddings.py --configs data/evals/embedding_dimensions.example.json
```

Run the full evaluation evidence suite and write slide-ready artifacts:

```bash
cd /Users/amin/dev/maistorage
PYTHONPATH=backend LANGSMITH_TRACING=false backend/.venv/bin/python scripts/run_evaluation_suite.py --skip-verification
```

Populate Pinecone with the interview-critical live demo set:

```bash
cd /Users/amin/dev/maistorage
PYTHONPATH=backend backend/.venv/bin/python scripts/bootstrap_assessment_index.py --demo-corpus
```

## Docker

Build and run:

```bash
cd /Users/amin/dev/maistorage
docker compose up --build
```

## Verification

Backend tests:

```bash
cd /Users/amin/dev/maistorage
PYTHONPATH=backend backend/.venv/bin/pytest backend/tests
```

Frontend build:

```bash
cd /Users/amin/dev/maistorage/frontend
npm run build
```

## Verification status

Verified locally in this workspace:

- offline corpus bundle and normalized chunks
- retrieval benchmark artifact bundle at [20260315-181658](/Users/amin/dev/maistorage/data/evals/results/20260315-181658)
- supplemental model/RAGAS artifact bundle at [20260316-030114](/Users/amin/dev/maistorage/data/evals/results/20260316-030114)
- backend test suite
- frontend production build
- trace, citations, and trust-signal UI paths
- Gemini live embedding path with `gemini-embedding-001` returning 3072 dimensions
- Pinecone assessment-mode connectivity and live namespace population for the demo corpus
- Tavily live fallback reachability
- live RAGAS evaluation with authored reference answers
- slide-ready artifacts for:
  - retrieval benchmark
  - RAGAS
  - 3-model embedding comparison
  - Gemini dimension ablation
  - chunking ablation
  - pipeline ablation
  - hybrid vs dense-only comparison
  - latency summary
  - demo query validation
- assessment-mode API checks for:
  - `Why is 4-GPU training scaling poorly?`
  - `When should I use mixed precision training and what are the tradeoffs?`
  - `What NVIDIA stack is needed to deploy training workloads on Linux or Kubernetes?`
  - `What changed in the latest NVIDIA Container Toolkit release?`

Implemented but still dependent on live credentials or environment:

- Docker end-to-end startup

Implemented but not live-verified in this workspace because the current key was rejected:

- LangSmith trace upload (`401 Invalid token` from the LangSmith API)

Important note:

- the live Pinecone validation in this workspace used the bundled `data/demo_chunks.json` subset for speed and cost control
- the full bundled NVIDIA corpus still exists under `data/corpus`
- if you want to expand Pinecone beyond the demo set, use `scripts/bootstrap_assessment_index.py` with explicit `--source-id` values
- the slide-ready benchmark artifacts used a capped benchmark subset:
  - 9 source ids covering all golden questions
  - max 12 chunks per source id
- the refreshed RAGAS bundle at [20260316-030114](/Users/amin/dev/maistorage/data/evals/results/20260316-030114) used `gemini-2.5-flash` for the RAGAS answer/judge path because the workspace hit the daily Gemini 3.1 Pro quota during regeneration; the main app still targets `gemini-3.1-pro-preview`

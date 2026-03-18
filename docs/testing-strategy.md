# Testing Strategy

## Goal

The assessment requires more than a passing unit test suite. This project validates retrieval quality, grounding, agent behavior, freshness, and live-demo readiness.

## Tiered Matrix

### Tier 1: Must have

- ingestion produces the expected chunk count for a fixture corpus
- chunks respect size and overlap rules
- chunk metadata is preserved
- router classifies training, distributed, deployment, and hardware queries correctly
- fallback gating keeps corpus-backed questions local
- document grading keeps relevant docs and rejects irrelevant docs
- generator returns non-empty answers
- citations contain full source information
- end-to-end known questions return expected domain keywords
- LangGraph workflow compiles and runs

### Tier 2: Should have

- grounding check passes for grounded answers
- answer quality check passes for relevant answers
- mixed batches are filtered correctly
- full pipeline returns all required state fields
- SSE stream returns the expected event order
- ingestion status exposes snapshot and refresh metadata

### Tier 3: Nice to have

- RAGAS on 3-5 golden pairs with target thresholds
- fallback path triggers on true corpus insufficiency
- embedding experiment runner compares multiple embedding configurations on this corpus

## Test Layers

### 1. Unit tests

Validate isolated logic:

- chunking and section extraction
- stable chunk ID generation
- metadata preservation
- query classification and routing
- citation formatting
- fallback gating
- assessment-mode config validation
- embedding task-type assignment

### 2. Integration tests

Validate component boundaries:

- local corpus normalization into JSONL
- local or Pinecone indexing path
- chat SSE event stream
- ingestion status lifecycle
- retrieval evaluation endpoint

### 3. Retrieval quality tests

Run on the golden NVIDIA question set:

- `hit@k`
- `MRR`
- `nDCG`
- `routing@3`
- expected-source match

These metrics come before answer-level evaluation because weak retrieval makes answer metrics misleading.

### 4. RAGAS

Use RAGAS only after retrieval quality is acceptable:

- faithfulness
- answer relevancy
- context precision
- context recall

Current project guidance:

- the repo now uses a unified 50-question benchmark at [agentic_golden_questions.json](/Users/amin/dev/maistorage/data/evals/agentic_golden_questions.json)
- the benchmark deliberately mixes:
  - 46 corpus-backed doc/RAG questions
  - 2 direct-chat routing questions
  - 1 refusal case
  - 1 recency-sensitive fallback case
- RAGAS should run on the corpus-backed authored-reference subset, not the whole 50-question file
- treat these thresholds as the release-style bar for the interview benchmark:
  - `faithfulness >= 0.80`
  - `answer_relevancy >= 0.55`
  - `context_precision >= 0.75`
  - `context_recall >= 0.70`
- if the live Gemini judge is unavailable, still use:
  - retrieval benchmark
  - trajectory benchmark
  - grounding checks
  - answer-quality checks
  - the deterministic 50-question benchmark and its corpus-backed 46-question retrieval subset

### 5. Citation and grounding tests

- every factual paragraph has at least one citation
- cited chunk actually supports the claim
- citation URLs and section paths are valid
- corpus and web citations are labeled separately

### 6. Agent workflow tests

- document grading rejects weak chunks
- rewrite happens only when confidence is low (LLM rewrite attempted first, static expansion fallback)
- retry loop stops at the configured limit
- Tavily fallback triggers only on insufficiency or recency-sensitive questions
- `post_generation_fallback` node triggers when quality fails and Tavily not yet tried
- Self-RAG reflection forces grounding failure when `groundedness < 3`
- low-evidence cases produce refusal or downgraded confidence
- `llm-knowledge` fallback fires when corpus + Tavily both exhausted and reasoner is enabled
- semantic cache returns cached state on similar queries (when enabled)
- query decomposition splits multi-part questions and merges retrieval results (when enabled)

### 7. Trajectory tests

Evaluate the agent path, not just the final answer:

- assistant mode accuracy: `direct_chat` vs `doc_rag`
- response mode accuracy: `direct-chat`, `corpus-backed`, `web-backed`, `insufficient-evidence`
- expected tool path accuracy
- retry budget enforcement
- fallback usage correctness
- accepted and rejected evidence propagation
- grounding and answer-quality validation outcomes

This repo now emits a dedicated trajectory artifact with per-question rows and aggregate pass rates.

### 8. Refresh and regression tests

- content hash changes trigger source refresh
- normalized chunks stay stable when content does not change
- post-refresh retrieval metrics do not regress below gates
- citation rendering survives prompt and retrieval changes

### 9. Negative and adversarial tests

- out-of-corpus questions
- ambiguous questions
- conflicting-source questions
- malformed HTML/PDF inputs
- missing Gemini / Pinecone / Tavily credentials
- provider timeout or failure paths

### 10. End-to-end demo tests

Required demo set:

- `Why is 4-GPU training scaling poorly?`
- `When should I use mixed precision training?`
- `What NVIDIA stack is needed to deploy training workloads on Linux or Kubernetes?`

Expected result for each:

- useful answer
- visible trace
- visible citations
- stable latency for a live demo

## Embedding Comparison Card

Recommended wording for the slide:

> In practice, I would compare multiple embedding models on this NVIDIA corpus and choose the one that gives the best retrieval and RAGAS context precision on my own data. Public benchmarks are a starting point, but corpus-specific evaluation is what matters.

This repo includes an experiment harness in:

- `scripts/compare_embeddings.py`
- `data/evals/embedding_experiments.example.json`
- `data/evals/embedding_dimensions.example.json`
- `scripts/run_evaluation_suite.py`
- [evaluation-evidence.md](/Users/amin/dev/maistorage/docs/evaluation-evidence.md)

The suite runner can now be started directly with:

- `backend/.venv/bin/python scripts/run_evaluation_suite.py --skip-verification`

If live Gemini embeddings are unavailable or rate-limited during the suite run, the retrieval-oriented sections fall back to the local keyword baseline and continue writing artifacts instead of aborting.

Only present experiment **results** if the harness has actually been run.

Runtime defaults for this repo:

- app generation model default: `gemini-2.5-flash`
- RAGAS judge default: `gemini-3.1-pro-preview`

## Current Test Status

- **206 backend tests pass** (5 pre-existing failures: Pinecone network unavailable in sandbox, stock-price routing edge case, A100 benchmark question)
- **13 frontend tests pass**

Test files added in Plan A/B:
- `test_agent_flow.py` — gap tests: retry, grounding failure, quality failure, llm-knowledge fallback, refusal content
- `test_classification_coverage.py` — 25 parametrized classification cases + last-chance rule regression
- `test_api.py` — error handling: missing question, empty question, debug endpoint, `generation_degraded` field
- `test_chunking.py` — nav-filter, PDF chunking, split overlap correctness

## Latest Evidence Run

Latest artifact bundle:

- [20260315-181658](/Users/amin/dev/maistorage/data/evals/results/20260315-181658)
- [20260316-030114](/Users/amin/dev/maistorage/data/evals/results/20260316-030114)

What was produced:

- retrieval benchmark
- RAGAS
- 3 embedding-model comparison
- Gemini dimension ablation
- chunking ablation
- pipeline ablation
- hybrid vs dense-only comparison
- latency summary
- demo-query validation

Important interpretation:

- the benchmark bundle used a capped subset:
  - 9 source ids covering the golden questions
  - max 12 chunks per source id
- the refreshed embedding comparison now uses **3 real models**:
  - `gemini-embedding-001`
  - `multilingual-e5-large`
  - `llama-text-embed-v2`
- the Gemini dimension comparison is tracked separately in `embedding_dimension_comparison.json`
- the chunking ablation used the keyword baseline as a fast local proxy
- RAGAS now uses authored reference answers instead of keyword proxies
- RAGAS now evaluates the 46-question corpus-backed authored-reference subset of the unified 50-question benchmark
- the refreshed RAGAS profile is stronger:
  - faithfulness, context precision, and context recall are all high
  - answer relevancy remains the weakest score and should still be presented honestly
- the latest successful RAGAS rerun used `gemini-2.5-flash` for the evaluation path because Gemini 3.1 Pro hit its daily quota during regeneration
- a later direct rerun on `gemini-3.1-pro-preview` failed with `429 ResourceExhausted`, so the saved successful RAGAS scores still come from the 2.5 Flash evaluation path
- the RAGAS implementation now defaults the judge to `gemini-3.1-pro-preview`
- row generation now has a local deterministic fallback when live embedding/retrieval calls are throttled
- the remaining blocker for a fresh Gemini 3.1 Pro artifact is judge quota itself, as documented in:
  - [20260316-043200](/Users/amin/dev/maistorage/data/evals/results/20260316-043200)
- the latest trajectory + Gemini 3.1 Pro judge attempt is documented in:
  - [20260316-052110](/Users/amin/dev/maistorage/data/evals/results/20260316-052110)
- the testing stack now maps cleanly to:
  - component tests
  - trajectory benchmark
  - end-to-end retrieval/RAGAS evaluation
  - optional Gemini 3.1 Pro live judge

## Presentation Framing

Recommended line for the presentation:

> I treat testing as a stack. Unit and integration tests protect the code paths, retrieval metrics validate whether the right chunks are being found, RAGAS evaluates answer quality, and custom grounding checks ensure the system does not look correct while citing the wrong evidence.

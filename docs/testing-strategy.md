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

### 5. Citation and grounding tests

- every factual paragraph has at least one citation
- cited chunk actually supports the claim
- citation URLs and section paths are valid
- corpus and web citations are labeled separately

### 6. Agent workflow tests

- document grading rejects weak chunks
- rewrite happens only when confidence is low
- retry loop stops at the configured limit
- Tavily fallback triggers only on insufficiency or recency-sensitive questions
- low-evidence cases produce refusal or downgraded confidence

### 7. Refresh and regression tests

- content hash changes trigger source refresh
- normalized chunks stay stable when content does not change
- post-refresh retrieval metrics do not regress below gates
- citation rendering survives prompt and retrieval changes

### 8. Negative and adversarial tests

- out-of-corpus questions
- ambiguous questions
- conflicting-source questions
- malformed HTML/PDF inputs
- missing Gemini / Pinecone / Tavily credentials
- provider timeout or failure paths

### 9. End-to-end demo tests

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

Only present experiment **results** if the harness has actually been run.

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
- the refreshed RAGAS profile is stronger:
  - faithfulness, context precision, and context recall are all high
  - answer relevancy remains the weakest score and should still be presented honestly
- the latest RAGAS rerun used `gemini-2.5-flash` for the evaluation path because Gemini 3.1 Pro hit its daily quota during regeneration

## Presentation Framing

Recommended line for the presentation:

> I treat testing as a stack. Unit and integration tests protect the code paths, retrieval metrics validate whether the right chunks are being found, RAGAS evaluates answer quality, and custom grounding checks ensure the system does not look correct while citing the wrong evidence.

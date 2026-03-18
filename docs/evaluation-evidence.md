# Evaluation Evidence

Primary full-suite artifact bundle:

- [20260315-181658](/Users/amin/dev/maistorage/data/evals/results/20260315-181658)
- [slide_summary.md](/Users/amin/dev/maistorage/data/evals/results/20260315-181658/slide_summary.md)

Latest supplemental artifact bundle for model comparison and improved RAGAS:

- [20260316-030114](/Users/amin/dev/maistorage/data/evals/results/20260316-030114)

Latest Gemini 3.1 Pro RAGAS rerun attempt:

- [20260315-195514](/Users/amin/dev/maistorage/data/evals/results/20260315-195514)
- [20260316-043200](/Users/amin/dev/maistorage/data/evals/results/20260316-043200)

Latest trajectory + Gemini 3.1 Pro judge artifact bundle:

- [20260316-052110](/Users/amin/dev/maistorage/data/evals/results/20260316-052110)

## What Was Run

- backend tests: `25 passed, 1 skipped`
- frontend build: passed
- retrieval benchmark on the capped benchmark subset:
  - 9 source ids covering all golden questions
  - max 12 chunks per source id
- unified benchmark file:
  - 50 total questions
  - 46 corpus-backed doc/RAG rows for retrieval and RAGAS
  - 2 direct-chat routing rows
  - 1 refusal row
  - 1 controlled recency-fallback row
- trajectory benchmark
- optional Gemini 3.1 Pro trajectory-and-answer judge
- RAGAS with authored reference answers
- 3 real embedding-model comparison
- Gemini dimension ablation
- chunking ablation
- pipeline ablation
- hybrid vs dense-only comparison
- latency summary
- 4 demo-query validation runs

## Key Results

### Retrieval benchmark

- `hit@5 = 1.0`
- `MRR = 1.0`
- `nDCG@5 = 0.8757`
- `routing@3 = 1.0`

Artifact:

- [retrieval_benchmark.json](/Users/amin/dev/maistorage/data/evals/results/20260315-181658/retrieval_benchmark.json)

### RAGAS — Current: 10-question slim set (OpenAI, March 18 2026)

- `faithfulness = 0.6943`
- `answer_relevancy = 0.7350`
- `context_precision = 0.6833`
- `context_recall = 0.6750`

Generation model: `gpt-5.4`. Judge model: `gpt-4.1`. 10-question curated slim set covering 7 categories, all 4 response modes, and 18 sources.

Artifact: `data/evals/results/20260318-070241/ragas_slim_10.json`

### RAGAS — Historical: 46-question set (Gemini, March 15-16 2026)

- `faithfulness = 0.9103`
- `answer_relevancy = 0.3804`
- `context_precision = 0.8667`
- `context_recall = 0.8`

> **Note:** These results are from the pre-migration era when the system used Gemini models for generation and judging. They are preserved here as historical context. The project has since migrated to OpenAI (GPT-5.4 for synthesis, GPT-5 Mini for pipeline, GPT-5.4 Nano for routing).

Interpretation:

- authored reference answers materially improved the grounding signal versus the earlier proxy-based run
- the latest successful run used `gemini-2.5-flash` for the answer/judge path because the workspace hit the daily Gemini 3.1 Pro quota during regeneration
- a direct rerun on `gemini-3.1-pro-preview` was attempted later and failed with `429 ResourceExhausted` before any score rows were produced
- the RAGAS service is now configured to use the 46-question corpus-backed authored-reference subset of the unified 50-question benchmark
- the intended release-style thresholds are:
  - `faithfulness >= 0.80`
  - `answer_relevancy >= 0.55`
  - `context_precision >= 0.75`
  - `context_recall >= 0.70`

### Trajectory benchmark

The project now includes an explicit trajectory layer on top of retrieval metrics:

- assistant-mode accuracy
- response-mode accuracy
- tool-path accuracy
- retry-budget pass rate
- fallback correctness
- grounding and answer-quality validation outcomes

The corresponding artifact is written as `trajectory_benchmark.json` by [run_evaluation_suite.py](/Users/amin/dev/maistorage/scripts/run_evaluation_suite.py).

Latest local run:

- [trajectory_benchmark.json](/Users/amin/dev/maistorage/data/evals/results/20260316-052110/trajectory_benchmark.json)

### Gemini 3.1 Pro judge

The project now includes an optional live judge path using `gemini-3.1-pro-preview` to grade:

- faithfulness
- answer relevance
- trajectory correctness
- tool-path correctness
- citation support

The judge is intentionally optional:

- it only runs when `GEMINI_API_KEY` exists
- it only runs when `RUN_LIVE_JUDGE_TESTS=true`
- otherwise it emits a structured `skipped` artifact instead of failing the suite
- the latest live attempt is recorded here:
  - [agentic_judge.json](/Users/amin/dev/maistorage/data/evals/results/20260316-052110/agentic_judge.json)
  - [summary.md](/Users/amin/dev/maistorage/data/evals/results/20260316-052110/summary.md)

Artifact:

- [ragas.json](/Users/amin/dev/maistorage/data/evals/results/20260316-030114/ragas.json)
- [failed Gemini 3.1 Pro rerun](/Users/amin/dev/maistorage/data/evals/results/20260315-195514/ragas.json)
- [failed Gemini 3.1 Pro sample rerun after fallback hardening](/Users/amin/dev/maistorage/data/evals/results/20260316-043200/ragas.json)

### Embedding models

Compared:

- `gemini-embedding-001 @ 3072`
- `multilingual-e5-large @ 1024`
- `llama-text-embed-v2 @ 1024`

Observed result on the capped benchmark subset:

- no measurable retrieval-metric difference between the 3 tested models on the capped benchmark subset

Interpretation:

- you can now truthfully say "3 embedding models were compared"
- on this capped subset, the benchmark was too easy to separate them

Artifact:

- [embedding_model_comparison.json](/Users/amin/dev/maistorage/data/evals/results/20260316-030114/embedding_model_comparison.json)

### Dimension ablation

Compared:

- `gemini-embedding-001 @ 3072`
- `gemini-embedding-001 @ 1536`

Observed result on the capped benchmark subset:

- no measurable retrieval-metric difference between the two dimensions on the current benchmark slice

Interpretation:

- this is useful as a cost/latency discussion point rather than proof that dimensionality never matters
- if you want separation here, expand the benchmark set or increase source coverage

Artifact:

- [embedding_dimension_comparison.json](/Users/amin/dev/maistorage/data/evals/results/20260316-030114/embedding_dimension_comparison.json)

### Chunking ablation

Winner:

- `600 / 80` on the keyword-proxy harness

Interpretation:

- smaller chunks improved retrieval ranking on the proxy harness
- the production demo still used the selected Gemini configuration with the current default chunking
- present this as a tuning signal, not a finalized production truth

Artifact:

- [chunking_ablation.json](/Users/amin/dev/maistorage/data/evals/results/20260315-181658/chunking_ablation.json)

### Pipeline ablation

Observed result:

- retrieval-only had lower keyword-hit rate
- grading improved keyword-hit rate from `0.75` to `1.0`
- full agentic mode was the only mode that produced the intended `web-backed` path for the recency query

Artifact:

- [pipeline_ablation.json](/Users/amin/dev/maistorage/data/evals/results/20260315-181658/pipeline_ablation.json)

### Latency

- average end-to-end latency: `14838.51 ms`
- average retrieval latency: `12.57 ms`
- average generation latency: `11273.43 ms`

Interpretation:

- the bottleneck is generation, not retrieval
- this supports a senior-level roadmap around caching, lighter validation, or smaller generator models

Artifact:

- [latency_summary.json](/Users/amin/dev/maistorage/data/evals/results/20260315-181658/latency_summary.json)

### Demo query validation

Validated:

- 3 primary demo queries returned `corpus-backed`
- 1 recency-sensitive query returned `web-backed`
- all 4 returned citations, grounding pass, and answer-quality pass

Artifact:

- [demo_query_validation.json](/Users/amin/dev/maistorage/data/evals/results/20260315-181658/demo_query_validation.json)

## Slide Guidance

Good claims:

- "I evaluated retrieval quality, answer quality, workflow behavior, and latency separately."
- "The pipeline ablation showed that grading improved the answer path, and the full agentic flow was the only one that handled the recency query correctly."
- "RAGAS improved materially once I replaced keyword-proxy ground truth with authored reference answers."
- "Even after that improvement, answer relevancy is still the weakest score, so I would treat that as the next tuning target."

Avoid these claims:

- "RAGAS is strong across the board"
- "This benchmark proves Pinecone is better than local retrieval"

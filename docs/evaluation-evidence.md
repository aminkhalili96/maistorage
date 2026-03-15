# Evaluation Evidence

Primary full-suite artifact bundle:

- [20260315-181658](/Users/amin/dev/maistorage/data/evals/results/20260315-181658)
- [slide_summary.md](/Users/amin/dev/maistorage/data/evals/results/20260315-181658/slide_summary.md)

Latest supplemental artifact bundle for model comparison and improved RAGAS:

- [20260316-030114](/Users/amin/dev/maistorage/data/evals/results/20260316-030114)

## What Was Run

- backend tests: `25 passed, 1 skipped`
- frontend build: passed
- retrieval benchmark on the capped benchmark subset:
  - 9 source ids covering all golden questions
  - max 12 chunks per source id
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

### RAGAS

- `faithfulness = 0.9103`
- `answer_relevancy = 0.3804`
- `context_precision = 0.8667`
- `context_recall = 0.8`

Interpretation:

- authored reference answers materially improved the grounding signal versus the earlier proxy-based run
- answer grounding is now strong, while answer relevancy is still the weakest score
- this is a better benchmark story, but still small enough that it should be presented as an interview eval set rather than a production-quality QA corpus
- the refreshed run used `gemini-2.5-flash` for the answer/judge path because the workspace hit the daily Gemini 3.1 Pro quota during regeneration

Artifact:

- [ragas.json](/Users/amin/dev/maistorage/data/evals/results/20260316-030114/ragas.json)

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

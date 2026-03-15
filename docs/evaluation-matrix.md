# Evaluation Matrix

| Category | Metric or Check | Pass Criteria | Example Question | Why It Matters |
| --- | --- | --- | --- | --- |
| Retrieval | `hit@5` | at least one expected source in top 5 | Why is GPU utilization high but throughput low? | proves the retriever can surface relevant evidence |
| Retrieval | `MRR` | first relevant chunk ranks early | Why is 4-GPU training scaling poorly? | validates ranking quality, not just recall |
| Retrieval | `nDCG@5` | relevance-weighted rank stays high | When should I use mixed precision training? | checks multi-source ranking quality |
| Retrieval | `routing@3` | expected source family appears in the top 3 | What NVIDIA stack is needed for Linux/Kubernetes deployment? | validates metadata routing |
| RAGAS | faithfulness | above agreed threshold | When should I use mixed precision training? | checks answer grounding |
| RAGAS | answer relevancy | above agreed threshold | What NVIDIA stack is needed for deployment? | checks whether the answer addresses the question |
| RAGAS | context precision | above agreed threshold | Why is 4-GPU training scaling poorly? | penalizes noisy retrieval |
| RAGAS | context recall | above agreed threshold | What hardware should I consider for a 7B fine-tune? | validates evidence coverage |
| Citations | support check | every factual paragraph has supporting citations | Any demo answer | required for Q1 bonus point |
| Workflow | rewrite gating | rewrite only on low-confidence retrieval | Low-evidence distributed question | shows agentic control, not random retries |
| Workflow | fallback gating | Tavily only on insufficiency or recency | latest driver/runtime question | keeps the corpus-backed story clean |
| Workflow | retry limit | never exceeds configured retry count | low-evidence question | protects cost and latency |
| Refresh | hash-based change detection | only changed sources are rebuilt | any source refresh | supports freshness without full rebuild |
| Regression | benchmark stability | retrieval quality does not regress after refresh | full golden set | prevents silent quality drops |
| Non-functional | streaming order | SSE events arrive in the intended order | any live chat run | required for a polished demo |
| Non-functional | latency | answer remains interactive | demo prompts | presentation readiness |

## Recommended Interview Gates

Use these as the release-style framing in the presentation:

- `hit@5 >= 0.80` on the golden retrieval set
- `MRR >= 0.60` on the same benchmark
- 100% citation coverage for factual demo answers
- grounding and answer-quality checks pass for the rehearsed demo prompts
- refreshed snapshots are not promoted if retrieval quality regresses

## Current Verified Status

- Latest artifact bundle:
  - [20260315-181658](/Users/amin/dev/maistorage/data/evals/results/20260315-181658)
  - [20260316-030114](/Users/amin/dev/maistorage/data/evals/results/20260316-030114)
- Retrieval:
  - capped benchmark subset retrieval metrics:
    - `hit@5 = 1.0`
    - `MRR = 1.0`
    - `nDCG@5 = 0.8757`
    - `routing@3 = 1.0`
  - live assessment-mode retrieval checks also ran successfully against the Pinecone-backed demo corpus subset
- Workflow:
  - the 3 primary demo prompts returned `corpus-backed`
  - the recency-sensitive Container Toolkit prompt returned `web-backed`
  - pipeline ablation showed grading improved keyword-hit rate from `0.75` to `1.0`
- Trust:
  - the live demo prompts surfaced citations, grounding status, answer-quality status, and rejected chunk counts
  - demo-query validation recorded citations, grounding pass, and answer-quality pass on all 4 rehearsed queries
- RAGAS:
  - `faithfulness = 0.9103`
  - `answer_relevancy = 0.3804`
  - `context_precision = 0.8667`
  - `context_recall = 0.8`
  - authored reference answers materially improved grounding coverage, but answer relevancy remains the weakest metric
- Embedding models:
  - 3 real models were compared:
    - `gemini-embedding-001`
    - `multilingual-e5-large`
    - `llama-text-embed-v2`
  - the current capped benchmark subset did not separate them on retrieval metrics
- Dimensions:
  - Gemini `3072` vs `1536` was also compared separately
  - the current capped benchmark subset did not separate the two dimensions
- Observability:
  - in-app trace is verified
  - LangSmith upload is not yet verified because the current token was rejected by the API

# Supplemental Evaluation Summary

Artifact bundle: `/Users/amin/dev/maistorage/data/evals/results/20260316-030114`

## What changed
- Added 2 real embedding models to the comparison harness:
  - `multilingual-e5-large`
  - `llama-text-embed-v2`
- Kept `gemini-embedding-001` as the project embedding baseline
- Replaced keyword-proxy RAGAS ground truth with authored reference answers
- Added a separate Gemini dimension ablation for `3072` vs `1536`

## Embedding model comparison
- Models compared:
  - `gemini-embedding-001 @ 3072`
  - `multilingual-e5-large @ 1024`
  - `llama-text-embed-v2 @ 1024`
- On the capped benchmark subset, all 3 tied on aggregate retrieval metrics:
  - `hit@5 = 1.0`
  - `MRR = 1.0`
  - `nDCG@5 = 0.8757`
  - `routing@3 = 1.0`

## Dimension ablation
- Compared:
  - `gemini-embedding-001 @ 3072`
  - `gemini-embedding-001 @ 1536`
- On the capped benchmark subset, the 2 dimensions also tied on aggregate retrieval metrics.

## RAGAS with authored reference answers
- Evaluated with:
  - `RAGAS_GENERATION_MODEL = gemini-2.5-flash`
  - `RAGAS_EVALUATOR_MODEL = gemini-2.5-flash`
- Reason for override:
  - the workspace hit the Gemini 3.1 Pro daily quota during regeneration
- Scores:
  - `faithfulness = 0.9103`
  - `answer_relevancy = 0.3804`
  - `context_precision = 0.8667`
  - `context_recall = 0.8`

## Slide-safe interpretation
- We can now truthfully say `3 embedding models were compared`.
- The stronger RAGAS grounding metrics came from replacing keyword proxies with authored reference answers.
- Answer relevancy is still the weakest RAGAS metric, so there is still a real improvement story to tell.
- The benchmark subset is still small and capped, so the tie between embedding models should be framed as `this benchmark slice did not separate them`, not `all models are equally good`.

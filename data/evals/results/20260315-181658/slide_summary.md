# Evaluation Summary

Generated at: `2026-03-15T18:29:41.051002+00:00`

## Verification

- Backend tests: `exit 0`
- Frontend build: `exit 0`

## Retrieval Benchmark

- `hit@5`: `1.0`
- `MRR`: `1.0`
- `nDCG@5`: `0.8757`
- `routing@3`: `1.0`

## RAGAS

- `answer_relevancy`: `0.3842`
- `context_precision`: `0.0`
- `context_recall`: `0.0`
- `faithfulness`: `0.7374`

## Slide-Worthy Comparisons

- Retrieval configurations compared: `3`
- Chunking winner: `chunk-600-overlap-80`
- Pipeline modes compared: `4`
- Hybrid vs dense-only compared: `2`

## Latency

- Average end-to-end latency: `14838.51 ms`
- Average retrieval latency: `12.57 ms`
- Average generation latency: `11273.43 ms`

## Demo Query Validation

- `distributed_scaling` -> mode `corpus-backed`, citations `4`, grounding `True`, answer-quality `True`
- `mixed_precision_tradeoffs` -> mode `corpus-backed`, citations `4`, grounding `True`, answer-quality `True`
- `deployment_stack` -> mode `corpus-backed`, citations `4`, grounding `True`, answer-quality `True`
- `runtime_release_change` -> mode `web-backed`, citations `4`, grounding `True`, answer-quality `True`

## Notes

- The embedding comparison is labeled as `3 retrieval configurations`, not 3 embedding models.
- The chunking ablation uses the keyword baseline as a fast local proxy; the selected assessment benchmark still uses `gemini-embedding-001` at 3072 dimensions.
- API call counts in the latency artifact are a cost proxy, not a billing report.
- The current RAGAS run uses expected-term proxies as `ground_truth`, so treat those scores as directional rather than a final benchmark.

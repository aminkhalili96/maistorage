# Gemini 3.1 Pro RAGAS Rerun Attempt

Artifact bundle: `/Users/amin/dev/maistorage/data/evals/results/20260315-195514`

## Outcome

- Attempted to rerun RAGAS with:
  - `RAGAS_GENERATION_MODEL = gemini-3.1-pro-preview`
  - `RAGAS_EVALUATOR_MODEL = gemini-3.1-pro-preview`
- Result:
  - failed before score generation with `429 ResourceExhausted`

## What blocked it

- Gemini reported quota exhaustion for:
  - daily requests
  - per-minute requests
  - daily input tokens
  - per-minute input tokens
- The failing model was `gemini-3.1-pro`

## Practical conclusion

- The latest successful authored-reference RAGAS scores remain the run in:
  - `/Users/amin/dev/maistorage/data/evals/results/20260316-030114`
- The application default still targets:
  - `gemini-3.1-pro-preview`
- A successful 3.1 Pro RAGAS rerun requires additional quota or waiting for the quota window to reset.

# Interview Demo Playbook

## Positioning Statement

Present this project as a **small production-style AI system**, not a student RAG chatbot.

Suggested opener:

> I built this as an agentic RAG assistant for NVIDIA AI infrastructure and training optimization. The goal is to help solution architects, deployment engineers, and AI engineers query large technical documentation corpora quickly with grounded citations, visible reasoning steps, and explicit failure handling.

## What To Show Live

### 1. Start on corpus status

Show:

- `snapshot_id`
- source-family counts
- indexed chunk count
- last refresh timestamp

Talking point:

> For demo reliability I use a versioned offline snapshot of official NVIDIA docs. In a production setup, the same manifest and normalization pipeline can refresh changed sources incrementally and only promote a new index after evaluation passes.

Live-validated note:

- the live Pinecone verification in this workspace used the bundled demo chunk set for speed and cost control
- the full NVIDIA snapshot remains bundled locally under `data/corpus`
- the slide-ready evaluation artifacts live under:
  - [20260315-181658](/Users/amin/dev/maistorage/data/evals/results/20260315-181658)
  - [20260316-030114](/Users/amin/dev/maistorage/data/evals/results/20260316-030114)

### 2. Demo query 1: troubleshooting

Question:

`Why is 4-GPU training scaling poorly?`

Show:

- distributed query routing
- accepted and rejected evidence
- corpus-backed answer
- trace events for retrieval, grading, and validation

Talking point:

> This demonstrates that the system is not just doing nearest-neighbor lookup. It classifies the question, narrows the search to the most relevant source families, grades the retrieved chunks, and only then synthesizes an answer with citations.

### 3. Demo query 2: multi-source synthesis

Question:

`When should I use mixed precision training and what are the tradeoffs?`

Show:

- multiple citations from different source families
- grounding and answer-quality checks
- trust panel

Talking point:

> This path is useful because many real operator questions are not answered by one chunk. The system has to combine evidence across performance, training, and library docs while keeping the answer grounded.

### 4. Demo query 3: deployment guidance

Question:

`What NVIDIA stack is needed to deploy training workloads on Linux or Kubernetes?`

Show:

- deployment/runtime routing
- infrastructure-oriented citations
- why this matters to MaiStorage customers

Talking point:

> This is closer to the business use case: helping teams deploy and troubleshoot AI infrastructure, not just answering textbook questions about CUDA.

### 5. Controlled failure or fallback case

Question:

`What changed in the latest NVIDIA Container Toolkit release?`

Show:

- web-backed fallback if Tavily is configured
- otherwise an explicit insufficient-evidence response
- visible response mode in the trust panel

Talking point:

> I wanted the system to be honest when the local corpus is not enough. If the question is recency-sensitive, it can use a controlled web fallback. If that still does not provide grounded evidence, it refuses instead of hallucinating.

Live-validated note:

- after tightening the recency policy, this query now routes to `web-backed` instead of staying `corpus-backed`
- the demo-query validation artifact confirms:
  - 3 `corpus-backed`
  - 1 `web-backed`
  - citations, grounding, and answer-quality pass on all 4 demo prompts

## Trust Model

Use this wording:

- `corpus-backed`: answer stayed within the offline NVIDIA corpus
- `web-backed`: answer required controlled fallback because the question was time-sensitive or out of corpus
- `insufficient-evidence`: the system refused to guess

Trust signals to point out in the UI:

- response mode
- source families
- citation count
- rejected chunk count
- grounding check result
- answer-quality check result

Live-verified in this workspace:

- `corpus-backed` for the 3 primary demo questions
- `web-backed` for the recency-sensitive Container Toolkit query
- visible citation count, rejected chunk count, grounding result, and answer-quality result

## Design Decisions and Tradeoffs

| Decision | Alternatives Considered | Why This Choice |
| --- | --- | --- |
| Offline snapshot | live crawl only | stable, reproducible, demo-safe |
| Pinecone in assessment mode | in-memory only | closer to enterprise deployment patterns and scalable indexing |
| Agentic retrieval | one-shot traditional RAG | improves observability, trust, and failure handling |
| `gemini-3.1-pro-preview` + `gemini-embedding-001` | older Gemini models or local embeddings | strong reasoning with stable retrieval embeddings |
| Controlled Tavily fallback | always corpus only or always web-enabled | keeps the primary story corpus-grounded while still handling recency |

## MaiStorage Business Framing

Use examples that sound like actual customer work:

- sizing and troubleshooting on-prem NVIDIA AI clusters
- answering deployment questions about drivers, CUDA, containers, or Kubernetes
- helping engineers reason about GPU utilization, communication bottlenecks, and throughput
- reducing time spent manually searching hundreds of pages of technical documentation

Suggested wording:

> Imagine a MaiStorage client deploying an NVIDIA-based AI server cluster. Their engineers need quick answers on scaling, deployment, and performance tuning, but they still need source attribution because these are operational decisions. This system gives them grounded answers with citations and traceability.

## Release Gates

Use these as interview talking points rather than hard promises:

- `hit@5 >= 0.80` on the golden retrieval set
- `MRR >= 0.60` on the same benchmark
- 100% citation coverage for factual demo answers
- grounding and answer-quality checks must pass for rehearsed demo queries
- refreshed corpus should not be promoted if retrieval quality regresses

Measured evidence from the latest artifact bundle:

- retrieval benchmark:
  - `hit@5 = 1.0`
  - `MRR = 1.0`
  - `nDCG@5 = 0.8757`
  - `routing@3 = 1.0`
- latency:
  - average end-to-end latency: `14838.51 ms`
  - average retrieval latency: `12.57 ms`
  - average generation latency: `11273.43 ms`
- pipeline ablation:
  - retrieval-only keyword-hit rate: `0.75`
  - retrieval + grading keyword-hit rate: `1.0`
  - full agentic flow was the only mode that produced the intended `web-backed` route for the recency query

## Limitations and Roadmap

Talk about limitations directly:

- latency grows because synthesis and validation require multiple model calls
- external APIs still introduce availability and cost constraints
- the offline corpus is a snapshot and needs refresh to stay current
- preview model availability can vary by account
- the refreshed RAGAS run improved materially once keyword-proxy ground truth was replaced with authored reference answers
- answer relevancy is still the weakest RAGAS metric, so there is still room to improve how directly some answers address the question
- the refreshed RAGAS run used `gemini-2.5-flash` for the eval path because the workspace hit the daily Gemini 3.1 Pro quota during regeneration

Concrete next steps:

- cache repeated retrieval plans and answer runs
- parallelize or simplify validation calls
- improve reranking and expand the golden benchmark set
- automate scheduled refresh and candidate-index promotion
- add LangSmith-backed production monitoring once live credentials are available

Current caveat:

- LangSmith wiring exists, but the current LangSmith token in this workspace was rejected by the API with `401 Invalid token`, so trace upload was not claimed as verified

## Claim Discipline

Only present these as **verified** if you actually ran them in your environment:

- live Gemini generation
- live Pinecone indexing and query path
- live Tavily fallback
- live RAGAS scores
- Docker end-to-end startup
- live LangSmith trace upload

Safe wording when something is implemented but not live-verified:

> The integration path is implemented in the codebase, but live validation still depends on runtime credentials and environment setup.

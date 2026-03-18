# Interview Demo Playbook

## Positioning Statement

Present this project as a **small production-style AI system**, not a student RAG chatbot.

Suggested opener:

> I built this as an agentic RAG assistant for NVIDIA AI infrastructure and training optimization. The goal is to help solution architects, deployment engineers, and AI engineers query large technical documentation corpora quickly with grounded citations, visible reasoning steps, and explicit failure handling. The system goes well beyond one-shot retrieval — it classifies queries, rewrites them using an LLM when results are weak, grades retrieved evidence, reflects on its own answer quality, and has a 5-mode trust model with a 4-layer fallback chain before it ever refuses to answer.

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
- the repo now uses a unified 50-question benchmark file:
  - 46 corpus-backed doc/RAG rows
  - 2 direct-chat routing rows
  - 1 refusal row
  - 1 recency-sensitive fallback row
- RAGAS targets the 46-question corpus-backed authored-reference subset of that benchmark

### 2. Demo query 1: troubleshooting

Question:

`Why is 4-GPU training scaling poorly?`

Show:

- distributed query routing
- accepted and rejected evidence
- corpus-backed answer
- trace events for retrieval, grading, and validation
- progressive trace: each node's event appears in the UI as it completes, not all at once

Talking point:

> This demonstrates that the system is not just doing nearest-neighbor lookup. It classifies the question, narrows the search to the most relevant source families, grades the retrieved chunks, and only then synthesizes an answer with citations. Watch the trace panel — you can see each pipeline stage complete in real time.

### 3. Demo query 2: multi-source synthesis

Question:

`When should I use mixed precision training and what are the tradeoffs?`

Show:

- multiple citations from different source families
- grounding and answer-quality checks
- Self-RAG reflection scores in the trace (relevance, groundedness, completeness)
- trust panel

Talking point:

> This path is useful because many real operator questions are not answered by one chunk. The system has to combine evidence across performance, training, and library docs while keeping the answer grounded. After generating, a Self-RAG reflection node scores the answer on groundedness — if the score is below 3 out of 5, the system forces a fallback rather than serving a weak answer.

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

> I wanted the system to be honest when the local corpus is not enough. If the question is recency-sensitive, it can use a controlled web fallback. If that still does not provide grounded evidence, it refuses instead of hallucinating. This is the 4-layer fallback in action: corpus → post-generation Tavily → LLM general knowledge → refusal.

Live-validated note:

- after tightening the recency policy, this query now routes to `web-backed` instead of staying `corpus-backed`
- the demo-query validation artifact confirms:
  - 3 `corpus-backed`
  - 1 `web-backed`
  - citations, grounding, and answer-quality pass on all 4 demo prompts

### 6. llm-knowledge demo: general ML concept

Question:

`What is the difference between data parallelism and model parallelism?`

Show:

- `llm-knowledge` response mode (purple badge)
- no corpus citations (no NVIDIA-specific chunk needed)
- OpenAI general-knowledge answer labeled transparently

Talking point:

> General AI/ML concepts that are not NVIDIA-specific do not need corpus retrieval. The system recognizes this and answers from OpenAI's general knowledge, labeled transparently. This avoids a pointless retrieval round-trip while still being honest about the answer source.

### 7. Optional: show query rewriting in the trace

Question (ask something slightly ambiguous first, then watch the rewrite):

`GPU bandwidth issues in training`

Show:

- the `rewrite` trace event with `rewrite_method: "llm"` or `"static_expansion"`
- the rewritten query vs the original

Talking point:

> When the first retrieval pass returns low-confidence results, the system does not give up. It calls OpenAI to rephrase the query into a more specific, technical form that is more likely to match documentation language. If the LLM call fails, it falls back to static expansion terms. The trace shows which method was used.

### 8. Optional: show query decomposition (if enabled)

Enable with `QUERY_DECOMPOSITION_ENABLED=true` in `.env`.

Question:

`What are the differences between NVLink and InfiniBand, and how do I configure NCCL for each?`

Show:

- the classification trace event noting decomposition into sub-questions
- multiple retrieval passes in the trace
- merged result set

Talking point:

> Multi-part questions are detected automatically. The system decomposes them into focused sub-questions, runs retrieval for each independently, then merges the evidence before synthesis. This avoids the "lost in the middle" problem where a single long query retrieves chunks that only partially address the question.

## Trust Model

Use this wording:

- `corpus-backed`: answer stayed within the offline NVIDIA corpus
- `web-backed`: answer required controlled fallback because the question was time-sensitive or out of corpus
- `llm-knowledge`: corpus and Tavily both exhausted, or general AI/ML concept not NVIDIA-specific; OpenAI answers from its own knowledge (purple badge)
- `insufficient-evidence`: the system refused to guess after all fallback layers were tried
- `direct-chat`: conversational turn, no retrieval needed

Trust signals to point out in the UI:

- response mode badge
- source families
- citation count
- rejected chunk count
- grounding check result
- answer-quality check result
- Self-RAG reflection scores (in trace)
- confidence percentage

Live-verified in this workspace:

- `corpus-backed` for the 3 primary demo questions
- `web-backed` for the recency-sensitive Container Toolkit query
- visible citation count, rejected chunk count, grounding result, and answer-quality result

## 4-Layer Fallback Architecture

This is a key talking point that distinguishes this from a basic RAG system:

1. **Corpus retrieval** — hybrid keyword + semantic search, document grading, adaptive top-k, LLM-powered query rewriting on retry
2. **Post-generation Tavily** — if answer quality fails and Tavily not yet tried, fetch web results and regenerate (implemented as a proper LangGraph node: `post_generation_fallback`)
3. **LLM general knowledge** — if corpus and Tavily both exhausted, OpenAI answers from its own knowledge, labeled `llm-knowledge`
4. **Refusal** — if nothing works, `insufficient-evidence` — the system never hallucinates

## Pipeline Node Walk-Through (for technical interviewers)

```
classify → retrieve → document_grading
  → [rewrite_if_needed | fallback_if_needed]
  → generate → self_reflect → grounding_check → answer_quality_check
  → [post_generation_fallback → generate → self_reflect → grounding_check → answer_quality_check]
```

| Node | What it does |
|------|-------------|
| `classify` | Routes to `direct_chat` or `doc_rag`; detects multi-part questions for decomposition |
| `retrieve` | Adaptive top-k hybrid search; runs per sub-question if decomposed |
| `document_grading` | Scores and filters chunks; caps diversity per source |
| `rewrite_if_needed` | LLM-powered query rewrite (falls back to static expansion) |
| `fallback_if_needed` | Tavily web search when corpus confidence is too low |
| `generate` | OpenAI synthesis with citation enforcement |
| `self_reflect` | LLM scores relevance/groundedness/completeness (1-5); forces fallback if groundedness < 3 |
| `grounding_check` | Checks citation coverage and detects hedging phrases |
| `answer_quality_check` | Checks answer relevance and minimum length |
| `post_generation_fallback` | Post-generation Tavily retry when quality fails |

## Traditional RAG vs Agentic RAG Talking Points

| Dimension | Traditional RAG | This System |
|-----------|----------------|-------------|
| Retrieval | Fixed top-k, one pass | Adaptive top-k, query rewriting, multi-pass |
| Query handling | Single query | Decomposition into sub-questions |
| Evidence grading | None | Document grading with diversity cap |
| Answer validation | None | Grounding check + quality check + Self-RAG reflection |
| Failure handling | Hallucinate or error | 4-layer fallback chain |
| Observability | None | Full trace panel, progressive streaming |
| Caching | None | Semantic cache (cosine similarity, LRU eviction) |

## Design Decisions and Tradeoffs

| Decision | Alternatives Considered | Why This Choice |
| --- | --- | --- |
| Offline snapshot | live crawl only | stable, reproducible, demo-safe |
| Pinecone in assessment mode | in-memory only | closer to enterprise deployment patterns and scalable indexing |
| Agentic retrieval | one-shot traditional RAG | improves observability, trust, and failure handling |
| 4-tier OpenAI (GPT-5.4 Nano / GPT-5 Mini / GPT-5.4) + `text-embedding-3-large` | single model or Gemini | per-task cost optimization; routing costs ~50x less than synthesis |
| Controlled Tavily fallback | always corpus only or always web-enabled | keeps the primary story corpus-grounded while still handling recency |
| LangGraph for orchestration | custom imperative loop | explicit state transitions, easy to extend, visualizable graph |
| Self-RAG reflection | no post-generation check | catches weak answers before they reach the user |
| Progressive SSE streaming | batch emit after pipeline | better UX; user sees reasoning as it happens |

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
- RAGAS (10-question slim set, GPT-5.4 + gpt-4.1 judge):
  - `faithfulness = 0.694`
  - `answer_relevancy = 0.735`
  - `context_precision = 0.683`
  - `context_recall = 0.675`
- latency:
  - average end-to-end latency: `14838.51 ms`
  - average retrieval latency: `12.57 ms`
  - average generation latency: `11273.43 ms`
- pipeline ablation:
  - retrieval-only keyword-hit rate: `0.75`
  - retrieval + grading keyword-hit rate: `1.0`
  - full agentic flow was the only mode that produced the intended `web-backed` route for the recency query

**RAGAS answer_relevancy talking point:**

> The 10-question RAGAS slim set gives a balanced picture: faithfulness at 0.694 and answer_relevancy at 0.735 reflect that the system grounds answers well but could be more direct. The earlier 46-question Gemini run scored 0.91 on faithfulness but 0.38 on relevancy — those numbers used a different model and judge. The current OpenAI pipeline is more balanced across all four metrics. Faithfulness remains the metric I care most about: it means when the system answers, it stays grounded.

## Limitations and Roadmap

Talk about limitations directly:

- latency grows because synthesis and validation require multiple model calls (Self-RAG adds one extra OpenAI call per answer)
- external APIs still introduce availability and cost constraints
- the offline corpus is a snapshot and needs refresh to stay current
- preview model availability can vary by account
- answer relevancy is still the weakest RAGAS metric; there is room to improve how directly some answers address the question
- the main app uses a 4-tier OpenAI model strategy (GPT-5.4 Nano for routing, GPT-5 Mini for pipeline, GPT-5.4 for synthesis)
- semantic cache and query decomposition are opt-in (disabled by default) because they require OpenAI to be available

**Production roadmap summary:**

| Layer | What it would look like |
|-------|------------------------|
| Corpus refresh | Scheduled crawler checks source hashes weekly; only re-chunks changed docs; diff-based upsert to Pinecone |
| Index promotion | New candidate index runs full retrieval benchmark + RAGAS before replacing prod namespace; rollback if metrics regress |
| Observability | LangSmith traces all agent runs; dashboards for latency, fallback rate, grounding pass rate per query class |
| Multi-tenancy | One Pinecone namespace per customer corpus; embedder and index config per-namespace; same agent graph |
| Latency | Semantic cache for repeated queries; parallelize validation calls; stream generation starts before validation |
| Confidence calibration | Replace binary pass/fail grounding with a 0–1 score; surface to UI as a confidence meter |

**Multi-turn query contextualization:** the system already contextualizes short follow-up questions (≤8 tokens) by prepending the prior user turn to the retrieval query. This means a follow-up like "What about latency?" after a question on NVLink gets enough context to retrieve the right chunks. In production this would extend to a sliding conversation window.

Concrete next steps:

- parallelize or simplify validation calls to reduce latency
- improve reranking and expand the golden benchmark set
- automate scheduled refresh and candidate-index promotion
- add LangSmith-backed production monitoring once live credentials are available
- run a fresh RAGAS eval with the LLM-knowledge and Self-RAG improvements in place

Current caveat:

- LangSmith wiring exists, but the current LangSmith token in this workspace was rejected by the API with `401 Invalid token`, so trace upload was not claimed as verified

## Handling Corpus Freshness (if asked)

**Q: "What if NVIDIA updates their docs? How do you keep the corpus current?"**

### Layer 1 — Runtime recency handling (already built)
"The system detects recency-sensitive queries — 'latest', 'current',
'release notes' — and routes them to live web search via Tavily instead
of the static corpus. The response is labeled `web-backed` so the user
sees the provenance. We never serve stale data for time-sensitive questions."

### Layer 2 — Freshness metadata infrastructure (already built)
"Every chunk carries a `snapshot_id`, `retrieved_at` timestamp, and
`content_hash` (SHA256). The download script hashes every page at crawl
time and stores it in a manifest. The infrastructure for change detection
is already in place — we just need to schedule the comparison."

Point to: `/api/sources` response → shows `snapshot_id` and `last_refresh_at`.
Point to: `data/corpus/manifest.json` → shows per-source `content_hash`.

### Layer 3 — Production roadmap (would add)
"For production, I'd add three things:
1. **Scheduled re-crawl** — weekly cron that re-fetches each source URL,
   hashes the content, flags sources with changed hashes
2. **Incremental re-indexing** — only re-chunk and re-embed changed
   sources; the manifest already tracks which sources changed
3. **HTTP conditional requests** — use ETag/If-Modified-Since headers
   so unchanged docs are skipped at the network level

This is incremental work, not a rewrite — the hashing and manifest
infrastructure is already in place."

### If they probe deeper
- "How long would a re-crawl take?" → "22 online sources, ~30 seconds
  with async HTTP. Normalization only runs for changed sources."
- "What about vector DB updates?" → "Pinecone supports upsert by ID.
  Delete old chunks for the changed source, upsert the new ones.
  No full index rebuild."
- "What about versioning?" → "The manifest tracks snapshot_id (date-based)
  and content_hash per source. We could keep previous snapshots for rollback."

## Claim Discipline

Only present these as **verified** if you actually ran them in your environment:

- live OpenAI generation
- live Pinecone indexing and query path
- live Tavily fallback
- live RAGAS scores
- Docker end-to-end startup
- live LangSmith trace upload

Safe wording when something is implemented but not live-verified:

> The integration path is implemented in the codebase, but live validation still depends on runtime credentials and environment setup.

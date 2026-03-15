import { FormEvent, useEffect, useMemo, useState } from "react";

import { getIngestionStatus, getRetrievalEvals, getSources, startIngestion, streamChat } from "./api";
import type { ChatDonePayload, ChatTurn, Citation, EvalRow, IngestionStatus, SourcesResponse, TraceEvent } from "./types";

type PanelTab = "assistant" | "corpus";

interface DemoPrompt {
  label: string;
  title: string;
  question: string;
  goal: string;
  path: string;
  variant: "core" | "failure";
}

const DEMO_PROMPTS: DemoPrompt[] = [
  {
    label: "Demo 1",
    title: "Corpus-backed troubleshooting",
    question: "Why is 4-GPU training scaling poorly?",
    goal: "Show distributed routing, retrieval, grading, and grounded troubleshooting.",
    path: "distributed -> corpus-backed",
    variant: "core",
  },
  {
    label: "Demo 2",
    title: "Multi-source cited synthesis",
    question: "When should I use mixed precision training and what are the tradeoffs?",
    goal: "Show cross-source synthesis with explicit citations and trust checks.",
    path: "training -> multi-source answer",
    variant: "core",
  },
  {
    label: "Demo 3",
    title: "Deployment guidance",
    question: "What NVIDIA stack is needed to deploy training workloads on Linux or Kubernetes?",
    goal: "Show deployment/runtime routing with infrastructure-focused sources.",
    path: "deployment -> corpus-backed",
    variant: "core",
  },
  {
    label: "Failure case",
    title: "Controlled insufficiency / fallback",
    question: "What changed in the latest NVIDIA Container Toolkit release?",
    goal: "Show honest failure handling or web-backed fallback instead of hallucination.",
    path: "recency -> web-backed or insufficient-evidence",
    variant: "failure",
  },
];

const STORY_CARDS = [
  {
    title: "MaiStorage fit",
    bullets: [
      "Useful for solution architects sizing NVIDIA-based AI servers.",
      "Useful for deployment teams debugging Linux, driver, container, and Kubernetes setup.",
      "Useful for AI engineers reasoning about throughput, communication, and hardware tradeoffs.",
    ],
  },
  {
    title: "Key tradeoffs",
    bullets: [
      "Offline snapshot improves demo reliability but needs a refresh pipeline for freshness.",
      "Pinecone reduces operational burden in assessment mode compared with local-only storage.",
      "Agentic RAG costs more latency than one-shot RAG, but gives observability and safer failure handling.",
    ],
  },
  {
    title: "Release gates",
    bullets: [
      "Retrieval must keep relevant evidence in the top results before answer metrics matter.",
      "Every factual paragraph needs citations and the grounding check must pass.",
      "Updated corpus snapshots should not be promoted if retrieval quality regresses.",
    ],
  },
  {
    title: "Limitations and roadmap",
    bullets: [
      "Latency is dominated by LLM calls for synthesis and validation.",
      "Preview model and external APIs introduce availability risk.",
      "Next steps: caching, async grading, stronger reranking, and scheduled refresh promotion.",
    ],
  },
];

const RELEASE_GATES = [
  "hit@5 >= 0.80 for the golden retrieval set",
  "MRR >= 0.60 on the same benchmark",
  "100% citation coverage for factual demo answers",
  "Grounding and answer-quality checks must pass on rehearsed demo queries",
];

function describeResponseMode(mode: string): string {
  if (mode === "web-backed") {
    return "Used only when the corpus evidence is too weak or the question is explicitly time-sensitive.";
  }
  if (mode === "insufficient-evidence") {
    return "The system refused to guess because it could not find grounded evidence.";
  }
  return "Answer stayed within the bundled NVIDIA corpus and its indexed citations.";
}

function formatConfidence(confidence: number): string {
  if (confidence <= 0) {
    return "n/a";
  }
  return `${Math.round(confidence * 100)}%`;
}

export default function App() {
  const [tab, setTab] = useState<PanelTab>("assistant");
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [draftAnswer, setDraftAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [trace, setTrace] = useState<TraceEvent[]>([]);
  const [sources, setSources] = useState<SourcesResponse | null>(null);
  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus | null>(null);
  const [evalRows, setEvalRows] = useState<EvalRow[]>([]);
  const [responseMode, setResponseMode] = useState("corpus-backed");
  const [retryCount, setRetryCount] = useState(0);
  const [confidence, setConfidence] = useState(0);
  const [groundingPassed, setGroundingPassed] = useState<boolean | null>(null);
  const [answerQualityPassed, setAnswerQualityPassed] = useState<boolean | null>(null);
  const [rejectedChunkCount, setRejectedChunkCount] = useState(0);
  const [citationCount, setCitationCount] = useState(0);
  const [queryClass, setQueryClass] = useState("pending");
  const [sourceFamilies, setSourceFamilies] = useState<string[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void refreshMeta();
  }, []);

  async function refreshMeta() {
    try {
      const [sourcesPayload, ingestionPayload] = await Promise.all([getSources(), getIngestionStatus()]);
      setSources(sourcesPayload);
      setIngestionStatus(ingestionPayload);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Failed to load metadata.");
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || isSending) {
      return;
    }

    const userTurn: ChatTurn = { role: "user", content: trimmed };
    const nextHistory = [...history, userTurn];
    setHistory(nextHistory);
    setQuestion("");
    setDraftAnswer("");
    setCitations([]);
    setTrace([]);
    setResponseMode("corpus-backed");
    setRetryCount(0);
    setConfidence(0);
    setGroundingPassed(null);
    setAnswerQualityPassed(null);
    setRejectedChunkCount(0);
    setCitationCount(0);
    setQueryClass("pending");
    setSourceFamilies([]);
    setError(null);
    setIsSending(true);

    try {
      await streamChat(trimmed, nextHistory, {
        onTrace: (eventPayload) => setTrace((current) => [...current, eventPayload]),
        onCitation: (citationPayload) =>
          setCitations((current) => {
            if (current.some((item) => item.chunk_id === citationPayload.chunk_id)) {
              return current;
            }
            return [...current, citationPayload];
          }),
        onAnswerChunk: (text) => setDraftAnswer((current) => current + text),
        onDone: (payload: ChatDonePayload) => {
          const answer = String(payload.answer ?? "").trim();
          setDraftAnswer(answer);
          setResponseMode(payload.response_mode ?? "corpus-backed");
          setRetryCount(Number(payload.retry_count ?? 0));
          setConfidence(Number(payload.confidence ?? 0));
          setGroundingPassed(Boolean(payload.grounding_passed));
          setAnswerQualityPassed(Boolean(payload.answer_quality_passed));
          setRejectedChunkCount(Number(payload.rejected_chunk_count ?? 0));
          setCitationCount(Number(payload.citation_count ?? 0));
          setQueryClass(payload.query_class ?? "general");
          setSourceFamilies(payload.source_families ?? []);
          setHistory((current) => [...current, { role: "assistant", content: answer }]);
        },
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Streaming failed.");
    } finally {
      setIsSending(false);
    }
  }

  async function handleRefreshEval() {
    try {
      setEvalRows(await getRetrievalEvals());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Failed to load retrieval evaluations.");
    }
  }

  async function handleStartIngestion() {
    try {
      await startIngestion();
      await refreshMeta();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Failed to start ingestion.");
    }
  }

  const familyCards = useMemo(() => Object.entries(sources?.families ?? {}), [sources]);
  const sourceFamilySummary = sourceFamilies.length > 0 ? sourceFamilies.join(", ") : "Pending classification";

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">MaiStorage Q1 Prototype</p>
          <h1>NVIDIA AI Infrastructure and Training Optimization Advisor</h1>
          <p className="hero-copy">
            A small production-style AI system for NVIDIA infrastructure and training questions, with grounded citations,
            explicit trust signals, a visible agent trace, and an offline-first corpus snapshot for assessment reliability.
          </p>
        </div>
        <div className="hero-meta">
          <div className="meta-card">
            <span>Indexed chunks</span>
            <strong>{sources?.indexed_chunks ?? 0}</strong>
          </div>
          <div className="meta-card">
            <span>Snapshot</span>
            <strong>{sources?.snapshot_id ?? ingestionStatus?.snapshot_id ?? "pending"}</strong>
          </div>
          <div className="meta-card">
            <span>Mode</span>
            <strong>{sources?.app_mode ?? "dev"}</strong>
          </div>
          <div className="meta-card">
            <span>Source families</span>
            <strong>{Object.keys(sources?.families ?? {}).length}</strong>
          </div>
        </div>
      </header>

      <nav className="top-nav">
        <button className={tab === "assistant" ? "tab active" : "tab"} onClick={() => setTab("assistant")}>
          Assistant
        </button>
        <button className={tab === "corpus" ? "tab active" : "tab"} onClick={() => setTab("corpus")}>
          Corpus & Evaluation
        </button>
      </nav>

      {error ? <div className="error-banner">{error}</div> : null}

      {tab === "assistant" ? (
        <main className="assistant-grid">
          <section className="panel chat-panel">
            <div className="panel-header">
              <div>
                <p className="panel-label">Live demo</p>
                <h2>Run the rehearsed interview paths</h2>
              </div>
              <div className="prompt-card-grid">
                {DEMO_PROMPTS.map((prompt) => (
                  <button
                    key={prompt.question}
                    type="button"
                    className={`prompt-card ${prompt.variant}`}
                    onClick={() => setQuestion(prompt.question)}
                  >
                    <span className="eyebrow">{prompt.label}</span>
                    <strong>{prompt.title}</strong>
                    <p>{prompt.goal}</p>
                    <span className="prompt-meta">{prompt.path}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="status-strip">
              <span>Response mode: {responseMode}</span>
              <span>Retries: {retryCount}</span>
              <span>Confidence: {formatConfidence(confidence)}</span>
              <span>Query class: {queryClass}</span>
              <span>Last refresh: {sources?.last_refresh_at ?? ingestionStatus?.last_refresh_at ?? "n/a"}</span>
            </div>

            <div className="conversation">
              {history.length === 0 ? (
                <div className="empty-state">
                  Use the rehearsed prompts above to show a corpus-backed troubleshooting path, a cited multi-source
                  synthesis path, a deployment path, and one controlled insufficiency or fallback path.
                </div>
              ) : null}
              {history.map((turn, index) => (
                <article key={`${turn.role}-${index}`} className={`message ${turn.role}`}>
                  <span className="message-role">{turn.role === "user" ? "Operator" : "Advisor"}</span>
                  <p>{turn.content}</p>
                </article>
              ))}
              {isSending ? (
                <article className="message assistant live">
                  <span className="message-role">Advisor</span>
                  <p>{draftAnswer || "Building retrieval plan..."}</p>
                </article>
              ) : null}
            </div>

            <form className="composer" onSubmit={handleSubmit}>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Example: Why is 4-GPU training scaling poorly even though GPU utilization is high?"
                rows={4}
              />
              <div className="composer-footer">
                <span>Hybrid retrieval, grading, rewrite, grounding check, answer-quality check, fallback, citations</span>
                <button type="submit" disabled={isSending || !question.trim()}>
                  {isSending ? "Streaming..." : "Ask advisor"}
                </button>
              </div>
            </form>
          </section>

          <section className="panel side-panel">
            <div className="panel-section">
              <p className="panel-label">Trust model</p>
              <h3>Why this answer is trustworthy</h3>
              <div className="signal-grid">
                <div className="signal-card">
                  <span>Response mode</span>
                  <strong>{responseMode}</strong>
                  <p>{describeResponseMode(responseMode)}</p>
                </div>
                <div className="signal-card">
                  <span>Source routing</span>
                  <strong>{queryClass}</strong>
                  <p>{sourceFamilySummary}</p>
                </div>
                <div className="signal-card">
                  <span>Grounding check</span>
                  <strong>{groundingPassed === null ? "Pending" : groundingPassed ? "Passed" : "Failed"}</strong>
                  <p>Every factual paragraph should stay anchored to cited evidence.</p>
                </div>
                <div className="signal-card">
                  <span>Answer quality</span>
                  <strong>{answerQualityPassed === null ? "Pending" : answerQualityPassed ? "Passed" : "Failed"}</strong>
                  <p>The answer should directly address the operator question before the run is accepted.</p>
                </div>
              </div>

              <ul className="trust-list">
                <li>
                  <strong>Citation coverage:</strong> {citationCount} cited chunks attached to the answer.
                </li>
                <li>
                  <strong>Weak evidence filtered:</strong> {rejectedChunkCount} chunks rejected before synthesis.
                </li>
                <li>
                  <strong>Claim discipline:</strong> the system surfaces `corpus-backed`, `web-backed`, or
                  `insufficient-evidence` instead of pretending every answer is equally reliable.
                </li>
              </ul>
            </div>

            <div className="panel-section">
              <p className="panel-label">Citations</p>
              <h3>Grounding evidence</h3>
              {citations.length === 0 ? <p className="muted">Citations will appear as the answer is assembled.</p> : null}
              <div className="stack">
                {citations.map((citation) => (
                  <a key={citation.chunk_id} href={citation.url} target="_blank" rel="noreferrer" className="citation-card">
                    <strong>{citation.title}</strong>
                    <span>
                      {citation.section_path} · {citation.source_kind}
                    </span>
                    <p>{citation.snippet}</p>
                  </a>
                ))}
              </div>
            </div>

            <div className="panel-section">
              <p className="panel-label">Trace</p>
              <h3>Agent reasoning steps</h3>
              {trace.length === 0 ? <p className="muted">Classification, retrieval, grading, retries, and fallback events land here.</p> : null}
              <div className="stack">
                {trace.map((eventPayload, index) => (
                  <div key={`${eventPayload.type}-${index}`} className="trace-card">
                    <strong>{eventPayload.type}</strong>
                    <p>{eventPayload.message}</p>
                    <code>{JSON.stringify(eventPayload.payload)}</code>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </main>
      ) : (
        <main className="corpus-grid">
          <section className="panel">
            <div className="panel-header inline">
              <div>
                <p className="panel-label">Corpus status</p>
                <h2>Source-family coverage</h2>
              </div>
              <button onClick={handleStartIngestion}>Run ingestion</button>
            </div>

            <div className="status-strip">
              <span>Active: {String(ingestionStatus?.active ?? false)}</span>
              <span>Demo corpus: {String(ingestionStatus?.loaded_demo_corpus ?? false)}</span>
              <span>Changed sources: {ingestionStatus?.changed_sources.length ?? 0}</span>
            </div>

            <div className="status-strip">
              <span>Snapshot: {ingestionStatus?.snapshot_id ?? sources?.snapshot_id ?? "pending"}</span>
              <span>Updated: {ingestionStatus?.updated_at ?? "n/a"}</span>
              <span>Refreshed: {ingestionStatus?.last_refresh_at ?? sources?.last_refresh_at ?? "n/a"}</span>
            </div>

            <div className="family-grid">
              {familyCards.map(([family, count]) => (
                <div key={family} className="family-card">
                  <span>{family}</span>
                  <strong>{count} sources</strong>
                </div>
              ))}
            </div>

            <div className="source-table">
              {sources?.sources.map((source) => (
                <div key={source.id} className="source-row">
                  <div>
                    <strong>{source.title}</strong>
                    <p>
                      {source.doc_family} · {source.doc_type}
                    </p>
                  </div>
                  <div className="tag-row">
                    {source.product_tags.map((tag) => (
                      <span key={tag} className="tag">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="panel-header inline">
              <div>
                <p className="panel-label">Evaluation</p>
                <h2>Interview framing and release gates</h2>
              </div>
              <button onClick={handleRefreshEval}>Load metrics</button>
            </div>

            <div className="story-grid">
              {STORY_CARDS.map((card) => (
                <div key={card.title} className="story-card">
                  <strong>{card.title}</strong>
                  <ul>
                    {card.bullets.map((bullet) => (
                      <li key={bullet}>{bullet}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            <div className="panel-section">
              <p className="panel-label">Release gates</p>
              <h3>What has to be true before I trust a refresh or demo run</h3>
              <ul className="gate-list">
                {RELEASE_GATES.map((gate) => (
                  <li key={gate}>{gate}</li>
                ))}
              </ul>
            </div>

            <div className="panel-section">
              <p className="panel-label">Benchmarks</p>
              <h3>Retrieval proof before answer-level proof</h3>
            </div>
            <div className="stack">
              {evalRows.length === 0 ? (
                <p className="muted">
                  Load the retrieval benchmark set to inspect `hit@5`, `MRR`, `nDCG`, and source-family routing before
                  discussing answer quality.
                </p>
              ) : null}
              {evalRows.map((row) => (
                <div key={row.question} className="eval-card">
                  <strong>{row.question}</strong>
                  <p>{row.query_class}</p>
                  <code>{JSON.stringify(row.metrics)}</code>
                  <span>Expected: {row.expected_sources.join(", ")}</span>
                  <span>Retrieved: {row.retrieved_sources.join(", ")}</span>
                </div>
              ))}
            </div>
          </section>
        </main>
      )}
    </div>
  );
}

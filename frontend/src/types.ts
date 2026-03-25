/**
 * Frontend TypeScript types — mirrors the Pydantic models in backend/app/models.py.
 *
 * These types define the JSON contracts between the backend SSE stream and the
 * React frontend. Keep in sync with the backend models when adding new fields.
 */

export type QueryClass =
  | "training_optimization"
  | "distributed_multi_gpu"
  | "deployment_runtime"
  | "hardware_topology"
  | "general";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  /** Per-message metadata (only on assistant turns, saved when answer completes) */
  citations?: Citation[];
  trace?: TraceEvent[];
  donePayload?: ChatDonePayload;
}

export interface Citation {
  chunk_id: string;
  title: string;
  url: string;
  citation_url: string;
  domain: string;
  section_path: string;
  snippet: string;
  source_kind: string;
  source_id?: string;
  score?: number;
  char_count?: number;
  page?: number;
}

export interface TraceEvent {
  type: string;
  message: string;
  payload: Record<string, unknown>;
  timestamp?: number;
}

export interface SourceRecord {
  id: string;
  title: string;
  url: string;
  doc_family: string;
  doc_type: string;
  crawl_prefix: string;
  product_tags: string[];
  pdf_url?: string | null;
  source_kind?: string;
}

export interface SourcesResponse {
  sources: SourceRecord[];
  families: Record<string, number>;
  indexed_chunks: number;
  app_mode: string;
  snapshot_id?: string | null;
  last_refresh_at?: string | null;
}

export interface IngestionStatus {
  active: boolean;
  last_job_id?: string | null;
  snapshot_id?: string | null;
  source_counts: Record<string, number>;
  chunk_counts: Record<string, number>;
  changed_sources: string[];
  errors: string[];
  updated_at?: string | null;
  last_refresh_at?: string | null;
  loaded_demo_knowledge_base: boolean;
}

export interface EvalRow {
  question: string;
  query_class: string;
  expected_sources: string[];
  retrieved_sources: string[];
  metrics: Record<string, number | string | boolean>;
}

export interface Conversation {
  id: string;
  title: string;
  history: ChatTurn[];
  createdAt: number;
  updatedAt: number;
}

/** Final SSE "done" event payload — contains all quality signals for the MetaBar display. */
export interface ChatDonePayload {
  answer: string;
  assistant_mode: "direct_chat" | "doc_rag" | "live_query";
  confidence: number;                     // 0.0–1.0, weighted avg of top-3 rerank scores
  used_fallback: boolean;                 // Whether Tavily web search was invoked
  response_mode: string;                  // Trust label shown in the UI badge
  retry_count: number;
  grounding_passed: boolean;              // Did citation grounding check pass?
  answer_quality_passed: boolean;
  rejected_chunk_count: number;
  citation_count: number;
  query_class: string;
  source_families: string[];
  model_used: string;
  generation_degraded?: boolean;          // True if LLM failed and keyword fallback was used
}

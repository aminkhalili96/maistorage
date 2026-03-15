export type QueryClass =
  | "training_optimization"
  | "distributed_multi_gpu"
  | "deployment_runtime"
  | "hardware_topology"
  | "general";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface Citation {
  chunk_id: string;
  title: string;
  url: string;
  section_path: string;
  snippet: string;
  source_kind: string;
}

export interface TraceEvent {
  type: string;
  message: string;
  payload: Record<string, unknown>;
}

export interface SourceRecord {
  id: string;
  title: string;
  url: string;
  doc_family: string;
  doc_type: string;
  crawl_prefix: string;
  product_tags: string[];
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
  loaded_demo_corpus: boolean;
}

export interface EvalRow {
  question: string;
  query_class: string;
  expected_sources: string[];
  retrieved_sources: string[];
  metrics: Record<string, number | string | boolean>;
}

export interface ChatDonePayload {
  answer: string;
  confidence: number;
  used_fallback: boolean;
  response_mode: string;
  retry_count: number;
  grounding_passed: boolean;
  answer_quality_passed: boolean;
  rejected_chunk_count: number;
  citation_count: number;
  query_class: string;
  source_families: string[];
}

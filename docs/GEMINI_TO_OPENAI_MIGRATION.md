# Gemini to OpenAI Migration Guide

Complete inventory of all Gemini integration points in the MaiStorage agentic RAG project, with exact line numbers, API endpoints, and usage patterns.

**Generated:** 2026-03-18
**Scope:** Full end-to-end Gemini replacement with OpenAI APIs
**Files affected:** 6 backend files, 1 frontend file, 1 environment file

---

## 1. Core Abstractions

### 1.1 GeminiReasoner Class
**File:** `backend/app/services/providers.py` (lines 109-143)
**Type:** Main LLM integration wrapper
**Instantiation:** `runtime.py` line 88

```python
class GeminiReasoner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.Client(timeout=settings.gemini_timeout)
    
    @property
    def enabled(self) -> bool:
        return bool(self.settings.gemini_api_key)
    
    def generate_text(self, prompt: str, model: str | None = None) -> str:
        # API: https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent
        # Returns: str (text content from response)
        # Raises: GeminiSafetyFilterError on SAFETY/RECITATION/BLOCKLIST finish_reason
```

**Migration mapping:**
- `enabled` property → check if OpenAI API key exists
- `generate_text()` → `openai.ChatCompletion.create()` with `model` param
- Finish reason checks → map OpenAI content_filter_results to GeminiSafetyFilterError

### 1.2 GoogleGeminiEmbedder Class
**File:** `backend/app/services/providers.py` (lines 61-98)
**Type:** Embedding provider
**Usage:** Optional; only in assessment mode or when `EMBEDDER_PROVIDER=google`

```python
class GoogleGeminiEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_embedding_model  # "gemini-embedding-001"
        self.dimensions = settings.gemini_embedding_dimensions  # 3072
        self.document_task_type = settings.gemini_embedding_document_task_type
        self.query_task_type = settings.gemini_embedding_query_task_type
        self.client = httpx.Client(timeout=settings.embedder_timeout)
    
    def _embed(self, text: str, task_type: str) -> list[float]:
        # API: https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent
        # taskType: "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
        # outputDimensionality: 3072
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Calls _embed(..., self.document_task_type) for each text
        
    def embed_query(self, text: str) -> list[float]:
        # Calls _embed(..., self.query_task_type) for single text
```

**Migration mapping:**
- Embedding model → `openai.Embedding.create(model="text-embedding-3-large")`
- Task type distinction → OpenAI embeddings don't have task types; use same model for both
- Dimensions → text-embedding-3-large provides 3072 dimensions (matches Gemini default)

### 1.3 GeminiSafetyFilterError Exception
**File:** `backend/app/services/providers.py` (lines 101-106)
**Usage:** Raised by `GeminiReasoner.generate_text()` when safety/recitation/blocklist blocks content

```python
class GeminiSafetyFilterError(RuntimeError):
    def __init__(self, finish_reason: str) -> None:
        self.finish_reason = finish_reason  # "SAFETY", "RECITATION", "BLOCKLIST"
        super().__init__(f"Gemini response blocked by safety filter: {finish_reason}")
```

**Migration mapping:**
- OpenAI uses `content_filter_results` object instead of finish_reason
- Create equivalent `OpenAISafetyFilterError` with mapping:
  - `finish_reason="SAFETY"` → `content_filter_results.hate=True` or `.violence=True`
  - `finish_reason="RECITATION"` → no direct equivalent; treat as safety
  - `finish_reason="BLOCKLIST"` → `finish_reason="content_filter"`

---

## 2. Configuration & Runtime Wiring

### 2.1 Settings Class (config.py)
**File:** `backend/app/config.py` (lines 24-166)
**Gemini-specific fields:**

| Line | Field | Type | Default | Env Var | Purpose |
|------|-------|------|---------|---------|---------|
| 41 | `gemini_api_key` | `str \| None` | `None` | `GEMINI_API_KEY` | API authentication |
| 42 | `gemini_model` | `str` | `"gemini-2.5-flash"` | `GEMINI_MODEL` | Default LLM model |
| 43 | `gemini_allowed_models` | `tuple[str, ...]` | `ALLOWED_GEMINI_MODELS` | (hardcoded) | Whitelist for model selection |
| 44 | `gemini_embedding_model` | `str` | `"gemini-embedding-001"` | `GEMINI_EMBEDDING_MODEL` | Embedding model |
| 45 | `gemini_embedding_dimensions` | `int` | `3072` | `GEMINI_EMBEDDING_DIMENSIONS` | Embedding output size |
| 46 | `gemini_embedding_document_task_type` | `str` | `"RETRIEVAL_DOCUMENT"` | `GEMINI_DOCUMENT_TASK_TYPE` | Task type for doc embeddings |
| 47 | `gemini_embedding_query_task_type` | `str` | `"RETRIEVAL_QUERY"` | `GEMINI_QUERY_TASK_TYPE` | Task type for query embeddings |
| 73 | `gemini_temperature` | `float` | `0.2` | `GEMINI_TEMPERATURE` | LLM sampling temperature |
| 74 | `gemini_timeout` | `float` | `60.0` | `GEMINI_TIMEOUT` | HTTP timeout for Gemini API |

**Constants:**
- Line 11-15: `ALLOWED_GEMINI_MODELS = ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.1-pro-preview")`

**Properties & methods:**
- Line 78-80: `generation_model` property returns `self.gemini_model`
- Line 83-84: `is_assessment_mode` property checks `self.app_mode == "assessment"`
- Line 86-109: `validate_runtime()` method includes gemini validation (lines 91-96)

**Migration mapping:**
- Rename `gemini_*` fields to `openai_*` throughout
- `ALLOWED_GEMINI_MODELS` → `ALLOWED_OPENAI_MODELS = ("gpt-5.4", "gpt-4-turbo", "gpt-4")`
- `gemini_embedding_model` → `openai_embedding_model = "text-embedding-3-large"`
- Remove `gemini_embedding_document_task_type` and `gemini_embedding_query_task_type` (OpenAI doesn't use)
- Keep `gemini_temperature` or rename to `llm_temperature`
- Keep `gemini_timeout` or rename to `openai_timeout`

### 2.2 buildEmbedder Factory Function
**File:** `backend/app/services/providers.py` (lines 219-229)

```python
def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder_provider == "google":
        try:
            return GoogleGeminiEmbedder(settings)
        except Exception:
            if settings.is_assessment_mode:
                raise
            return KeywordEmbedder()
    if settings.is_assessment_mode:
        raise ValueError("Assessment mode requires the Google embedder.")
    return KeywordEmbedder()
```

**Migration mapping:**
- Change condition to `if settings.embedder_provider == "openai"`
- Return `OpenAIEmbedder(settings)` instead of `GoogleGeminiEmbedder(settings)`
- Update error message: "Assessment mode requires the OpenAI embedder."

### 2.3 Runtime Instantiation (runtime.py)
**File:** `backend/app/runtime.py`

| Line | Code | Purpose |
|------|------|---------|
| 88 | `reasoner = GeminiReasoner(settings)` | Instantiate LLM reasoner |
| 90 | `agent = AgentService(..., reasoner, ...)` | Wire reasoner into agent service |

**Migration mapping:**
- Line 88: `reasoner = OpenAIReasoner(settings)`
- Import: Change `from app.services.providers import GeminiReasoner` to `OpenAIReasoner`

### 2.4 Environment Variables (.env.example)
**File:** `.env.example` (lines 4-9)

```bash
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_EMBEDDING_DIMENSIONS=3072
GEMINI_DOCUMENT_TASK_TYPE=RETRIEVAL_DOCUMENT
GEMINI_QUERY_TASK_TYPE=RETRIEVAL_QUERY
GEMINI_TEMPERATURE=0.2
GEMINI_TIMEOUT=60.0
```

**Migration mapping:**
- `GEMINI_API_KEY` → `OPENAI_API_KEY`
- `GEMINI_MODEL` → `OPENAI_MODEL=gpt-5.4`
- `GEMINI_EMBEDDING_MODEL` → `OPENAI_EMBEDDING_MODEL=text-embedding-3-large`
- `GEMINI_EMBEDDING_DIMENSIONS` → Keep (still 3072 for text-embedding-3-large)
- `GEMINI_DOCUMENT_TASK_TYPE` → Remove (OpenAI doesn't use)
- `GEMINI_QUERY_TASK_TYPE` → Remove (OpenAI doesn't use)
- `GEMINI_TEMPERATURE` → `LLM_TEMPERATURE=0.2` or `OPENAI_TEMPERATURE=0.2`
- `GEMINI_TIMEOUT` → `OPENAI_TIMEOUT=60.0`

---

## 3. Agent Service (agent.py)

**File:** `backend/app/services/agent.py` (1464 lines)
**Import:** Line 19 `from app.services.providers import GeminiReasoner`

### 3.1 Instantiation
**Line 178-189:** `AgentService.__init__` constructor

```python
def __init__(
    self,
    settings: Settings,
    retrieval: RetrievalService,
    reasoner: GeminiReasoner,  # ← Dependency injection
    tavily: TavilyClient | None = None,
    embedder: Embedder | None = None,
    max_retries: int = 2,
) -> None:
    self.settings = settings
    self.retrieval = retrieval
    self.reasoner = reasoner  # ← Stored as instance variable
    self.tavily = tavily or TavilyClient(settings)
    self.embedder = embedder
    self.max_retries = max_retries
    self.semantic_cache = SemanticCache(embedder) if settings.semantic_cache_enabled else None
    self._thread_local_emit: Callable[[TraceEvent], None] | None = None
```

**Migration:** Change type hint from `GeminiReasoner` to `OpenAIReasoner`

### 3.2 Query Decomposition
**Lines 246-278:** Multi-part question detection and decomposition

```python
def _should_decompose(self, question: str) -> bool:
    # Line 254: if not self.reasoner.enabled: return False
    # Detects multi-part questions with specific patterns
    
def _decompose_question(self, question: str) -> list[str]:
    # Line 265: result = self.reasoner.generate_text(decomposition_prompt)
    # Calls Gemini to split into 2-3 focused sub-questions
    # Returns: list of sub-questions
```

**API call pattern:**
```python
decomposition_prompt = f"""You are a query optimizer...
Question: {question}
Return JSON array of 2-3 focused sub-questions."""
result = self.reasoner.generate_text(decomposition_prompt)
```

**Migration mapping:**
- Keep the decomposition logic unchanged
- Replace `self.reasoner.generate_text(prompt)` with OpenAI equivalent
- Parse JSON response the same way

### 3.3 Query Rewriting (LLM-based)
**Lines 393-436:** Low-confidence query optimization

```python
def _llm_rewrite_query(self, question: str, top_results: list[RetrievalResult]) -> str | None:
    # Line 404: result = self.reasoner.generate_text(rewrite_prompt)
    # Calls Gemini to rephrase low-confidence queries
    
def _graph_rewrite(state: GraphState) -> GraphState:
    # Line 427: rewritten = self._llm_rewrite_query(state.question, top_results)
    # LangGraph node wrapper
```

**API call pattern:**
```python
rewrite_prompt = f"""You are a search query optimizer...
Original query: {question}
Return only the optimized query."""
result = self.reasoner.generate_text(rewrite_prompt)
```

**Migration mapping:**
- Keep rewriting logic unchanged
- Replace `self.reasoner.generate_text(prompt)` with OpenAI

### 3.4 Self-Reflection (RAG Quality Scoring)
**Lines 548-603:** LLM-based answer quality assessment

```python
def _graph_self_reflect(state: GraphState) -> GraphState:
    # Line 554: if not self.reasoner.enabled: return state  # Skip if no LLM
    # Line 571: result = self.reasoner.generate_text(reflect_prompt)
    # Calls Gemini to score relevance/groundedness/completeness (1-5)
    # Line 590: groundedness = int(float(parsed.get("groundedness", 3)))
    # Line 601: if groundedness < 3: state.grounding_passed = False
```

**API call pattern:**
```python
reflect_prompt = f"""Score this RAG answer...
Answer: {state.draft_answer}
Return JSON with relevance, groundedness, completeness (1-5)."""
result = self.reasoner.generate_text(reflect_prompt)
parsed = json.loads(result)
```

**Migration mapping:**
- Keep scoring logic unchanged
- Replace `self.reasoner.generate_text(prompt)` with OpenAI
- JSON parsing remains the same

### 3.5 Query Reformulation (Follow-ups)
**Lines 1101-1148:** Convert follow-up questions to standalone queries

```python
def _reformulate_follow_up(self, question: str, history: list[ChatTurn]) -> str:
    # Line 1112: if not self._is_follow_up(question): return question
    # Line 1126: if self.reasoner.enabled:
    #            reformulated = self.reasoner.generate_text(reformulation_prompt)
    # Line 1143: Falls back to static concat if Gemini unavailable
    
    # Returns: (reformulated_query, method: "llm" | "static")
```

**API call pattern:**
```python
reformulation_prompt = f"""Rewrite as standalone...
History: {last_qa_pairs}
Original: {question}
Return only the rewritten question."""
reformulated = self.reasoner.generate_text(reformulation_prompt)
```

**Migration mapping:**
- Keep reformulation logic unchanged
- Replace `self.reasoner.generate_text(prompt)` with OpenAI
- Static fallback remains the same

### 3.6 Synthesis (Main Answer Generation)
**Lines 1187-1230:** Generate grounded answer from context

```python
def _synthesize_answer(
    self,
    context_text: str,
    question: str,
    history_context: str | None = None,
) -> str:
    # Line 1207: if not self.reasoner.enabled:
    #            return self._direct_chat_answer(question, history=...)
    # Line 1215: result = self.reasoner.generate_text(synthesis_prompt)
    # Returns: Answer text with [N] citation markers
```

**API call pattern:**
```python
synthesis_prompt = f"""You are an NVIDIA AI infrastructure assistant...
Question: {question}
History context: {history_context}
Context:
{numbered_context_blocks}

Instructions: Write an answer grounded in the context..."""
result = self.reasoner.generate_text(synthesis_prompt)
```

**Key difference:** Synthesis prompt includes detailed citation rules and numbered context passages. Migration must preserve this structure.

**Migration mapping:**
- Keep prompt structure unchanged
- Replace `self.reasoner.generate_text(prompt)` with OpenAI
- Citation marker extraction remains the same

### 3.7 Conversational Responses (Direct Chat)
**Lines 916-940:** Respond to conversational turns without retrieval

```python
def _direct_chat_answer(self, question: str, history: list[ChatTurn]) -> str:
    # Line 927: if not self.reasoner.enabled:
    #           return "I apologize..."  # Fallback response
    # Line 934: result = self.reasoner.generate_text(chat_prompt)
    # Returns: Conversational response
```

**API call pattern:**
```python
chat_prompt = f"""You are a friendly NVIDIA AI infrastructure assistant...
Conversation history:
{formatted_history}
User: {question}
Assistant:"""
result = self.reasoner.generate_text(chat_prompt)
```

**Migration mapping:**
- Keep conversational logic unchanged
- Replace `self.reasoner.generate_text(prompt)` with OpenAI

### 3.8 LLM Knowledge Fallback
**Lines 1362-1376:** Answer from LLM general knowledge when corpus/web exhausted

```python
def _llm_knowledge_answer(
    self,
    question: str,
    history_context: str | None = None,
) -> str:
    # Line 1370: result = self.reasoner.generate_text(knowledge_prompt)
    # Returns: Answer based on LLM's training knowledge
    # Sets response_mode to "llm-knowledge"
```

**API call pattern:**
```python
knowledge_prompt = f"""You are an NVIDIA AI infrastructure expert...
History context: {history_context}
Question: {question}
Provide an answer based on your knowledge..."""
result = self.reasoner.generate_text(knowledge_prompt)
```

**Migration mapping:**
- Keep fallback logic unchanged
- Replace `self.reasoner.generate_text(prompt)` with OpenAI

### 3.9 Live Web Query (Tavily Integration)
**Lines 781-875:** Real-time data queries (weather, stock prices, news)

```python
def _run_live_query(self, question: str) -> str:
    # Line 804: if not self.tavily or not self.tavily.enabled:
    #           return self._direct_chat_answer(question, ...)
    # Line 808: web_results = self.tavily.search(question)
    # Line 820: answer = self._synthesize_answer(web_results_text, question)
    # Uses the standard synthesis pipeline with web data
```

**Note:** This doesn't directly call Gemini; it uses `_synthesize_answer()` which internally calls Gemini. No migration changes needed beyond synthesis.

### 3.10 Main Entry Point & Progressive Streaming
**Lines 655-677, 1260-1459:** Request routing and SSE streaming

```python
def run(self, ...) -> AgentRunState:
    # Line 667: return self._run_with_optional_trace(...)
    # Main synchronous entry point
    
async def stream(self, ...) -> AsyncGenerator[TraceEvent, None]:
    # Line 1360: Manages asyncio.Queue for progressive event emission
    # Line 1368: _run_and_enqueue() calls sync self._sync_run()
    # Line 1390: Wraps in try/except for error handling
```

**Key pattern:** Progressive SSE uses `asyncio.Queue` to emit events as each LLM call completes. Migration doesn't change this architecture.

**Migration mapping:**
- No changes to streaming logic
- All LLM calls within stream continue to use OpenAI

### 3.11 Fallback Chain (Error Handling)
**Lines 680-758:** Main error recovery flow

```python
def _run_with_optional_trace(self, ...) -> AgentRunState:
    # Line 702: _reformulate_follow_up() — reformulates follow-ups using Gemini
    # Line 706: classify_assistant_mode() — routes to pipeline
    # Line 752: LLM-knowledge fallback — calls _llm_knowledge_answer()
```

**Migration mapping:**
- No logic changes to fallback chain
- All Gemini calls within chain replaced with OpenAI

---

## 4. Frontend Model Selection

### 4.1 Model Dropdown (App.tsx)
**File:** `frontend/src/App.tsx`

| Line | Code | Purpose |
|------|------|---------|
| 15 | `const MODEL_STORAGE_KEY = "maistorage-selected-model"` | localStorage key |
| 17-21 | `const AVAILABLE_MODELS = [...]` | Array of available models |
| 48-54 | `const [selectedModel, setSelectedModel] = useState(...)` | State management with localStorage persistence |
| 257-267 | `<select className="model-selector" ...>` | HTML select element |
| 158 | `streamChat(trimmed, nextHistory, selectedModel, ...)` | Pass selected model to API |

**Current models (lines 17-21):**
```javascript
const AVAILABLE_MODELS = [
  { id: "gemini-2.5-flash", label: "Gemini 2.5 Flash" },
  { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
  { id: "gemini-3.1-pro-preview", label: "Gemini 3.1 Pro" },
];
```

**Migration mapping:**
```javascript
const AVAILABLE_MODELS = [
  { id: "gpt-4o", label: "GPT-4o" },
  { id: "gpt-4-turbo", label: "GPT-4 Turbo" },
  { id: "gpt-5.4", label: "GPT-5.4" },
];
```

### 4.2 Model Passing to Backend
**File:** `frontend/src/api.ts`

The `streamChat()` function receives `selectedModel` and includes it in the request. No frontend changes needed beyond model list; the backend API remains unchanged (accepts `model` field).

---

## 5. API Endpoints Summary

### 5.1 Gemini API Endpoints (to be replaced)

| Endpoint | Method | Purpose | Called from |
|----------|--------|---------|-------------|
| `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` | POST | Text generation | `GeminiReasoner.generate_text()` (providers.py:122-129) |
| `https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent` | POST | Embeddings | `GoogleGeminiEmbedder._embed()` (providers.py:73-81) |

### 5.2 OpenAI API Equivalents

| Gemini Endpoint | OpenAI Equivalent | Implementation |
|-----------------|-------------------|-----------------|
| `models/{model}:generateContent` | `/chat/completions` | `openai.ChatCompletion.create(model=model, messages=[...])` |
| `models/{model}:embedContent` | `/embeddings` | `openai.Embedding.create(model=model, input=text)` |

---

## 6. Error Handling & Safety

### 6.1 Gemini Safety Filter Handling
**File:** `backend/app/services/providers.py` (lines 136-138)

```python
finish_reason = candidate.get("finishReason", "")
if finish_reason in ("SAFETY", "RECITATION", "BLOCKLIST"):
    raise GeminiSafetyFilterError(finish_reason)
```

**OpenAI equivalent:**
- OpenAI uses `content_filter_results` object instead of `finish_reason`
- Check: `response.choices[0].content_filter_results`
- Fields: `hate`, `self_harm`, `sexual`, `violence` (all boolean)

**Migration mapping:**
```python
if response.choices[0].content_filter_results:
    results = response.choices[0].content_filter_results
    if any([results.get(field) for field in ['hate', 'self_harm', 'sexual', 'violence']]):
        raise OpenAISafetyFilterError("content_filter")
```

### 6.2 Exception Handling in agent.py
**Line 1390-1450:** Try/except in streaming handles all exceptions

```python
try:
    self._sync_run(...)
except Exception as exc:
    # Emits error SSE event
    # Sets generation_degraded=True
```

**No changes needed:** Exception handling is framework-agnostic

---

## 7. Configuration Validation (Assessment Mode)

**File:** `backend/app/config.py` (lines 91-96)

```python
if not self.gemini_api_key:
    errors.append("GEMINI_API_KEY is required in assessment mode.")
if self.gemini_model not in self.gemini_allowed_models:
    errors.append("Assessment mode requires GEMINI_MODEL to be an allowed Gemini model.")
if self.gemini_embedding_model != "gemini-embedding-001":
    errors.append("Assessment mode requires GEMINI_EMBEDDING_MODEL=gemini-embedding-001.")
if self.embedder_provider != "google":
    errors.append("Assessment mode requires EMBEDDER_PROVIDER=google.")
```

**Migration mapping:**
- Change all `gemini_*` to `openai_*`
- Update allowed models list
- Update embedding model constraint
- Update embedder_provider check to `"openai"`

---

## 8. Import Statements to Update

### Backend
| File | Line | Current | Replace with |
|------|------|---------|--------------|
| `agent.py` | 19 | `from app.services.providers import GeminiReasoner` | `OpenAIReasoner` |
| `runtime.py` | 14 | (implicit via GeminiReasoner) | Add explicit OpenAI import |

### Frontend
No imports need updating; model list is just a data structure.

---

## 9. Migration Execution Plan

### Phase 1: Provider Classes
1. Create `OpenAIReasoner` class in `providers.py` (replace `GeminiReasoner`)
2. Create `OpenAIEmbedder` class in `providers.py` (replace `GoogleGeminiEmbedder`)
3. Create `OpenAISafetyFilterError` exception class
4. Update `build_embedder()` factory function

### Phase 2: Configuration
1. Update `Settings` dataclass: rename `gemini_*` → `openai_*` or `llm_*`
2. Update `ALLOWED_GEMINI_MODELS` → `ALLOWED_OPENAI_MODELS`
3. Update `get_settings()` function: read new env vars
4. Update `.env.example` with new variable names
5. Update `validate_runtime()` method for assessment mode

### Phase 3: Runtime Wiring
1. Update `runtime.py`: instantiate `OpenAIReasoner` instead of `GeminiReasoner`
2. Update type hints throughout

### Phase 4: Agent Service
1. Update `agent.py` imports
2. Update `AgentService.__init__` type hint
3. Verify all `self.reasoner.generate_text()` calls work with OpenAI (they should — only implementation changes)
4. Test error handling with OpenAI exceptions

### Phase 5: Frontend
1. Update `AVAILABLE_MODELS` list in `App.tsx`
2. Update localStorage keys if desired (or keep for backward compatibility)

### Phase 6: Testing
1. Run backend tests with new configuration
2. Test all LLM code paths: decomposition, rewriting, synthesis, fallback
3. Test embedding functionality
4. Test error handling with OpenAI API responses
5. Test frontend model selection

---

## 10. Summary of Changes

### Files Modified
- `backend/app/services/providers.py` — 2 classes, 1 exception
- `backend/app/config.py` — Settings dataclass, 8+ fields, validation method
- `backend/app/services/agent.py` — 1 import, 1 type hint (no logic changes)
- `backend/app/runtime.py` — 1 line (class instantiation)
- `.env.example` — 8 environment variables
- `frontend/src/App.tsx` — 1 constant (AVAILABLE_MODELS)

### Scope of Changes
- **No architectural changes**: LangGraph, streaming, caching, fallback chains remain identical
- **No prompt changes**: All LLM prompts work with OpenAI
- **No API changes**: Backend endpoints remain the same
- **API transformation needed**: Gemini HTTP endpoints → OpenAI library calls
- **Safety handling changes**: `finishReason` → `content_filter_results`

### Risk Assessment
- **Low risk**: Migration is isolated to provider layer and configuration
- **Test coverage**: 190+ backend tests will verify compatibility
- **Backward compatibility**: Old config files will fail validation (intentional; forces update)
- **Fallback compatibility**: Keyword embedder still available for dev mode

---

## Appendix: Quick Reference

### Gemini to OpenAI Model Names
| Gemini Model | Use Case | OpenAI Equivalent |
|--------------|----------|------------------|
| gemini-2.5-flash | Default fast model | gpt-4o |
| gemini-2.5-pro | Larger context | gpt-4-turbo |
| gemini-3.1-pro-preview | Advanced reasoning | gpt-5.4 |
| gemini-embedding-001 | Embeddings (3072-dim) | text-embedding-3-large (3072-dim) |

### Environment Variable Mapping
| Old (Gemini) | New (OpenAI) | Default |
|--------------|--------------|---------|
| `GEMINI_API_KEY` | `OPENAI_API_KEY` | (none) |
| `GEMINI_MODEL` | `OPENAI_MODEL` | `gpt-4o` |
| `GEMINI_TEMPERATURE` | `OPENAI_TEMPERATURE` or `LLM_TEMPERATURE` | `0.2` |
| `GEMINI_TIMEOUT` | `OPENAI_TIMEOUT` or `LLM_TIMEOUT` | `60.0` |
| `GEMINI_EMBEDDING_MODEL` | `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` |
| `GEMINI_EMBEDDING_DIMENSIONS` | (keep if using text-embedding-3-large) | `3072` |
| `GEMINI_DOCUMENT_TASK_TYPE` | (remove) | N/A |
| `GEMINI_QUERY_TASK_TYPE` | (remove) | N/A |

### HTTP Client Comparison

**Gemini (httpx):**
```python
response = self.client.post(
    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    params={"key": self.api_key},
    json={...},
)
```

**OpenAI (official library - recommended):**
```python
response = client.chat.completions.create(
    model=model,
    messages=[...],
    temperature=self.temperature,
    timeout=self.timeout,
)
```

---

**Document version:** 1.0
**Last updated:** 2026-03-18
**Status:** Ready for implementation

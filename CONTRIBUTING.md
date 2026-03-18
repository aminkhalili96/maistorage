# Contributing

## Local setup

1. Clone the repo and copy `.env.example` to `.env`
2. Backend: `cd backend && python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. Frontend: `cd frontend && npm install`
4. Start backend: `uvicorn app.main:app --reload --port 8000` (from `backend/`)
5. Start frontend: `npm run dev` (from `frontend/`)

See `CLAUDE.md` for full environment variable documentation.

## Running tests

```bash
# Backend (from repo root)
PYTHONPATH=backend backend/.venv/bin/pytest backend/tests -q

# Frontend (from frontend/)
cd frontend && npm test
```

## Code style

- Python: type hints on all public functions, Pydantic models for data structures
- TypeScript: strict mode, React functional components with `React.memo` where appropriate
- No `any` casts unless interfacing with untyped external data (trace payloads)

## Environment modes

- `APP_MODE=dev` — in-memory index, no external API keys needed
- `APP_MODE=assessment` — requires Gemini, Pinecone, and Google embedder credentials

## Adding corpus sources

1. Add entry to `data/sources/nvidia_sources.json`
2. Add content to `data/corpus/raw/markdown/{source_id}/content.md`
3. Run `PYTHONPATH=backend backend/.venv/bin/python scripts/normalize_corpus.py`
4. Verify: `PYTHONPATH=backend backend/.venv/bin/pytest backend/tests/test_simple_chat_flow.py -v`

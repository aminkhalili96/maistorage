# Assistant Router Test List

The assistant router decides between:

- `direct_chat`
- `doc_rag`

The goal is conservative routing:

- default to `direct_chat` for normal conversation and general knowledge
- enter `doc_rag` only for clearly NVIDIA-doc, operational, deployment, hardware, optimization, or recency-sensitive infra questions

## Direct Chat Cases

- `hello`
- `what is nvidia`
- `who founded nvidia`
- `where is nvidia headquartered`
- `tell me about H100`
- `what does cuda stand for`
- `can you check the docs for me?`
- follow-up general chat after an assistant reply that happens to mention docs:
  - history contains assistant text mentioning NVIDIA docs
  - current question: `what day is it`

## Doc RAG Cases

- `What NVIDIA stack is needed on Linux for GPU containers?`
- `How do I install CUDA on Ubuntu?`
- `Why is 4-GPU training scaling poorly?`
- `According to official NVIDIA docs, when should I use mixed precision?`
- `What changed in the latest NVIDIA Container Toolkit release?`
- `Show me the NVIDIA guide for GPU Operator installation`
- `Show me the guide for TensorRT`

## Context Carry-Over Cases

If the recent **user-authored** context is clearly doc/RAG, short ambiguous follow-ups should stay in `doc_rag`:

- previous user turn: `Show me the official NVIDIA docs for CUDA installation`
- follow-up:
  - `where?`
  - `which guide?`
  - `CUDA`

## Failure Modes This List Protects Against

- routing on raw vendor/entity mentions alone
- routing based on assistant-authored history instead of user-authored context
- generic `docs/documentation` wording forcing RAG without NVIDIA/official/operational context
- short follow-up prompts losing doc/RAG context after a clearly doc-focused turn

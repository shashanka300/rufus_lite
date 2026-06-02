# Local Rufus

A local clone of Amazon Rufus — a conversational AI shopping assistant — running entirely on your own hardware via Ollama.

## What it does

- Embeds Amazon ESCI product catalog with BGE-M3 into a local Qdrant vector store
- Answers natural-language shopping questions by retrieving relevant products and grounding an LLM response in them (RAG)
- Interactive REPL or single-query CLI

## Hardware target

RTX 5090 (32 GB VRAM). All models fit locally; no cloud API required.

## Project layout

```
rufus/          core library
  retriever.py  BGE-M3 embedding + Qdrant nearest-neighbour search
  rag.py        retrieval → prompt → Ollama answer pipeline
  llm.py        thin Ollama chat wrapper (streaming + non-streaming)
scripts/
  download_datasets.py   download ESCI, SQID, Amazon Reviews, MG-ShopDial
  ingest_esci.py         embed ESCI products → upsert into Qdrant
  query.py               interactive query CLI
data/           local data and Qdrant storage (git-ignored)
```

## Setup

```bash
# 1. Install uv if you don't have it
curl -Ls https://astral.sh/uv/install.sh | sh

# 2. Create venv and install deps
uv sync

# 3. Pull the LLM via Ollama
ollama pull qwen3.5:latest
```

## Quickstart

### Step 1 — download data

```bash
# ESCI only (needed for Week 1 RAG)
uv run python scripts/download_datasets.py esci

# All datasets
uv run python scripts/download_datasets.py all
```

### Step 2 — ingest products into Qdrant

```bash
# Full English ESCI catalog (~1.3M products, takes ~20 min on GPU)
uv run python scripts/ingest_esci.py

# Quick smoke-test with 5 000 products
uv run python scripts/ingest_esci.py --limit 5000

# Rebuild from scratch
uv run python scripts/ingest_esci.py --reset
```

### Step 3 — query Rufus

```bash
# Interactive REPL
uv run python scripts/query.py

# Single question
uv run python scripts/query.py --q "best noise-cancelling headphones under $200"

# More retrieved products, different model
uv run python scripts/query.py --top-k 8 --model qwen3.5:27b
```

## Models

| Role | Default | Notes |
|---|---|---|
| LLM | `qwen3.5:latest` | Conversation + Q&A |
| Embedding | `BAAI/bge-m3` | Dense 1024-dim, loaded from HF |
| Reranker | — | Week 3+ |

Recommended Ollama pulls for the full stack:
```bash
ollama pull qwen3.5:27b      # main reasoning model
ollama pull glm-5.1          # best tool calling / JSON output
ollama pull lfm2.5-thinking  # 1.2B intent router (<50 ms)
```

## Datasets

| Dataset | Purpose | Size |
|---|---|---|
| Amazon ESCI | Product catalog + search relevance labels | ~2.6 M query-item pairs |
| SQID | ESCI + images + CLIP embeddings | several GB |
| Amazon Reviews 2023 | Customer reviews for RAG Q&A grounding | metadata ~1–2 GB, full reviews ~22 GB |
| MG-ShopDial | Multi-goal shopping dialogues for fine-tuning | small |

## Build roadmap

- **Week 1 (done):** Catalog RAG — ESCI → BGE-M3 → Qdrant → Qwen3 Q&A
- **Week 2:** Conversational layer — LangGraph session state, intent classification, MG-ShopDial few-shot
- **Week 3:** Multimodal — SQID + CLIP, dual-encoder retrieval, fuse with BGE-M3 text scores
- **Week 4:** Evaluation — ESCI NDCG benchmark + eval harness

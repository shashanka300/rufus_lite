# Rufus Lite

A local clone of Amazon Rufus — a conversational AI shopping assistant extended with supply chain and inventory management — running entirely on your own hardware via Ollama. No cloud API required.

## What it does

**Shopping assistant**
- Natural-language product search across 156,542 image-complete products (BGE-M3 + CLIP, 100% image coverage)
- Multi-turn conversation with 13-intent routing
- Dual-encoder retrieval: BGE-M3 text + CLIP image vectors fused with RRF
- Cross-encoder reranking (bge-reranker-v2-m3)
- Personalization — user preference tracking biases reranking per session
- Cart management persisted across conversation turns
- Gift / occasion search mode
- Seller Q&A templates by category
- Product metadata enrichment: price, star rating, review snippets (5.5 M ASINs, 33 Amazon categories)
- Streaming responses via AG-UI SSE protocol

**Query intelligence**
- Ranking-modifier stripping: "best selling sarees" → "sarees" before embedding (prevents drift toward books/music)
- Short-query LLM expansion: "saries" → "sarees Indian traditional ethnic dress"
- Fast-path rule classifier for common patterns (0 ms); LLM fallback only for ambiguous cases

**Supply chain layer** *(beyond Amazon Rufus)*
- Inventory tracking — 38 K SKUs, stock levels, reorder points
- Reorder alerts with urgency scoring (critical / soon / ok)
- Demand forecasting — 30-day ahead NeuralForecast NHITS (GPU), 110 K pre-computed forecasts
- Supplier catalog with lead times (Olist sellers + SCMS vendor data)
- 5 SC intents: check_stock, reorder_alert, demand_forecast, supplier_query, sc_analytics

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5090, 32 GB VRAM |
| CUDA | 13.x (cu128 wheels) |
| PyTorch | 2.11.0+cu128 |
| Python | 3.11 |
| Package manager | uv |

All models run fp16 on CUDA. VRAM usage with all models loaded: ~10 GB (22 GB free).

## Architecture

```
User query
    │
    ▼
classify  (0 ms fast-path rules → qwen3.5:latest LLM fallback for ambiguous/short queries)
    │
    ├─ search / qa / compare / gift_search
    │       │
    │       ├─ retrieve  (BGE-M3 + CLIP → RRF → cross-encoder rerank → filter)
    │       │           └─ personalization bias (session viewed-product centroid)
    │       └─ generate  (qwen3.5:latest, streaming)
    │
    ├─ followup          → generate (reuse products from client state, no re-retrieval)
    ├─ add_to_cart       → cart node
    ├─ view_cart         → cart node
    ├─ chitchat          → generate (1-sentence friendly reply)
    │
    └─ supply chain intents
            │
            ├─ check_stock        → search_inventory → sc_generate
            ├─ reorder_alert      → bulk_reorder_check → sc_generate
            ├─ demand_forecast    → forecast_sku → sc_generate
            ├─ supplier_query     → get_all_suppliers → sc_generate
            └─ sc_analytics       → inventory_health_report → sc_generate
                    └─ sc_generate (qwen3.5:latest)
```

## Project layout

```
rufus/
  hardware.py         TF32 + cuDNN benchmark flags — imported globally
  retriever.py        BGE-M3 fp16 embedding + Qdrant nearest-neighbour
  clip_retriever.py   CLIP ViT-L/14 fp16 text/image encoding
  fusion.py           Reciprocal Rank Fusion (RRF, k=60)
  reranker.py         bge-reranker-v2-m3 fp16 cross-encoder, batch=128
  intent.py           13-intent classifier: fast-path rules + qwen3.5 LLM fallback
                        _clean_query() strips ranking modifiers before embedding
                        LLM fallback expands/corrects typos (num_ctx=2048, num_predict=128)
  llm.py              Ollama client — circuit breaker, retry, keep_alive, think=False
                        DEFAULT_MODEL = qwen3.5:latest, num_ctx=2048, num_predict=256
  rag.py              RAG context formatter + SYSTEM_PROMPT (strict anti-hallucination rules)
                        _format_context(): products from Qdrant retrieval
                        _format_raw_context(): products from client state (followup/compare)
  reviews.py          Amazon Reviews 2023 metadata + review snippet lookup (rufus_reviews.db)
  cache.py            LRU+TTL caches for embeddings and rerank results
  telemetry.py        RED-method structured logging
  qdrant.py           Shared QdrantClient singleton with auto-reconnect:
                        prefers server mode (localhost:6333), falls back to local-file
                        auto-upgrades from local-file → server if server starts later
  personalization.py  User preference profiles (SQLite) + BGE-M3 centroid rerank bias
  cart.py             Session cart (add / remove / view / total)
  seller_qa.py        Category Q&A templates (electronics/headphones/clothing/…)
  inventory.py        Supply chain SQLite schema + query helpers
  demand.py           Demand forecasting — ForecastResult, ReorderAlert
  sc_rag.py           Supply chain LLM context formatter + SC_SYSTEM_PROMPT

server.py             FastAPI AG-UI SSE backend (main entry point)
                        5-step startup warmup: qwen3:1.7b, qwen3.5:latest, Qdrant probe,
                        BGE-M3, CLIP — eliminates cold-start on first request
app.py                Legacy Streamlit UI (week 2)
frontend/
  index.html          Single-file Tailwind streaming chat UI

scripts/
  download_datasets.py            Week 1 datasets (ESCI, SQID, Amazon Reviews, MGShopDial)
  download_all_data.py            Full-scale download (HF, Kaggle, GitHub, HTTP)
  ingest_esci.py                  ESCI products → Qdrant rufus_products collection
  ingest_clip.py                  SQID CLIP vectors → Qdrant rufus_clip collection
  rebuild_products_from_clip.py   Rebuild rufus_products from rufus_clip (image-complete)
  ingest_olist.py                 Olist → rufus_sc.db (inventory, demand, suppliers)
  ingest_scms.py                  SCMS → rufus_sc.db suppliers (lead times)
  ingest_amazon_reviews_full.py   All-category metadata → rufus_reviews.db
  train_demand_forecast.py        Inventory SKUs → NeuralForecast NHITS forecasts
  eval_ndcg.py                    NDCG@K evaluation on ESCI test set
  start_qdrant.ps1                Start Qdrant standalone binary

data/                   Git-ignored — databases + Qdrant storage
  qdrant_bin/           Qdrant standalone binary (v1.18.1) + config
    qdrant.exe
    config.yaml         → storage_path: data/qdrant_storage, port 6333
  qdrant_storage/       Qdrant data (rufus_products + rufus_clip, 156 K products each)
  rufus_sc.db           Supply chain SQLite (inventory 38K, demand 651K, forecasts 110K)
  rufus_reviews.db      Amazon Reviews metadata (5.5M rows, price + ratings + review text)
  rufus_personalization.db  Co-purchase 1M, co-view 500K, item popularity 235K
  rufus_images.db       Image URL fallback DB (1.33M URLs; primary source is Qdrant payload)
```

## Setup

```bash
# 1. Install uv
curl -Ls https://astral.sh/uv/install.sh | sh   # Linux/Mac
# or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install dependencies (PyTorch cu128 pulled automatically)
uv sync

# 3. Pull LLM models
ollama pull qwen3:1.7b
ollama pull qwen3.5:latest
```

## Quickstart

### 1 — Download data

```bash
# Week 1 essentials (ESCI, SQID, Amazon Reviews Electronics, MGShopDial)
uv run python scripts/download_datasets.py all

# Full-scale download — all datasets across all groups
uv run python scripts/download_all_data.py all

# Individual groups
uv run python scripts/download_all_data.py catalog          # Amazon Reviews all categories
uv run python scripts/download_all_data.py supply           # Olist, DataCo, SCMS, UCI
uv run python scripts/download_all_data.py personalization  # RetailRocket, Instacart
uv run python scripts/download_all_data.py dialogue         # SIMMC2, DuRecDial, ReDial

# Check what's downloaded
uv run python scripts/download_all_data.py status
```

### 2 — Ingest into Qdrant

> ⚠️ **Start Qdrant before any ingest or server command.** It must be running first.

```bash
# Start Qdrant (keep this terminal open)
.\scripts\start_qdrant.ps1          # Windows PowerShell
# or: cd data/qdrant_bin && .\qdrant.exe --config-path config.yaml

# Build image-complete catalog (~20 min on GPU; the recommended path)
uv run python scripts/ingest_clip.py              # CLIP vectors first (3 min)
uv run python scripts/rebuild_products_from_clip.py  # BGE-M3 re-embed from CLIP set

# Quick smoke-test (5000 products only, no image guarantee)
uv run python scripts/ingest_esci.py --limit 5000
```

### 3 — Ingest supply chain + reviews

```bash
# Olist → inventory, demand history, suppliers (rufus_sc.db)
uv run python scripts/ingest_olist.py

# SCMS → supplier lead times (appends to rufus_sc.db)
uv run python scripts/ingest_scms.py

# Amazon Reviews all categories → price, ratings, features (rufus_reviews.db)
uv run python scripts/ingest_amazon_reviews_full.py

# Train demand forecasts (inventory SKUs → 30-day NHITS forecasts on GPU)
uv run python scripts/train_demand_forecast.py
```

### 4 — Start the server

> ⚠️ **Qdrant must already be running** before starting the server. Starting out of order causes `Collection rufus_products not found` on every request. The `qdrant.py` singleton will auto-recover if the server comes online later, but the first few requests will fail.

```bash
# Terminal 1 — Qdrant (keep running)
.\scripts\start_qdrant.ps1

# Terminal 2 — FastAPI AG-UI server
uv run uvicorn server:app --reload
```

Open **http://localhost:8000** — the chat UI loads automatically.

The server runs a 5-step warmup at startup: loads `qwen3:1.7b`, `qwen3.5:latest`, probes Qdrant, warms BGE-M3, warms CLIP. First request fires after warmup completes (~30 s); subsequent requests are fully warm.

### 5 — Evaluate retrieval quality

```bash
uv run python scripts/eval_ndcg.py                    # 300 queries, NDCG@10
uv run python scripts/eval_ndcg.py --n-queries 1000   # tighter confidence intervals
uv run python scripts/eval_ndcg.py --k 5              # NDCG@5
```

## Intent guide

| Intent | Trigger examples | Pipeline |
|---|---|---|
| `search` | "show me wireless headphones" | retrieve → generate |
| `followup` | "does it come in black?", "the second one" | reuse client-state products → generate |
| `qa` | "what's the battery life?" | seller Q&A template or retrieve → generate |
| `compare` | "compare the first two", "which is better" | retrieve → generate |
| `gift_search` | "gift for my dad", "present for my wife" | retrieve → generate (gift framing) |
| `add_to_cart` | "add to cart", "I'll take it" | cart node → CustomEvent("cart") |
| `view_cart` | "show my cart", "cart total" | cart node |
| `chitchat` | "hello", "thanks" | single-sentence generate |
| `check_stock` | "is the Anker charger in stock?", "stock level for headphones" | search_inventory → sc_generate |
| `reorder_alert` | "what needs to be reordered?", "low stock items" | bulk_reorder_check → sc_generate |
| `demand_forecast` | "forecast demand for electronics" | forecast_sku → sc_generate |
| `supplier_query` | "who are our suppliers?", "vendor lead time" | get_all_suppliers → sc_generate |
| `sc_analytics` | "show me inventory levels", "supply chain dashboard" | inventory_health_report → sc_generate |

**Fast-path rules** handle chitchat, followup, compare, gift, cart, and multi-word search queries in 0 ms. Single/two-word queries and ambiguous multi-turn messages fall through to the `qwen3.5:latest` LLM classifier, which also corrects typos and expands unusual terms.

## Models

| Role | Model | Config |
|---|---|---|
| **Generation** | `qwen3.5:latest` (6.6 GB) | `think=False` (direct kwarg), `temperature=0`, `num_ctx=2048`, `num_predict=256`, `keep_alive=60m` |
| **LLM classify fallback** | `qwen3.5:latest` | `format="json"`, `num_ctx=2048`, `num_predict=128` |
| **Embedding** | `BAAI/bge-m3` (1024-dim, fp16) | ~2 ms/query on RTX 5090, cached |
| **CLIP** | `openai/clip-vit-large-patch14` (768-dim, fp16) | ~4 ms/encode |
| **Reranker** | `BAAI/bge-reranker-v2-m3` (fp16) | batch=128, ~30 ms for 40 candidates |

> **`think=False` pitfall:** must be passed as a **direct kwarg** to `ollama.chat()`, never inside the `options` dict. Passing it inside `options` routes all output to the `thinking` field and leaves `content` empty — silent 0-token responses.

## Performance (warm, steady-state)

| Step | Latency |
|---|---|
| classify (fast-path) | 0–1 ms |
| classify (LLM fallback) | ~500 ms |
| retrieve (BGE-M3 + Qdrant + rerank) | 200–400 ms |
| generate (qwen3.5:latest, streaming) | 400–2 100 ms |
| **End-to-end** | **~1–4.5 s** |
| **TTFT (time to first token)** | **~2.5 s** |

Cold-start after server restart: first search request takes ~18 s (BGE-M3 GPU load). The startup warmup eliminates this in normal operation — requests only become live after warmup completes.

## Databases

| File | Contents | Size |
|---|---|---|
| `data/qdrant_storage/` | rufus_products (156K × 1024-dim BGE-M3) + rufus_clip (156K × 768-dim CLIP); 100% image coverage | ~3 GB |
| `rufus_reviews.db` | product_meta 5.5M rows (price/rating/features) + reviews 4.8M snippets + c4_metadata 1M Q&A | ~8 GB |
| `rufus_sc.db` | inventory 38K SKUs + demand_history 651K + forecasts 110K (3693 SKUs × 30 days) + suppliers | 117 MB |
| `rufus_personalization.db` | basket_copurchase 1M + co_view 500K + item_popularity 235K | 103 MB |
| `rufus_images.db` | 1.33M image URLs (fallback; primary source is Qdrant payload) | 158 MB |

## Hardware optimization notes

All models run fp16 on CUDA. Applied via `rufus/hardware.py` at import time:

```python
torch.set_float32_matmul_precision("high")  # TF32 tensor cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True        # auto-selects fastest kernels
```

Verified timings on RTX 5090:
- BGE-M3 embed: **~2 ms** (cached after first call)
- CLIP encode: **~4 ms**
- Rerank 40 candidates: **~30 ms** (fp16, batch=128)
- qwen3.5:latest TTFT (warm, VRAM-pinned): **~2.5 s**
- Full query pipeline (warm): **~3 s** average

## Build roadmap

- **Week 1 ✅** Catalog RAG — ESCI → BGE-M3 → Qdrant → Qwen3 Q&A
- **Week 2 ✅** Conversational layer — LangGraph, 5-intent classifier, MG-ShopDial few-shots
- **Week 3 ✅** Multimodal retrieval — SQID CLIP vectors, dual-encoder, RRF fusion
- **Week 4 ✅** Evaluation — ESCI NDCG@K benchmark, BGE-M3 vs fusion comparison
- **Week 5 ✅** Production reliability — circuit breaker, retry, LRU+TTL cache, telemetry, cross-encoder reranker
- **Week 6 ✅** AG-UI server + streaming frontend, supply chain layer, mock features, hardware optimisation
- **Week 7 ✅** All data ingested, image lookup (1.33M URLs), C4 FTS, review snippets, price display, image search
- **Week 8 ✅** Full NeuralForecast NHITS forecasts, real session personalization, image mismatch fix
- **Week 9 ✅** Image-complete catalog rebuild — dropped 1.2M imageless ESCI products, rebuilt 156K from CLIP set (100% image coverage, BGE-M3 re-embedded)
- **Week 10 ✅** Answer quality + performance hardening:
  - Switched generation to `qwen3.5:latest` — eliminates hallucinated prices, handles domain terms (sarees, kurta) and typos
  - `_clean_query()` strips ranking modifiers ("best selling", "top rated") before embedding — fixes "best selling sarees → books" drift
  - Short-query LLM expansion: "saries" → "sarees Indian traditional ethnic dress"
  - SC keyword lists tightened (removed shopping false-positives) and restored (unambiguous SC phrases)
  - `_format_raw_context()` for state-based followup/compare context
  - `qdrant.py` auto-reconnect: upgrades from local-file to server mode when Qdrant starts
  - `server.py` 5-step startup warmup including BGE-M3 and Qdrant probe
  - `num_ctx 4096 → 2048`, `num_predict 512 → 256` — faster generation, lower TTFT

**Roadmap complete.** All data is ingested. Core quality and performance issues resolved.

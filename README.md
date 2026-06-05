# Rufus Lite

A local clone of Amazon Rufus — a conversational AI shopping assistant extended with supply chain and inventory management — running entirely on your own hardware via Ollama. No cloud API required.

## What it does

**Shopping assistant**
- Natural-language product search across 1.2 M ESCI products
- Multi-turn conversation with intent routing (12 intents)
- Dual-encoder retrieval: BGE-M3 text + CLIP image vectors fused with RRF
- Cross-encoder reranking (bge-reranker-v2-m3)
- Personalization — user preference tracking biases reranking per session
- Cart management persisted across conversation turns
- Gift / occasion search mode
- Seller Q&A templates by category
- Product metadata enrichment: price, star rating, features (5.5 M ASINs across 33 Amazon categories)
- Streaming responses via AG-UI SSE protocol

**Supply chain layer** *(beyond Amazon Rufus)*
- Inventory tracking — 32 K SKUs, stock levels, reorder points
- Reorder alerts with urgency scoring (critical / soon / ok)
- Demand forecasting — 30-day ahead, rolling average + optional AutoETS
- Supplier catalog with lead times (Olist sellers + SCMS vendor data)
- 15 K pre-computed forecasts for top SKUs

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5090, 34 GB VRAM |
| CUDA | 13.x (cu128 wheels) |
| PyTorch | 2.11.0+cu128 |
| Python | 3.11 |
| Package manager | uv |

All inference is fp16 on CUDA. VRAM usage with all models loaded: ~3.2 GB (31 GB free for LLM).

## Architecture

```
User query
    │
    ▼
classify  (qwen3:1.7b, <2ms fast-path rules, LLM fallback)
    │
    ├─ search / qa / compare / gift_search
    │       │
    │       ├─ retrieve  (BGE-M3 + CLIP → RRF → cross-encoder rerank)
    │       │           └─ personalization bias
    │       └─ generate  (qwen3:1.7b, seller Q&A template or LLM)
    │
    ├─ followup          → generate (reuse products from state)
    ├─ add_to_cart        → cart node
    ├─ view_cart          → cart node
    ├─ chitchat           → chitchat node
    │
    └─ supply chain intents
            │
            ├─ check_stock        → inventory_lookup
            ├─ reorder_alert      → bulk_reorder_check
            ├─ demand_forecast    → forecast_sku
            ├─ supplier_query     → supplier lookup
            └─ sc_analytics       → category_demand_summary
                    └─ sc_generate (qwen3:1.7b)
```

## Project layout

```
rufus/
  hardware.py       TF32 + cuDNN benchmark flags — imported globally
  retriever.py      BGE-M3 fp16 embedding + Qdrant nearest-neighbour
  clip_retriever.py CLIP ViT-L/14 fp16 text/image encoding
  fusion.py         Reciprocal Rank Fusion (RRF, k=60)
  reranker.py       bge-reranker-v2-m3 fp16 cross-encoder, batch=128
  intent.py         12-intent classifier (fast-path rules + qwen3:1.7b LLM)
  graph.py          LangGraph StateGraph with MemorySaver
  state.py          ShoppingState TypedDict (messages, products, cart, sc_items …)
  llm.py            Ollama client — circuit breaker, retry, keep_alive, think=False
  rag.py            RAG context formatter + SYSTEM_PROMPT
  reviews.py        Amazon Reviews 2023 metadata lookup (rufus_reviews.db)
  cache.py          LRU+TTL caches for embeddings and rerank results
  telemetry.py      RED-method structured logging
  personalization.py User preference profiles (SQLite) + rerank bias
  cart.py           Session cart (add / remove / view / total)
  seller_qa.py      Category Q&A templates (electronics/headphones/clothing/…)
  inventory.py      Supply chain SQLite schema + query helpers
  demand.py         Demand forecasting — ForecastResult, ReorderAlert
  sc_rag.py         Supply chain LLM context formatter

server.py           FastAPI AG-UI SSE backend (main entry point)
app.py              Legacy Streamlit UI (week 2)
frontend/
  index.html        Single-file Tailwind streaming chat UI

scripts/
  download_datasets.py        Week 1 datasets (ESCI, SQID, Amazon Reviews, MGShopDial)
  download_all_data.py        Full-scale download (all sources: HF, Kaggle, GitHub, HTTP)
  ingest_esci.py              ESCI products → Qdrant rufus_products collection
  ingest_clip.py              SQID CLIP vectors → Qdrant rufus_clip collection
  ingest_olist.py             Olist → rufus_sc.db (inventory, demand, suppliers)
  ingest_scms.py              SCMS → rufus_sc.db suppliers (lead times)
  ingest_amazon_reviews_full.py  All-category metadata → rufus_reviews.db
  train_demand_forecast.py    M5 + Olist → forecasts table in rufus_sc.db
  eval_ndcg.py                NDCG@K evaluation on ESCI test set
  chat.py                     Multi-turn CLI (legacy)
  query.py                    Single-query CLI (legacy)
  start_qdrant.ps1            Start Qdrant standalone binary

data/                         Git-ignored — datasets + DB files
  esci/                       ESCI shopping queries dataset
  sqid/                       CLIP image vectors (pre-computed)
  amazon_reviews/             Electronics metadata (week 1)
  amazon_reviews_full/        All-category metadata (33 categories, 5.5 M ASINs)
  amazon_c4/                  Product Q&A pairs
  supply_chain/
    olist/                    Brazilian e-commerce orders/sellers (~100 K orders)
    m5/                       Walmart daily sales (30 K SKUs, 5 years)
    dataco/                   Supply chain event records
    scms/                     SCMS supplier delivery history
    uci_retail/               UK e-commerce transactions
    rossmann/                 M4 daily retail sales
  personalization/
    retailrocket/             Session events (view/cart/purchase)
    instacart/                Grocery basket analysis
  dialogue/
    simmc2/                   Multimodal shopping dialogues
    durecdial/                Goal-driven recommendation dialogues
    redial/                   Conversational movie recommendation
  mgshop_dial/                Multi-goal shopping dialogues
  qdrant_storage/             Qdrant data (rufus_products + rufus_clip)
  rufus_sc.db                 Supply chain SQLite (inventory, demand, forecasts, suppliers)
  rufus_reviews.db            Amazon Reviews metadata (5.5 M rows, price + ratings)
```

## Setup

```bash
# 1. Install uv
curl -Ls https://astral.sh/uv/install.sh | sh   # Linux/Mac
# or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install dependencies (PyTorch cu128 pulled automatically)
uv sync

# 3. Pull the LLM
ollama pull qwen3:1.7b
```

## Quickstart

### 1 — Download data

```bash
# Week 1 essentials (ESCI, SQID, Amazon Reviews Electronics, MGShopDial)
uv run python scripts/download_datasets.py all

# Full-scale download — all datasets across all groups
uv run python scripts/download_all_data.py all

# Individual groups
uv run python scripts/download_all_data.py catalog       # Amazon Reviews all categories
uv run python scripts/download_all_data.py supply        # Olist, M5, DataCo, SCMS, UCI
uv run python scripts/download_all_data.py personalization  # RetailRocket, Instacart
uv run python scripts/download_all_data.py dialogue      # SIMMC2, DuRecDial, ReDial

# Check what's downloaded
uv run python scripts/download_all_data.py status
```

### 2 — Ingest into Qdrant

```bash
# Start Qdrant first (must be running before ingest or app)
.\scripts\start_qdrant.ps1          # Windows PowerShell
# or: docker compose up -d qdrant   # if Docker available

# Full ESCI catalog (~1.2 M products, ~20 min on GPU)
uv run python scripts/ingest_esci.py

# Quick smoke-test
uv run python scripts/ingest_esci.py --limit 5000

# CLIP image vectors (optional, ~3 min)
uv run python scripts/ingest_clip.py
```

### 3 — Ingest supply chain + reviews

```bash
# Olist → inventory, demand history, suppliers (rufus_sc.db)
uv run python scripts/ingest_olist.py

# SCMS → supplier lead times (appends to rufus_sc.db)
uv run python scripts/ingest_scms.py

# Amazon Reviews all categories → price, ratings, features (rufus_reviews.db)
uv run python scripts/ingest_amazon_reviews_full.py

# Train demand forecasts (M5 + Olist → forecasts table)
uv run python scripts/train_demand_forecast.py
```

### 4 — Start the server

```bash
# Terminal 1 — Qdrant (keep running)
.\scripts\start_qdrant.ps1

# Terminal 2 — FastAPI AG-UI server
uv run uvicorn server:app --reload
```

Open **http://localhost:8000** — the chat UI loads automatically.

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
| `followup` | "does it come in black?", "the second one" | reuse products → generate |
| `qa` | "what's the battery life?" | seller Q&A template or retrieve → generate |
| `compare` | "compare the first two", "which is better" | retrieve → generate |
| `gift_search` | "gift for my dad", "present for my wife" | retrieve → generate (gift framing) |
| `add_to_cart` | "add to cart", "I'll take it" | cart node |
| `view_cart` | "show my cart", "cart total" | cart node |
| `chitchat` | "hello", "thanks" | chitchat node |
| `check_stock` | "is the Anker charger in stock?" | inventory lookup → sc_generate |
| `reorder_alert` | "what needs to be reordered?" | bulk_reorder_check → sc_generate |
| `demand_forecast` | "forecast demand for electronics" | forecast_sku → sc_generate |
| `supplier_query` | "who are our suppliers?" | supplier lookup → sc_generate |

## Models

| Role | Model | Notes |
|---|---|---|
| LLM | `qwen3:1.7b` | Generation + intent (when LLM path taken). think=False kwarg, temperature=0, keep_alive=60m |
| Embedding | `BAAI/bge-m3` | 1024-dim fp16, 1.8 ms/query on RTX 5090 |
| CLIP | `openai/clip-vit-large-patch14` | 768-dim fp16, 3.8 ms/encode |
| Reranker | `BAAI/bge-reranker-v2-m3` | fp16, batch=128, 30 ms for 40 candidates |

**Important:** `think=False` must be passed as a **direct kwarg** to `ollama.chat()`, not inside the `options` dict. Passing it inside `options` routes all output to the `thinking` field and leaves `content` empty.

## Datasets

| Dataset | Location | Rows / Size | Used for |
|---|---|---|---|
| Amazon ESCI | `data/esci/` | 1.2 M products | Catalog, retrieval, eval |
| SQID CLIP vectors | `data/sqid/` | 164 K products | Image search |
| Amazon Reviews (Electronics) | `data/amazon_reviews/` | 1.6 M products | RAG enrichment (week 1) |
| Amazon Reviews (all 33 categories) | `data/amazon_reviews_full/` | 5.5 M ASINs | Price, ratings, features |
| Amazon-C4 Q&A | `data/amazon_c4/` | 1 M+ pairs | Q&A grounding |
| MGShopDial | `data/mgshop_dial/` | 64 dialogues | Intent few-shots |
| Olist E-Commerce | `data/supply_chain/olist/` | 100 K orders | Inventory, demand, suppliers |
| M5 Forecasting | `data/supply_chain/m5/` | 30 K SKUs × 1 969 days | Demand history |
| DataCo Supply Chain | `data/supply_chain/dataco/` | 180 K records | Supply chain events |
| SCMS Delivery History | `data/supply_chain/scms/` | 10 K shipments | Supplier lead times |
| UCI Online Retail II | `data/supply_chain/uci_retail/` | 1 M transactions | Demand patterns |
| RetailRocket | `data/personalization/retailrocket/` | 4.7 M events | Session modelling |
| Instacart | `data/personalization/instacart/` | 3.4 M orders | Basket analysis |
| SIMMC 2.1 | `data/dialogue/simmc2/` | 11 K dialogues | Intent training |
| DuRecDial 2.0 | `data/dialogue/durecdial/` | 16 K dialogues | Goal-driven rec |
| ReDial | `data/dialogue/redial/` | 10 K dialogues | Conversational rec |

## Hardware optimization notes

All models run fp16 on CUDA. Applied via `rufus/hardware.py` at import time:

```python
torch.set_float32_matmul_precision("high")  # TF32 tensor cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True        # auto-selects fastest kernels
```

Verified timings on RTX 5090:
- BGE-M3 embed: **1.8 ms**
- CLIP encode: **3.8 ms**
- Rerank 40 items: **30 ms** (fp16, batch=128)
- LLM TTFT (warm, pinned in VRAM): **~300 ms**
- Full query pipeline: **~1.5 s** (LLM dominates)

## Build roadmap

- **Week 1 ✅** Catalog RAG — ESCI → BGE-M3 → Qdrant → Qwen3 Q&A
- **Week 2 ✅** Conversational layer — LangGraph, 5-intent classifier, MG-ShopDial few-shots
- **Week 3 ✅** Multimodal retrieval — SQID CLIP vectors, dual-encoder, RRF fusion
- **Week 4 ✅** Evaluation — ESCI NDCG@K benchmark, BGE-M3 vs fusion comparison
- **Week 5 ✅** Production reliability — circuit breaker, retry, caching, telemetry, cross-encoder reranker
- **Week 6 ✅** AG-UI server + streaming frontend, supply chain layer, mock features, hardware optimisation

**Remaining / in progress:**
- Demand forecast model training at scale (statsforecast AutoETS on full M5)
- Personalization from real clickstream (RetailRocket + Instacart ingestion)

**Completed this session:**
- Review summarization — `reviews.py` now uses `rufus_reviews.db` (5.5M ASINs, 33 categories); `_format_context()` includes top helpful review snippet per product
- Image search — base64 image extracted from AG-UI messages; routed to `CLIPRetriever.retrieve_by_image()` via RRF fusion
- Price/rating display — product cards now show `$XX.XX` and `⭐ X.X (N)` from reviews DB
- DataCo ingestion — `scripts/ingest_dataco.py` builds inventory + demand history from 180K supply chain records

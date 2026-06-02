"""
Rufus — local Amazon shopping assistant (Streamlit UI)

Streaming architecture
----------------------
1. classify  (qwen3:1.7b, ~100 ms warm)  → intent + query shown on right panel
2. retrieve  (BGE-M3 + CLIP, ~500 ms warm) → product cards rendered immediately
3. generate  (qwen3.5 streamed)           → tokens appear as they're produced

Run:
  .\\scripts\\start_qdrant.ps1   # terminal 1 — keep running
  PYTHONUTF8=1 uv run streamlit run app.py  # terminal 2
"""
from __future__ import annotations

import io
import time
import uuid

import ollama
import streamlit as st

st.set_page_config(page_title="Rufus", page_icon="🛍️", layout="wide")

# ── Cached model singletons (loaded once per process) ──────────────────────

@st.cache_resource(show_spinner="Loading BGE-M3 onto GPU…")
def _retriever():
    from rufus.graph import _get_retriever
    r = _get_retriever()
    r.model
    return r


@st.cache_resource(show_spinner="Loading CLIP model…")
def _clip():
    from rufus.clip_retriever import CLIPRetriever
    return CLIPRetriever()


@st.cache_resource(show_spinner="Loading cross-encoder reranker (compiling CUDA kernels)…")
def _reranker():
    from rufus.reranker import ProductReranker
    r = ProductReranker()
    # Warm up CUDA kernels now so first real query isn't slow
    r.model.predict([("warmup query", "warmup product title")], show_progress_bar=False)
    return r


@st.cache_resource(show_spinner="Loading Amazon Reviews metadata…")
def _reviews():
    from rufus.reviews import get_meta, _load
    _load()   # pre-load all 10 parquet shards into module-level cache
    return True


# ── Session init ───────────────────────────────────────────────────────────

_reviews()   # pre-load metadata before first query

if "sid"     not in st.session_state: st.session_state.sid     = str(uuid.uuid4())
if "history" not in st.session_state: st.session_state.history = []
if "up_key"  not in st.session_state: st.session_state.up_key  = 0
if "last"    not in st.session_state: st.session_state.last    = {}  # last query meta

# ── Helpers ────────────────────────────────────────────────────────────────

BADGE_COLOR = {
    "search":   "#1f77b4",
    "followup": "#ff7f0e",
    "qa":       "#2ca02c",
    "compare":  "#9467bd",
    "chitchat": "#7f7f7f",
    "image":    "#d62728",
}

PLACEHOLDER_IMG = "https://placehold.co/160x160?text=No+Image"


def badge(label: str, color: str = "#888") -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:.75rem;font-weight:700">{label}</span>'
    )


def product_grid(products: list, n: int = 5) -> None:
    if not products:
        return
    cols = st.columns(min(len(products), n))
    for col, p in zip(cols, products[:n]):
        with col:
            st.image(p.image_url or PLACEHOLDER_IMG, use_container_width=True)
            title = p.title[:55] + ("…" if len(p.title) > 55 else "")
            st.markdown(f"**{title}**")
            if p.brand: st.caption(f"🏷 {p.brand}")
            if p.color: st.caption(f"🎨 {p.color}")
            st.caption(f"score {p.score:.3f}")


def render_history(col) -> None:
    for m in st.session_state.history:
        with col:
            with st.chat_message(m["role"]):
                if m.get("img"):
                    st.image(m["img"], width=140)
                if m.get("intent"):
                    color = BADGE_COLOR.get(m["intent"], "#888")
                    st.markdown(badge(m["intent"], color), unsafe_allow_html=True)
                if m.get("products"):
                    product_grid(m["products"])
                    st.markdown("---")
                st.markdown(m["text"])


def _retrieve(query: str, intent: str, top_k: int) -> list:
    from rufus.fusion import rrf_fuse
    # Large pool: reranker needs enough candidates to pick from
    pool = max(top_k * 8, 40)

    text_hits = _retriever().retrieve(query, top_k=pool)
    clip = _clip()
    if intent in ("search", "qa", "compare") and clip.available():
        clip_hits = clip.retrieve(query, top_k=pool)
        candidates = rrf_fuse([text_hits, clip_hits], top_k=pool)
    else:
        candidates = text_hits

    # Cross-encoder rerank — pass full pool, return top_k * 2 for image filtering
    reranked = _reranker().rerank(query, candidates, top_k=min(top_k * 2, len(candidates)))

    # Keep only products with real image URLs
    with_image = [p for p in reranked if p.image_url]
    return with_image[:top_k]


def _build_history(raw_history: list) -> list[dict]:
    """
    Convert session history to LLM message format, preserving full multi-turn
    context.  Each assistant turn includes the product context that was shown,
    so follow-up questions ("which one is cheaper?") have the right grounding.
    """
    msgs = []
    for m in raw_history:
        if m["role"] == "user":
            msgs.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant" and m.get("text"):
            msgs.append({"role": "assistant", "content": m["text"]})
    return msgs[-12:]   # keep last 6 turns (12 messages) to stay within context


def _stream_answer(model: str, products: list, user_text: str, history: list):
    """Yield tokens; uses OllamaClient for circuit breaker + retry."""
    from rufus.llm import OllamaClient
    from rufus.rag import SYSTEM_PROMPT, _format_context
    context = _format_context(products) if products else "No products found."
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"Retrieved products:\n{context}\n\nCustomer question: {user_text}"},
    ]
    stream = OllamaClient(model=model).chat(msgs, stream=True)
    if stream is None:
        yield "⚠️ LLM unavailable — showing products above. Please try again shortly."
        return
    yield from stream


def _stream_chitchat(model: str, user_text: str, history: list):
    from rufus.llm import OllamaClient
    from rufus.rag import SYSTEM_PROMPT
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
            {"role": "user", "content": user_text}]
    stream = OllamaClient(model=model).chat(msgs, stream=True)
    if stream is None:
        yield "⚠️ LLM unavailable. Please try again shortly."
        return
    yield from stream


# ── Layout ─────────────────────────────────────────────────────────────────

chat_col, info_col = st.columns([7, 3])

# ── Right panel (static legend + live query details) ───────────────────────

with info_col:
    st.markdown("### Query Details")
    intent_slot  = st.empty()
    query_slot   = st.empty()
    filter_slot  = st.empty()
    timing_slot  = st.empty()

    # Pre-fill with last query if available
    last = st.session_state.last
    if last.get("intent"):
        c = BADGE_COLOR.get(last["intent"], "#888")
        intent_slot.markdown(badge(last["intent"], c), unsafe_allow_html=True)
    if last.get("query"):
        query_slot.caption(f'Query: *"{last["query"]}"*')
    if last.get("filters"):
        active = {k: v for k, v in last["filters"].items() if v}
        if active:
            filter_slot.json(active)
    if last.get("ms"):
        timing_slot.caption(
            f"classify {last['ms']['classify']} ms · "
            f"retrieve {last['ms']['retrieve']} ms · "
            f"ttft {last['ms']['ttft']} ms"
        )

    st.divider()
    st.markdown("**Intent guide**")
    guide = [
        ("search",   "Find products in catalog"),
        ("followup", "About products already shown"),
        ("qa",       "Specific product question"),
        ("compare",  "Compare two or more items"),
        ("chitchat", "Greetings / general chat"),
        ("image",    "Visual similarity search"),
    ]
    for intent, desc in guide:
        c = BADGE_COLOR[intent]
        st.markdown(f"{badge(intent, c)}&nbsp; {desc}", unsafe_allow_html=True)

    st.divider()
    st.caption(f"Session `{st.session_state.sid[:12]}…`")

# ── Sidebar (settings) ─────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛍️ Rufus")
    model  = st.selectbox("Model", ["qwen3.5:latest", "qwen3.5:cloud", "glm-5:cloud"])
    top_k  = st.slider("Results", 3, 10, 5)
    st.divider()
    if st.button("🔄 New session", use_container_width=True):
        st.session_state.sid     = str(uuid.uuid4())
        st.session_state.history = []
        st.session_state.last    = {}
        st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────

render_history(chat_col)

# ── Input ──────────────────────────────────────────────────────────────────

with chat_col:
    uploaded = st.file_uploader(
        "📎 Attach an image (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"up_{st.session_state.up_key}",
    )
    if uploaded:
        c1, c2 = st.columns([1, 5])
        c1.image(uploaded, width=72)
        c2.caption(f"**{uploaded.name}** — add text below or press Enter to search by image.")

    prompt = st.chat_input("What are you looking for?")

if not prompt and not uploaded:
    st.stop()

img_bytes = uploaded.getvalue() if uploaded else None
text      = prompt or ""

# ── User message ───────────────────────────────────────────────────────────

with chat_col:
    with st.chat_message("user"):
        if img_bytes: st.image(img_bytes, width=140)
        st.markdown(text if text else "_image search_")

st.session_state.history.append({
    "role": "user", "text": text or "_image search_", "img": img_bytes,
})

# Build full multi-turn conversation history (includes both user and assistant)
llm_history = _build_history(st.session_state.history[:-1])

# ── Assistant turn ─────────────────────────────────────────────────────────

ms = {}   # timing dict

with chat_col:
    with st.chat_message("assistant"):

        if img_bytes:
            # ── Image path ─────────────────────────────────────────────────
            clip = _clip()
            intent = "image"
            color  = BADGE_COLOR["image"]
            intent_slot.markdown(badge("image search", color), unsafe_allow_html=True)
            if text:
                query_slot.caption(f'Refinement: *"{text}"*')

            if not clip.available():
                answer = "Image search not ready — run `uv run python scripts/ingest_clip.py` first."
                products = []
                st.markdown(answer)
            else:
                with st.spinner("Encoding image…"):
                    pil = __import__("PIL.Image", fromlist=["Image"]).open(
                        io.BytesIO(img_bytes)
                    ).convert("RGB")
                    if text:
                        import numpy as np
                        from rufus.qdrant import get_client
                        iv = np.array(clip.encode_image(pil))
                        tv = np.array(clip.encode_text(text))
                        cv = iv + tv
                        cv = (cv / np.linalg.norm(cv)).tolist()
                        hits = get_client().query_points(
                            collection_name="rufus_clip", query=cv, limit=top_k
                        )
                        products = clip._hits_to_products(hits)
                    else:
                        products = clip.retrieve_by_image(pil, top_k=top_k)

                st.markdown(badge("image search", color) +
                            (f"  {badge('+ text', '#555')}" if text else ""),
                            unsafe_allow_html=True)
                product_grid(products)
                if products: st.markdown("---")

                t_gen = time.perf_counter()
                answer = st.write_stream(_stream_answer(model, products, text or "What are these?", llm_history))
                ms["ttft"] = int((time.perf_counter() - t_gen) * 1000)

        else:
            # ── Text path ──────────────────────────────────────────────────
            products = []

            # Step 1: Classify (fast — qwen3:1.7b)
            t0 = time.perf_counter()
            with st.spinner("Classifying…"):
                from rufus.intent import classify
                clf = classify(text, llm_history)
            intent  = clf["intent"]
            query   = clf.get("query") or text
            filters = clf.get("filters") or {}
            ms["classify"] = int((time.perf_counter() - t0) * 1000)

            # Update right panel immediately
            color = BADGE_COLOR.get(intent, "#888")
            intent_slot.markdown(badge(intent, color), unsafe_allow_html=True)
            query_slot.caption(f'Query: *"{query}"*')
            active_filters = {k: v for k, v in filters.items() if v}
            if active_filters:
                filter_slot.json(active_filters)
            else:
                filter_slot.empty()

            st.markdown(badge(intent, color), unsafe_allow_html=True)

            # Step 2: Retrieve (BGE-M3 + CLIP, fast after warm-up)
            if intent in ("search", "qa", "compare"):
                t1 = time.perf_counter()
                with st.spinner("Retrieving products…"):
                    products = _retrieve(query, intent, top_k)
                ms["retrieve"] = int((time.perf_counter() - t1) * 1000)
                product_grid(products)
                if products: st.markdown("---")

            elif intent == "followup":
                # reuse products from previous assistant turn (already image-filtered)
                for m in reversed(st.session_state.history[:-1]):
                    if m["role"] == "assistant" and m.get("products"):
                        products = [p for p in m["products"] if p.image_url]
                        break
                if products:
                    product_grid(products)
                    st.markdown("---")

            # Step 3: Stream LLM generation (tokens appear immediately)
            t2 = time.perf_counter()
            if intent == "chitchat":
                answer = st.write_stream(_stream_chitchat(model, text, llm_history))
            else:
                answer = st.write_stream(_stream_answer(model, products, text, llm_history))
            ms["ttft"] = int((time.perf_counter() - t2) * 1000)

        # Update timing on right panel + emit structured log
        from rufus.telemetry import log_query, metrics as _metrics
        from rufus.llm import OllamaClient as _OllamaClient
        from rufus.cache import embedding_cache, rerank_cache

        timing_slot.caption(
            "⏱ " +
            (f"classify {ms.get('classify','—')} ms · " if "classify" in ms else "") +
            (f"retrieve {ms.get('retrieve','—')} ms · " if "retrieve" in ms else "") +
            f"first token {ms.get('ttft','—')} ms"
            + (f"  |  cache {embedding_cache.hit_rate:.0%}" if embedding_cache.total_queries > 0 else "")
        )

        log_query(
            session_id=st.session_state.sid,
            intent=intent,
            timings=ms,
            n_products=len(products),
            model=model,
        )
        _metrics.record(sum(ms.values()), error=False)

# ── Persist ────────────────────────────────────────────────────────────────

st.session_state.history.append({
    "role": "assistant", "text": answer,
    "intent": intent, "products": products,
})
st.session_state.last = {
    "intent": intent,
    "query": query if "query" in dir() else "",
    "filters": filters if "filters" in dir() else {},
    "ms": ms,
}

if uploaded:
    st.session_state.up_key += 1
    st.rerun()

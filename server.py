"""
Rufus AG-UI Server — FastAPI backend implementing the Agent-User Interaction Protocol.

Each query runs as a pipeline and emits AG-UI events over Server-Sent Events:

  RunStarted
  ├─ StepStarted("classify")
  │    CustomEvent("intent", {intent, query, filters})
  │  StepFinished("classify")
  ├─ StepStarted("retrieve")          [search / qa / compare only]
  │    CustomEvent("products", [...])
  │  StepFinished("retrieve")
  ├─ StepStarted("generate")
  │    TextMessageStart
  │    TextMessageContent × N         [streaming tokens]
  │    TextMessageEnd
  │  StepFinished("generate")
  RunFinished

Serve with:
  .\\scripts\\start_qdrant.ps1        # terminal 1
  uv run uvicorn server:app --reload  # terminal 2
  open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path

from ag_ui.core.events import (
    CustomEvent,
    RunAgentInput,
    RunFinishedEvent,
    RunStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder
import rufus.hardware  # apply TF32 + cuDNN benchmark at import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Rufus AG-UI Server")


@app.on_event("startup")
async def _warmup():
    """Pre-load models into GPU on startup so the first request has no cold-start delay."""
    import asyncio, ollama as _ollama
    loop = asyncio.get_event_loop()
    def _load():
        try:
            # Ping both models with keep_alive=60m — loads them into VRAM now
            _ollama.chat(model="qwen3:1.7b",
                         messages=[{"role":"user","content":"hi"}],
                         options={"num_predict": 1, "temperature": 0},
                         keep_alive="60m")
            print("[startup] qwen3:1.7b loaded into GPU")
        except Exception as e:
            print(f"[startup] warmup failed: {e}")
    await loop.run_in_executor(None, _load)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_encoder = EventEncoder()
_FRONTEND = Path(__file__).parent / "frontend"


# ── Lazy model singletons (loaded once per server process) ─────────────────

_retriever_instance = None
_clip_instance = None
_reranker_instance = None


def _get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        from rufus.retriever import ProductRetriever
        _retriever_instance = ProductRetriever()
        _ = _retriever_instance.model   # warm BGE-M3
    return _retriever_instance


def _get_clip():
    global _clip_instance
    if _clip_instance is None:
        from rufus.clip_retriever import CLIPRetriever
        _clip_instance = CLIPRetriever()
    return _clip_instance


def _get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        from rufus.reranker import ProductReranker
        _reranker_instance = ProductReranker()
        _reranker_instance.model.predict(
            [("warmup", "warmup")], show_progress_bar=False
        )
    return _reranker_instance


def _retrieve(
    query: str,
    intent: str,
    top_k: int = 5,
    image_data: str | None = None,
    session_id: str = "",
) -> list:
    from rufus.fusion import rrf_fuse
    pool = max(top_k * 8, 40)
    text_hits = _get_retriever().retrieve(query, top_k=pool)
    clip = _get_clip()

    ranker_lists = [text_hits]

    if image_data and clip.available():
        import base64, io
        from PIL import Image as _Image
        _, b64 = image_data.split(",", 1)
        pil_img = _Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        ranker_lists.append(clip.retrieve_by_image(pil_img, top_k=pool))
    elif intent in ("search", "qa", "compare") and clip.available():
        ranker_lists.append(clip.retrieve(query, top_k=pool))

    # Blend in session-based similar-to-viewed results (personalization signal)
    if session_id and intent in ("search", "followup", "qa"):
        try:
            from rufus.personalization import get_similar_to_viewed
            viewed_hits = get_similar_to_viewed(session_id, top_k=pool // 4)
            if viewed_hits:
                ranker_lists.append(viewed_hits)
        except Exception:
            pass

    candidates = rrf_fuse(ranker_lists, top_k=pool) if len(ranker_lists) > 1 else text_hits
    reranked = _get_reranker().rerank(query, candidates, top_k=top_k * 2)
    return reranked[:top_k]


def _products_to_json(products: list) -> list[dict]:
    from rufus.images import get_image_urls_batch
    from rufus.reviews import get_meta_batch
    ids      = [p.product_id for p in products]
    meta_map = get_meta_batch(ids)
    # Fill image URLs: prefer what's already on the Product (CLIP collection),
    # fall back to the image lookup DB for BGE-M3-only results.
    missing  = [p.product_id for p in products if not p.image_url]
    img_map  = get_image_urls_batch(missing) if missing else {}
    result   = []
    for p in products:
        meta = meta_map.get(p.product_id) or {}
        result.append({
            "product_id":   p.product_id,
            "title":        p.title,
            "brand":        p.brand,
            "color":        p.color,
            "bullet_point": p.bullet_point,
            "score":        round(p.score, 4),
            "image_url":    p.image_url or img_map.get(p.product_id),
            "price":        meta.get("price"),
            "avg_rating":   meta.get("avg_rating"),
            "rating_count": meta.get("rating_count"),
        })
    return result


def _extract_message(messages: list) -> tuple[str, list[dict], str | None]:
    """Return (last_user_text, llm_history, image_data_url) from AG-UI messages.

    image_data_url is the last base64 data: URL found in the most recent user
    message, or None if no image was attached.
    """
    history: list[dict] = []
    text = ""
    image_data_url: str | None = None
    for m in messages:
        role = getattr(m, "role", None) or m.get("role", "user")
        content = getattr(m, "content", None) or m.get("content", "")
        if isinstance(content, list):
            text_parts: list[str] = []
            msg_img: str | None = None
            for part in content:
                if hasattr(part, "text"):
                    text_parts.append(part.text)
                elif isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:image"):
                        msg_img = url
                elif hasattr(part, "image_url"):
                    url = getattr(getattr(part, "image_url", None), "url", "") or ""
                    if url.startswith("data:image"):
                        msg_img = url
            content = " ".join(text_parts)
            if role == "user" and msg_img:
                image_data_url = msg_img
        if role == "user":
            text = str(content)
        history.append({"role": role, "content": str(content)})
    return text, history[:-1], image_data_url


def _run_pipeline(input_data: RunAgentInput, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    """Run the full Rufus pipeline synchronously, pushing AG-UI SSE strings into queue."""

    def emit(event) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, _encoder.encode(event))

    def done() -> None:
        loop.call_soon_threadsafe(queue.put_nowait, None)

    run_id  = str(uuid.uuid4())
    msg_id  = str(uuid.uuid4())
    thread_id = input_data.thread_id or str(uuid.uuid4())

    try:
        emit(RunStartedEvent(thread_id=thread_id, run_id=run_id))

        text, history, image_data = _extract_message(input_data.messages or [])
        if not text:
            emit(RunFinishedEvent(thread_id=thread_id, run_id=run_id))
            return

        # ── Classify ───────────────────────────────────────────────────────
        emit(StepStartedEvent(step_name="classify"))
        from rufus.intent import classify
        if image_data:
            # Image queries bypass the LLM classifier — always visual search
            intent  = "search"
            query   = text or "find visually similar products"
            filters = {}
            emit(CustomEvent(name="intent", value={"intent": "image", "query": query, "filters": filters}))
        else:
            clf     = classify(text, history)
            intent  = clf["intent"]
            query   = clf.get("query") or text
            filters = clf.get("filters") or {}
            emit(CustomEvent(name="intent", value={"intent": intent, "query": query, "filters": filters}))
        emit(StepFinishedEvent(step_name="classify"))

        # ── Personalization preference update ─────────────────────────────
        session_id = input_data.thread_id or thread_id
        try:
            from rufus.personalization import update_profile
        except Exception:
            update_profile = None

        # ── Retrieve ───────────────────────────────────────────────────────
        products = []
        if intent in ("search", "qa", "compare", "gift_search") or image_data:
            emit(StepStartedEvent(step_name="retrieve"))
            products = _retrieve(query, intent, image_data=image_data, session_id=session_id)
            if update_profile:
                try:
                    update_profile(session_id, products)
                except Exception:
                    pass
            emit(CustomEvent(name="products", value=_products_to_json(products)))
            emit(StepFinishedEvent(step_name="retrieve"))
        elif intent == "followup":
            # reuse products from state if sent by client
            state = input_data.state or {}
            raw = state.get("products", [])
            if raw:
                emit(CustomEvent(name="products", value=raw))

        # ── Generate (streaming) ───────────────────────────────────────────
        emit(StepStartedEvent(step_name="generate"))
        emit(TextMessageStartEvent(message_id=msg_id))

        from rufus.llm import OllamaClient
        from rufus.rag import SYSTEM_PROMPT, _format_context
        context = _format_context(products) if products else "No products found."
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history[-8:],
            {"role": "user", "content": f"Retrieved products:\n{context}\n\nCustomer question: {text}"},
        ]
        if intent == "chitchat":
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *history[-8:],
                    {"role": "user", "content": text}]

        stream = OllamaClient().chat(msgs, stream=True)
        token_count = 0
        if stream:
            for token in stream:
                emit(TextMessageContentEvent(message_id=msg_id, delta=token))
                token_count += 1
        if token_count == 0:
            print("[server] LLM returned no tokens — Ollama may not be running")
            emit(TextMessageContentEvent(
                message_id=msg_id,
                delta="⚠️ LLM unavailable — make sure Ollama is running (`ollama serve`).",
            ))

        emit(TextMessageEndEvent(message_id=msg_id))
        emit(StepFinishedEvent(step_name="generate"))
        emit(RunFinishedEvent(thread_id=thread_id, run_id=run_id))

    except Exception as exc:
        from ag_ui.core.events import RunErrorEvent
        emit(RunErrorEvent(message=str(exc)))
    finally:
        done()


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/awp")
async def agent_endpoint(body: RunAgentInput):
    """AG-UI SSE endpoint — runs the Rufus pipeline and streams events."""
    queue: asyncio.Queue = asyncio.Queue()
    loop  = asyncio.get_event_loop()

    thread = threading.Thread(
        target=_run_pipeline, args=(body, queue, loop), daemon=True
    )
    thread.start()

    async def event_stream():
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    from rufus.qdrant import get_client
    try:
        collections = [c.name for c in get_client().get_collections().collections]
        return {"status": "ok", "collections": collections}
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}


# ── Serve frontend ─────────────────────────────────────────────────────────

if _FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")

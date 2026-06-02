"""
LangGraph conversation graph for Rufus.

Turn flow
---------
  user message
      ↓
  classify  ← intent + clean query extracted by qwen3.5
      ↓ (route on intent)
  ┌─────────────────────────────────────────────────────┐
  │ search / qa / compare → retrieve → generate         │
  │   retrieve = BGE-M3 top_k*4 + CLIP top_k*4 → RRF   │
  │ followup              → generate (reuse products)   │
  │ chitchat              → chitchat (no retrieval)     │
  └─────────────────────────────────────────────────────┘
      ↓
    END

Config keys (passed via config["configurable"]):
  thread_id : str   — session ID for MemorySaver
  model     : str   — Ollama model for generation (default: qwen3.5:latest)
  top_k     : int   — number of products to surface (default: 5)
"""

from __future__ import annotations

import time

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from rufus.clip_retriever import CLIPRetriever
from rufus.fusion import rrf_fuse
from rufus.intent import classify
from rufus.llm import OllamaClient
from rufus.rag import SYSTEM_PROMPT, _format_context
from rufus.reranker import ProductReranker
from rufus.retriever import ProductRetriever
from rufus.state import ShoppingState

# ── Lazy singletons ─────────────────────────────────────────────────────────

_retriever: ProductRetriever | None = None
_clip_retriever: CLIPRetriever | None = None
_reranker: ProductReranker | None = None
_llm: OllamaClient | None = None
_llm_model: str = ""


def _get_retriever() -> ProductRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ProductRetriever()
    return _retriever


def _get_clip_retriever() -> CLIPRetriever:
    global _clip_retriever
    if _clip_retriever is None:
        _clip_retriever = CLIPRetriever()
    return _clip_retriever


def _get_reranker() -> ProductReranker:
    global _reranker
    if _reranker is None:
        _reranker = ProductReranker()
    return _reranker


def _get_llm(model: str) -> OllamaClient:
    global _llm, _llm_model
    if _llm is None or _llm_model != model:
        _llm = OllamaClient(model=model)
        _llm_model = model
    return _llm


def _cfg(config: RunnableConfig) -> dict:
    return config.get("configurable", {})


# ── Nodes ────────────────────────────────────────────────────────────────────

def node_classify(state: ShoppingState, config: RunnableConfig) -> dict:
    t0 = time.perf_counter()
    last = state["messages"][-1]["content"]
    history = state.get("messages", [])[:-1]
    result = classify(last, history)
    print(f"[classify] {time.perf_counter()-t0:.2f}s  intent={result['intent']}")
    return {
        "intent": result["intent"],
        "query":  result.get("query") or last,
        "filters": result.get("filters") or {},
    }


def node_retrieve(state: ShoppingState, config: RunnableConfig) -> dict:
    t0    = time.perf_counter()
    query = state["query"]
    top_k = int(_cfg(config).get("top_k", 5))
    pool  = max(top_k * 8, 40)

    # BGE-M3
    t1 = time.perf_counter()
    text_results = _get_retriever().retrieve(query, top_k=pool)
    print(f"[retrieve] bge-m3: {time.perf_counter()-t1:.2f}s  ({len(text_results)} hits)")

    # CLIP — only for search/compare/qa
    clip = _get_clip_retriever()
    intent = state.get("intent", "search")
    if intent in ("search", "qa", "compare") and clip.available():
        t2 = time.perf_counter()
        clip_results = clip.retrieve(query, top_k=pool)
        print(f"[retrieve] clip:   {time.perf_counter()-t2:.2f}s  ({len(clip_results)} hits)")
        candidates = rrf_fuse([text_results, clip_results], top_k=pool)
    else:
        candidates = text_results

    # Cross-encoder rerank
    products = _get_reranker().rerank(query, candidates, top_k=top_k)
    print(f"[retrieve] total:  {time.perf_counter()-t0:.2f}s  ({len(products)} products after rerank)")
    return {"products": products}


def node_generate(state: ShoppingState, config: RunnableConfig) -> dict:
    t0      = time.perf_counter()
    c       = _cfg(config)
    model   = c.get("model", "qwen3.5:latest")
    products = state.get("products") or []
    context  = _format_context(products) if products else "No matching products found."
    user_q   = state["messages"][-1]["content"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *state["messages"][:-1][-8:],
        {"role": "user", "content": f"Retrieved products:\n{context}\n\nCustomer question: {user_q}"},
    ]
    answer = _get_llm(model).chat(messages)
    print(f"[generate] {time.perf_counter()-t0:.2f}s  model={model}")
    return {"messages": [{"role": "assistant", "content": answer}]}


def node_followup(state: ShoppingState, config: RunnableConfig) -> dict:
    return node_generate(state, config)


def node_chitchat(state: ShoppingState, config: RunnableConfig) -> dict:
    t0    = time.perf_counter()
    model = _cfg(config).get("model", "qwen3.5:latest")
    msgs  = [{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]]
    answer = _get_llm(model).chat(msgs)
    print(f"[chitchat] {time.perf_counter()-t0:.2f}s")
    return {"messages": [{"role": "assistant", "content": answer}]}


# ── Routing ──────────────────────────────────────────────────────────────────

def _route(state: ShoppingState) -> str:
    intent = state.get("intent", "search")
    if intent in ("search", "qa", "compare"):
        return "retrieve"
    if intent == "followup":
        return "followup"
    return "chitchat"


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ShoppingState)
    g.add_node("classify",  node_classify)
    g.add_node("retrieve",  node_retrieve)
    g.add_node("generate",  node_generate)
    g.add_node("followup",  node_followup)
    g.add_node("chitchat",  node_chitchat)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", _route, {
        "retrieve": "retrieve",
        "followup": "followup",
        "chitchat": "chitchat",
    })
    g.add_edge("retrieve", "generate")
    g.add_edge("generate",  END)
    g.add_edge("followup",  END)
    g.add_edge("chitchat",  END)

    return g.compile(checkpointer=MemorySaver())

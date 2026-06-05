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

    # CLIP — only for search/compare/qa/gift_search
    clip = _get_clip_retriever()
    intent = state.get("intent", "search")
    if intent in ("search", "qa", "compare", "gift_search") and clip.available():
        t2 = time.perf_counter()
        clip_results = clip.retrieve(query, top_k=pool)
        print(f"[retrieve] clip:   {time.perf_counter()-t2:.2f}s  ({len(clip_results)} hits)")
        candidates = rrf_fuse([text_results, clip_results], top_k=pool)
    else:
        candidates = text_results

    # Cross-encoder rerank
    products = _get_reranker().rerank(query, candidates, top_k=top_k)

    # Personalization bias — boost preferred brands/categories
    session_id = state.get("session_id", "")
    if session_id:
        try:
            from rufus.personalization import apply_preference_bias
            products = apply_preference_bias(products, session_id)
        except Exception:
            pass

    print(f"[retrieve] total:  {time.perf_counter()-t0:.2f}s  ({len(products)} products after rerank)")
    return {"products": products}


def node_generate(state: ShoppingState, config: RunnableConfig) -> dict:
    t0       = time.perf_counter()
    c        = _cfg(config)
    model    = c.get("model", "qwen3:1.7b")
    intent   = state.get("intent", "search")
    products = state.get("products") or []
    context  = _format_context(products) if products else "No matching products found."
    user_q   = state["messages"][-1]["content"]
    session_id = state.get("session_id", "")

    # Seller Q&A fallback — try template before LLM
    if intent == "qa" and products:
        from rufus.seller_qa import answer_question
        cat = getattr(products[0], "category", "") or ""
        qa_ans = answer_question(user_q, category=cat)
        if qa_ans:
            return {"messages": [{"role": "assistant", "content": qa_ans}]}

    # Personalization context suffix
    pref_ctx = ""
    if session_id:
        try:
            from rufus.personalization import preference_summary
            s = preference_summary(session_id)
            if s:
                pref_ctx = f"\nUser preferences: {s}"
        except Exception:
            pass

    # Gift framing
    gift_prefix = ""
    if intent == "gift_search":
        gift_prefix = "The user is looking for a gift. Frame your answer as gift recommendations.\n\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *state["messages"][:-1][-8:],
        {"role": "user", "content":
            f"{gift_prefix}Retrieved products:\n{context}{pref_ctx}\n\nCustomer question: {user_q}"},
    ]
    answer = _get_llm(model).chat(messages) or ""
    print(f"[generate] {time.perf_counter()-t0:.2f}s  model={model}  intent={intent}")
    return {"messages": [{"role": "assistant", "content": answer}]}


def node_followup(state: ShoppingState, config: RunnableConfig) -> dict:
    return node_generate(state, config)


def node_gift(state: ShoppingState, config: RunnableConfig) -> dict:
    """Gift/occasion search — retrieve products then generate gift-framed answer."""
    # Reuse retrieve then generate with gift-specific system prompt prefix
    result = node_retrieve(state, config)
    return result   # generate node handles the gift framing via intent


def node_cart(state: ShoppingState, config: RunnableConfig) -> dict:
    """Handle add_to_cart and view_cart intents."""
    from rufus.cart import add_item, format_cart
    intent    = state.get("intent", "view_cart")
    cart      = list(state.get("cart") or [])
    products  = state.get("products") or []
    model     = _cfg(config).get("model", "qwen3:1.7b")

    if intent == "add_to_cart" and products:
        # Add first product to cart
        p = products[0]
        item = {
            "product_id": getattr(p, "product_id", ""),
            "title":      getattr(p, "title", ""),
            "brand":      getattr(p, "brand", ""),
            "unit_price": getattr(p, "score", 0),   # use score as price proxy
        }
        cart = add_item(cart, item)
        answer = f"Added **{item['title'][:50]}** to your cart.\n\n{format_cart(cart)}"
    else:
        answer = format_cart(cart)

    return {
        "cart": cart,
        "messages": [{"role": "assistant", "content": answer}],
    }


def node_chitchat(state: ShoppingState, config: RunnableConfig) -> dict:
    t0    = time.perf_counter()
    model = _cfg(config).get("model", "qwen3.5:latest")
    msgs  = [{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]]
    answer = _get_llm(model).chat(msgs)
    print(f"[chitchat] {time.perf_counter()-t0:.2f}s")
    return {"messages": [{"role": "assistant", "content": answer}]}


# ── Supply chain nodes ────────────────────────────────────────────────────────

def node_supply_chain(state: ShoppingState, config: RunnableConfig) -> dict:
    """
    Handles all SC intents: check_stock, reorder_alert, demand_forecast,
    supplier_query, sc_analytics.
    Queries rufus_sc.db and formats context for the generate node.
    """
    t0     = time.perf_counter()
    intent = state.get("intent", "check_stock")
    query  = state.get("query") or ""

    try:
        from rufus.demand import (
            bulk_reorder_check,
            category_demand_summary,
            forecast_sku,
            inventory_health_report,
        )
        from rufus.inventory import (
            get_all_suppliers,
            get_low_stock,
            search_inventory,
        )
        from rufus.sc_rag import format_sc_context

        sc_items: list = []
        inventory_hits: list = []
        alerts: list = []
        forecast = None
        suppliers: list = []

        if intent == "check_stock":
            inventory_hits = search_inventory(query, limit=8)
            sc_items = inventory_hits

        elif intent == "reorder_alert":
            alerts = bulk_reorder_check()
            sc_items = [a.to_dict() for a in alerts[:20]]

        elif intent == "demand_forecast":
            hits = search_inventory(query, limit=1)
            if hits:
                forecast = forecast_sku(hits[0]["sku"])
                sc_items = [forecast.to_dict()] if forecast else []
            else:
                # category-level forecast
                summary = category_demand_summary(query)
                sc_items = summary.get("top_items", [])

        elif intent == "supplier_query":
            suppliers = get_all_suppliers(limit=10)
            sc_items = suppliers

        elif intent == "sc_analytics":
            if query:
                summary = category_demand_summary(query)
                sc_items = summary.get("top_items", [])
                inventory_hits = sc_items
            else:
                report = inventory_health_report()
                sc_items = report.get("top_critical", [])

        ctx = format_sc_context(
            intent,
            inventory=inventory_hits or None,
            alerts=alerts or None,
            forecast=forecast,
            suppliers=suppliers or None,
        )
        print(f"[sc] {time.perf_counter()-t0:.2f}s  intent={intent}  items={len(sc_items)}")
        return {"sc_items": sc_items, "sc_context": ctx}

    except Exception as exc:
        print(f"[sc] error: {exc}")
        return {"sc_items": [], "sc_context": f"Supply chain data unavailable: {exc}"}


def node_sc_generate(state: ShoppingState, config: RunnableConfig) -> dict:
    """Generate LLM answer for supply chain queries using sc_context."""
    t0    = time.perf_counter()
    model = _cfg(config).get("model", "qwen3.5:latest")
    from rufus.sc_rag import SC_SYSTEM_PROMPT
    ctx     = state.get("sc_context") or "No supply chain data."
    user_q  = state["messages"][-1]["content"]
    history = state["messages"][:-1][-6:]
    msgs = [
        {"role": "system", "content": SC_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"Supply chain data:\n{ctx}\n\nQuestion: {user_q}"},
    ]
    answer = _get_llm(model).chat(msgs)
    print(f"[sc_generate] {time.perf_counter()-t0:.2f}s")
    return {"messages": [{"role": "assistant", "content": answer}]}


# ── Routing ──────────────────────────────────────────────────────────────────

_SC_INTENTS   = frozenset({"check_stock", "reorder_alert", "demand_forecast",
                            "supplier_query", "sc_analytics"})
_CART_INTENTS = frozenset({"add_to_cart", "view_cart"})


def _route(state: ShoppingState) -> str:
    intent = state.get("intent", "search")
    if intent in _SC_INTENTS:
        return "supply_chain"
    if intent in _CART_INTENTS:
        return "cart"
    if intent == "gift_search":
        return "gift"
    if intent in ("search", "qa", "compare"):
        return "retrieve"
    if intent == "followup":
        return "followup"
    return "chitchat"


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ShoppingState)
    g.add_node("classify",     node_classify)
    g.add_node("retrieve",     node_retrieve)
    g.add_node("generate",     node_generate)
    g.add_node("followup",     node_followup)
    g.add_node("chitchat",     node_chitchat)
    g.add_node("gift",         node_gift)
    g.add_node("cart",         node_cart)
    g.add_node("supply_chain", node_supply_chain)
    g.add_node("sc_generate",  node_sc_generate)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", _route, {
        "retrieve":     "retrieve",
        "followup":     "followup",
        "chitchat":     "chitchat",
        "gift":         "gift",
        "cart":         "cart",
        "supply_chain": "supply_chain",
    })
    g.add_edge("retrieve",     "generate")
    g.add_edge("gift",         "generate")
    g.add_edge("generate",     END)
    g.add_edge("followup",     END)
    g.add_edge("chitchat",     END)
    g.add_edge("cart",         END)
    g.add_edge("supply_chain", "sc_generate")
    g.add_edge("sc_generate",  END)

    return g.compile(checkpointer=MemorySaver())

"""
Intent classifier for the Rufus conversational shopping assistant.

Uses qwen3.5 via Ollama to classify the user's latest message into one of
five intents and extract a clean search query + filter hints.

Intents
-------
search    – user wants to find/discover products
followup  – refers to products already shown ("that one", "does it come in blue")
qa        – specific feature/spec question about a shown product
compare   – wants to compare two or more products
chitchat  – greeting, thanks, or off-topic
"""

from __future__ import annotations

import json

import ollama

INTENTS = frozenset({
    "search", "followup", "qa", "compare", "chitchat",
    # supply chain intents
    "check_stock", "reorder_alert", "demand_forecast", "supplier_query", "sc_analytics",
})

SC_INTENTS = frozenset({
    "check_stock", "reorder_alert", "demand_forecast", "supplier_query", "sc_analytics",
})

# Few-shot examples derived from MG-ShopDial + supply chain patterns
_FEW_SHOT = """
Examples:
User: "I'm looking for a noise-cancelling headphone under $200"
-> {"intent":"search","query":"noise-cancelling headphones under 200 dollars","filters":{"price_max":200,"brand":null,"color":null,"category":"headphones"}}

User: "Do any of them come in white?"
-> {"intent":"followup","query":null,"filters":{"color":"white","brand":null,"price_max":null,"category":null}}

User: "What's the battery life on the second one?"
-> {"intent":"qa","query":null,"filters":{"brand":null,"color":null,"price_max":null,"category":null}}

User: "Which is better for gaming, the first or third?"
-> {"intent":"compare","query":null,"filters":{"brand":null,"color":null,"price_max":null,"category":null}}

User: "Thanks, that's all I needed!"
-> {"intent":"chitchat","query":null,"filters":{"brand":null,"color":null,"price_max":null,"category":null}}

User: "Is the Anker USB-C charger in stock?"
-> {"intent":"check_stock","query":"Anker USB-C charger","filters":{"brand":"Anker","color":null,"price_max":null,"category":"chargers"}}

User: "What needs to be reordered this week?"
-> {"intent":"reorder_alert","query":null,"filters":{"brand":null,"color":null,"price_max":null,"category":null}}

User: "How much will we sell of product X next month?"
-> {"intent":"demand_forecast","query":"product X","filters":{"brand":null,"color":null,"price_max":null,"category":null}}

User: "Who supplies our electronics and what's the lead time?"
-> {"intent":"supplier_query","query":"electronics","filters":{"brand":null,"color":null,"price_max":null,"category":"electronics"}}

User: "Show me inventory health for the sports category"
-> {"intent":"sc_analytics","query":"sports","filters":{"brand":null,"color":null,"price_max":null,"category":"sports"}}
"""

_SYSTEM = f"""\
You are an intent router for a shopping and supply chain assistant. Given the \
conversation history and the latest user message, output ONLY a single JSON \
object — no prose, no markdown fences.

Schema:
{{
  "intent": "<search|followup|qa|compare|chitchat|check_stock|reorder_alert|demand_forecast|supplier_query|sc_analytics>",
  "query":  "<search query string, or null>",
  "filters": {{"brand": null, "color": null, "price_max": null, "category": null}}
}}

Rules:
- Shopping intents: search, followup, qa, compare, chitchat
- Supply chain intents: check_stock (stock level lookup), reorder_alert (items to reorder),
  demand_forecast (sales prediction), supplier_query (vendor/lead-time lookup),
  sc_analytics (inventory dashboard/summary)
- query = clean keyword string for product/SKU lookup; null for reorder_alert and sc_analytics
- price_max must be a number or null; brand/color/category must be string or null

{_FEW_SHOT}\
"""


_EMPTY_FILTERS = {"brand": None, "color": None, "price_max": None, "category": None}

_GIFT_WORDS    = ("gift for", "present for", "buying for my", "for my mom", "for my dad",
                   "for my wife", "for my husband", "for my friend", "gift idea",
                   "something for", "buying for someone", "for a friend")
_CART_ADD      = ("add to cart", "add this", "add that", "add the first", "add the second",
                   "put it in cart", "i'll take it", "buy this", "purchase this")
_CART_VIEW     = ("what's in my cart", "show cart", "view cart", "my cart", "cart total")
_CART_REMOVE   = ("remove from cart", "delete from cart", "take out", "remove the")

_SC_REORDER    = ("reorder", "what needs to be ordered", "low stock", "out of stock",
                   "what to reorder", "items to reorder", "purchase order")
_SC_FORECAST   = ("how much will we sell", "demand forecast", "predict sales",
                   "forecast", "next month sales", "expected demand")
_SC_STOCK      = ("in stock", "stock level", "how many", "do we have", "inventory",
                   "is there any", "available units", "qty on hand")
_SC_SUPPLIER   = ("supplier", "vendor", "lead time", "who supplies", "procurement",
                   "purchase from", "source from")
_SC_ANALYTICS  = ("inventory health", "overstock", "inventory report", "stock summary",
                   "dashboard", "supply chain analytics")

_FOLLOWUP_STARTS = ("does it", "do any", "do they", "which one", "can you",
                    "what about", "is there", "are there", "how about",
                    "tell me more", "more about", "the first", "the second",
                    "the third", "that one", "this one")
_COMPARE_WORDS = ("compare", " vs ", " versus ", "difference between", "better than",
                  "which is better", "which one is")
_CHITCHAT = ("hello", "hi ", "hey ", "thanks", "thank you", "bye", "goodbye",
             "good morning", "good evening", "how are you")


def _fast_classify(message: str, has_history: bool) -> dict | None:
    """
    Rule-based fast path — skips the LLM call for obvious cases.
    Returns None if uncertain (LLM call required).
    """
    m = message.lower().strip()

    # Cart intents
    if any(w in m for w in _CART_VIEW):
        return {"intent": "view_cart", "query": None, "filters": _EMPTY_FILTERS}
    if any(w in m for w in _CART_ADD):
        return {"intent": "add_to_cart", "query": message, "filters": _EMPTY_FILTERS}
    if any(w in m for w in _CART_REMOVE):
        return {"intent": "add_to_cart", "query": message, "filters": _EMPTY_FILTERS}

    # Gift / occasion search
    if any(w in m for w in _GIFT_WORDS):
        return {"intent": "gift_search", "query": message, "filters": _EMPTY_FILTERS}

    # Supply chain intents — check before shopping intents
    if any(w in m for w in _SC_REORDER):
        return {"intent": "reorder_alert", "query": None, "filters": _EMPTY_FILTERS}
    if any(w in m for w in _SC_ANALYTICS):
        return {"intent": "sc_analytics", "query": message, "filters": _EMPTY_FILTERS}
    if any(w in m for w in _SC_FORECAST):
        return {"intent": "demand_forecast", "query": message, "filters": _EMPTY_FILTERS}
    if any(w in m for w in _SC_SUPPLIER):
        return {"intent": "supplier_query", "query": message, "filters": _EMPTY_FILTERS}
    if any(w in m for w in _SC_STOCK):
        return {"intent": "check_stock", "query": message, "filters": _EMPTY_FILTERS}

    # Shopping intents
    if any(m.startswith(c) for c in _CHITCHAT) and len(m) < 60:
        return {"intent": "chitchat", "query": None, "filters": _EMPTY_FILTERS}

    if has_history and any(m.startswith(p) for p in _FOLLOWUP_STARTS):
        return {"intent": "followup", "query": None, "filters": _EMPTY_FILTERS}

    if any(w in m for w in _COMPARE_WORDS):
        return {"intent": "compare", "query": message, "filters": _EMPTY_FILTERS}

    # First turn with no history → almost certainly a search
    if not has_history:
        return {"intent": "search", "query": message, "filters": _EMPTY_FILTERS}

    return None   # ambiguous — fall through to LLM


def classify(message: str, history: list[dict]) -> dict:
    """
    Classify the user message and extract search intent + filters.
    Uses a rule-based fast path; falls back to LLM only when ambiguous.

    Returns a dict with keys: intent, query, filters.
    """
    fast = _fast_classify(message, bool(history))
    if fast is not None:
        return fast

    msgs: list[dict] = [{"role": "system", "content": _SYSTEM}]
    msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": message})

    try:
        resp = ollama.chat(
            model="qwen3:1.7b",
            messages=msgs,
            format="json",
            options={"temperature": 0, "num_predict": 256},
            keep_alive="60m",
            think=False,
        )
        # With qwen3, final answer is in content; thinking CoT is in .thinking
        raw = resp.message.content or "{}"
        # strip any accidental <think>...</think> block before json
        if "<think>" in raw:
            raw = raw[raw.rfind("</think>") + 8:].strip()
        result = json.loads(raw)
    except Exception:
        result = {}

    # Sanitise
    if result.get("intent") not in INTENTS:
        result["intent"] = "search"
    if not result.get("query"):
        result["query"] = message if result["intent"] == "search" else None
    filters = result.get("filters") or {}
    result["filters"] = {
        "brand": filters.get("brand") or None,
        "color": filters.get("color") or None,
        "price_max": filters.get("price_max") or None,
        "category": filters.get("category") or None,
    }
    return result

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
import re

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
- Fix typos and expand short/unusual queries: "saries" → "sarees Indian traditional ethnic dress";
  "kurta" → "kurta Indian men traditional clothing"; "duvet" → "duvet comforter bedding"
- Remove ranking/sorting modifiers from query: "best selling", "top rated", "trending", etc.
- Keep product-specific terms unchanged: "noise cancelling", "IPX7", "OLED", "4K" etc.

{_FEW_SHOT}\
"""


_EMPTY_FILTERS = {"brand": None, "color": None, "price_max": None, "category": None}

# Ranking/quality modifiers that pollute semantic search embeddings.
# "best selling sarees" → embedding drifts toward books/music; strip to "sarees"
# Applied to the retrieval query ONLY — filters use the original message.
_MODIFIER_RE = re.compile(
    r"\b(?:"
    r"best[\s\-]?selling|best[\s\-]?seller|bestselling|top[\s\-]?selling|"
    r"top[\s\-]?rated|most[\s\-]?popular|most[\s\-]?loved|most[\s\-]?reviewed|"
    r"highly[\s\-]?rated|highly[\s\-]?reviewed|well[\s\-]?reviewed|"
    r"trending|hot[\s\-]?selling|fast[\s\-]?selling|"
    r"new[\s\-]?arrivals?|latest[\s\-]?arrivals?|"
    r"award[\s\-]?winning|editor[s']?[\s\-]?choice|"
    r"good|great|nice|excellent|amazing|awesome|fantastic|incredible|"
    r"affordable|budget[\s\-]?friendly|inexpensive|economical|cheap"
    r")\b\s*",
    re.I,
)


def _clean_query(message: str) -> str:
    """Strip ranking/quality modifier words that pull embeddings off-target.

    Examples:
      'best selling sarees'            -> 'sarees'
      'top rated gaming headset'       -> 'gaming headset'
      'trending affordable sneakers'   -> 'sneakers'
      'noise cancelling headphones'    -> 'noise cancelling headphones' (unchanged)
    """
    cleaned = _MODIFIER_RE.sub(" ", message).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # Fall back to original if we wiped everything (e.g. bare "best")
    return cleaned if len(cleaned) >= 3 else message


_COLORS = frozenset({
    "black", "white", "red", "blue", "green", "yellow", "gray", "grey",
    "silver", "gold", "pink", "purple", "orange", "brown", "beige", "navy",
    "rose", "mint", "teal", "coral", "maroon", "cream", "tan", "charcoal",
})
_PRICE_RE  = re.compile(r"(?:under|less than|below|max|up to|at most)\s+\$?\s*(\d+(?:\.\d+)?)", re.I)
_PRICE_RE2 = re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(?:or less|max|maximum)", re.I)


def _extract_filters(message: str) -> dict:
    """Regex-based filter extraction for price and color — fast, no LLM."""
    m = message.lower()
    words = set(re.findall(r"\b\w+\b", m))

    price_max = None
    pm = _PRICE_RE.search(m) or _PRICE_RE2.search(m)
    if pm:
        try:
            price_max = float(pm.group(1))
        except ValueError:
            pass

    color = next((c for c in _COLORS if c in words), None)

    return {"brand": None, "color": color, "price_max": price_max, "category": None}

_GIFT_WORDS    = ("gift for", "present for", "buying for my", "for my mom", "for my dad",
                   "for my wife", "for my husband", "for my friend", "gift idea",
                   "something for", "buying for someone", "for a friend")
_CART_ADD      = ("add to cart", "add this", "add that", "add the first", "add the second",
                   "put it in cart", "i'll take it", "buy this", "purchase this")
_CART_VIEW     = ("what's in my cart", "show cart", "view cart", "my cart", "cart total")
_CART_REMOVE   = ("remove from cart", "delete from cart", "take out", "remove the")

_SC_REORDER    = ("reorder alert", "what needs to be reordered", "items to reorder",
                   "purchase order", "create a po", "replenish stock", "what to reorder",
                   "low stock", "low inventory", "stock shortage")
_SC_FORECAST   = ("demand forecast", "predict sales", "sales forecast",
                   "next month sales", "expected demand", "how much will we sell")
_SC_STOCK      = ("stock level", "qty on hand", "units on hand",
                   "check inventory", "inventory level", "inventory status",
                   "show inventory", "view inventory", "inventory check",
                   "warehouse stock", "current stock",
                   "how many units do we have", "how many units in stock",
                   "check stock", "stock check",
                   "what is in stock", "what's in stock",
                   "what is out of stock", "what's out of stock",
                   "out of stock report", "in stock report")
_SC_SUPPLIER   = ("our supplier", "our vendor", "vendor lead time", "who supplies us",
                   "supplier information", "supplier details", "supplier list",
                   "procurement team", "source from supplier", "supplier lead time")
_SC_ANALYTICS  = ("inventory health", "overstock", "inventory report",
                   "stock summary", "supply chain", "supply chain analytics",
                   "inventory dashboard", "sc dashboard", "show me the inventory",
                   "all inventory", "full inventory")

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
        return {"intent": "compare", "query": _clean_query(message), "filters": _EMPTY_FILTERS}

    # First turn with no history → almost certainly a search.
    # Use fast-path only when the cleaned query has ≥ 3 meaningful words.
    # Short or heavily-modified queries (e.g. "saries", "best selling saries")
    # fall through to the LLM, which rewrites them into proper search terms.
    if not has_history:
        cleaned = _clean_query(message)
        word_count = len(cleaned.split())
        if word_count >= 3:
            return {"intent": "search", "query": cleaned, "filters": _extract_filters(message)}
        # 1-2 word query: let LLM rewrite for better retrieval accuracy
        return None

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
            model="qwen3.5:latest",
            messages=msgs,
            format="json",
            options={"temperature": 0, "num_predict": 128, "num_ctx": 2048},
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
    # Clean modifier words from retrieval query (LLM usually handles this but be safe)
    if result.get("query"):
        result["query"] = _clean_query(result["query"])
    filters = result.get("filters") or {}
    result["filters"] = {
        "brand": filters.get("brand") or None,
        "color": filters.get("color") or None,
        "price_max": filters.get("price_max") or None,
        "category": filters.get("category") or None,
    }
    return result

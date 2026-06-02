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

INTENTS = frozenset({"search", "followup", "qa", "compare", "chitchat"})

# Few-shot examples derived from MG-ShopDial conversation patterns
_FEW_SHOT = """
Examples:
User: "I'm looking for a noise-cancelling headphone under $200"
→ {"intent":"search","query":"noise-cancelling headphones under 200 dollars","filters":{"price_max":200,"brand":null,"color":null,"category":"headphones"}}

User: "Do any of them come in white?"
→ {"intent":"followup","query":null,"filters":{"color":"white","brand":null,"price_max":null,"category":null}}

User: "What's the battery life on the second one?"
→ {"intent":"qa","query":null,"filters":{"brand":null,"color":null,"price_max":null,"category":null}}

User: "Which is better for gaming, the first or third?"
→ {"intent":"compare","query":null,"filters":{"brand":null,"color":null,"price_max":null,"category":null}}

User: "Thanks, that's all I needed!"
→ {"intent":"chitchat","query":null,"filters":{"brand":null,"color":null,"price_max":null,"category":null}}

User: "Show me Sony wireless earbuds"
→ {"intent":"search","query":"Sony wireless earbuds","filters":{"brand":"Sony","color":null,"price_max":null,"category":"earbuds"}}
"""

_SYSTEM = f"""\
You are an intent router for a shopping assistant. Given the conversation \
history and the latest user message, output ONLY a single JSON object — \
no prose, no markdown fences.

Schema:
{{
  "intent": "<search|followup|qa|compare|chitchat>",
  "query":  "<rephrased product search query, or null if not a product search>",
  "filters": {{"brand": null, "color": null, "price_max": null, "category": null}}
}}

Rules:
- intent must be exactly one of: search, followup, qa, compare, chitchat
- query should be a clean, keyword-rich product search string for vector search
- For followup/qa/compare/chitchat, set query to null
- price_max must be a number or null (never a string)
- brand/color/category must be a string or null

{_FEW_SHOT}\
"""


def classify(message: str, history: list[dict]) -> dict:
    """
    Classify the user message and extract search intent + filters.

    Returns a dict with keys: intent, query, filters.
    """
    msgs: list[dict] = [{"role": "system", "content": _SYSTEM}]
    # include up to last 3 turns of history for followup detection
    msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": message})

    try:
        resp = ollama.chat(
            model="qwen3:1.7b",   # small fast router — warm latency ~80 ms
            messages=msgs,
            format="json",
            options={"temperature": 0, "num_predict": 256},
        )
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

"""
Week 1 RAG pipeline: retrieve relevant products → ground Qwen3 answer.
"""

from __future__ import annotations

from collections.abc import Iterator

from rufus.llm import OllamaClient
from rufus.retriever import Product, ProductRetriever

SYSTEM_PROMPT = """\
You are Rufus, an Amazon-style AI shopping assistant.

RULES — follow every time, no exceptions:
1. Use ONLY the product data provided. Never invent specs, prices, ratings, or features.
2. PRICE: If a product has "Price: not listed", do NOT mention a price for it — omit it entirely.
3. RATINGS: Only mention ratings if they are explicitly shown in the product data.
4. Always format product names in **bold**.
5. Keep answers under 120 words.
6. Do not start with "I" or "Sure" or "Of course" or "Great question".
7. Do not repeat the user's question back to them.
8. NEVER guess or estimate any data not explicitly provided.

OUTPUT FORMAT by intent:
- search / compare: numbered list, one product per line; include price ONLY if shown in data
- qa / followup: 1-2 sentences answering the specific question from the data
- chitchat: 1 sentence, friendly

IMPORTANT:
- Use "I couldn't find an exact match" ONLY when the retrieved products list is COMPLETELY EMPTY.
- If products ARE listed, describe them directly — do NOT open with apologies or disclaimers
  about sorting modifiers ("best selling", "top rated") not being in the data. Just list them.\
"""


def _format_context(products: list[Product]) -> str:
    from rufus.reviews import get_c4_metadata_batch, get_meta, get_reviews
    ids     = [p.product_id for p in products]
    c4_map  = get_c4_metadata_batch(ids)
    lines: list[str] = []
    for i, p in enumerate(products, 1):
        line = f"{i}. **{p.title}**"
        if p.brand:
            line += f" — Brand: {p.brand}"
        if p.color:
            line += f", Color: {p.color}"

        meta = get_meta(p.product_id)
        if meta:
            if meta.get("price"):
                line += f", Price: ${meta['price']:.2f}"
            else:
                line += ", Price: not listed"
            if meta.get("avg_rating") and meta.get("rating_count"):
                line += f", Rating: {meta['avg_rating']:.1f}/5 ({meta['rating_count']:,} reviews)"
            if meta.get("store") and meta["store"] != p.brand:
                line += f", Store: {meta['store']}"
        else:
            line += ", Price: not listed"

        # (relevance score intentionally omitted — internal metric, not for LLM context)

        features = None
        if meta and meta.get("features"):
            features = " • ".join(meta["features"][:2])
        elif p.bullet_point:
            features = p.bullet_point.replace("\n", " ").strip()[:150]
        if features:
            line += f"\n   Features: {features}"
        elif meta and meta.get("description"):
            line += f"\n   About: {meta['description'][:200]}"
        elif c4_map.get(p.product_id):
            line += f"\n   About: {c4_map[p.product_id][:250]}"

        # Top helpful review snippet for richer Q&A grounding
        snippets = get_reviews(p.product_id, limit=1)
        if snippets:
            line += f'\n   Review: "{snippets[0]}"'

        lines.append(line)
    return "\n".join(lines)


def _format_raw_context(raw: list[dict]) -> str:
    """
    Format raw product dicts (from client state, already JSON-serialised) as LLM context.
    Used when intent is followup — products come back from the client's last turn state.
    """
    lines: list[str] = []
    for i, p in enumerate(raw, 1):
        line = f"{i}. **{p.get('title') or 'Unknown Product'}**"
        if p.get("brand"):
            line += f" — Brand: {p['brand']}"
        if p.get("color"):
            line += f", Color: {p['color']}"
        price = p.get("price")
        if price:
            try:
                line += f", Price: ${float(price):.2f}"
            except (ValueError, TypeError):
                line += ", Price: not listed"
        else:
            line += ", Price: not listed"
        rating = p.get("avg_rating")
        count  = p.get("rating_count")
        if rating and count:
            try:
                line += f", Rating: {float(rating):.1f}/5 ({int(count):,} reviews)"
            except (ValueError, TypeError):
                pass
        bp = p.get("bullet_point")
        if bp:
            line += f"\n   Features: {str(bp).replace(chr(10), ' ').strip()[:150]}"
        lines.append(line)
    return "\n".join(lines) if lines else "No products found."


class RufusRAG:
    def __init__(
        self,
        ollama_model: str = "qwen3.5:latest",
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k
        self.retriever = ProductRetriever()
        self.llm = OllamaClient(model=ollama_model)

    def query(self, question: str, stream: bool = False) -> tuple[list[Product], str | Iterator[str]]:
        products = self.retriever.retrieve(question, top_k=self.top_k)
        context = _format_context(products)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Retrieved products:\n{context}\n\n"
                    f"Customer question: {question}"
                ),
            },
        ]
        answer = self.llm.chat(messages, stream=stream)
        return products, answer

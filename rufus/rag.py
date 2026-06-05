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
1. Use ONLY the product data provided. Never invent specs, prices, or features.
2. Always format product names in **bold**.
3. Keep answers under 120 words.
4. Do not start with "I" or "Sure" or "Of course" or "Great question".
5. Do not repeat the user's question back to them.

OUTPUT FORMAT by intent:
- search / compare: numbered list, one product per line, key feature + price if available
- qa / followup: 1-2 sentences answering the specific question
- chitchat: 1 sentence, friendly

If no products match, say: "I couldn't find an exact match — try searching for [better keywords]."\
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
            if meta.get("avg_rating") and meta.get("rating_count"):
                line += f", Rating: {meta['avg_rating']:.1f}★ ({meta['rating_count']:,} reviews)"
            if meta.get("store") and meta["store"] != p.brand:
                line += f", Store: {meta['store']}"

        line += f" (relevance: {p.score:.2f})"

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

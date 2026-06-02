"""
Week 1 RAG pipeline: retrieve relevant products → ground Qwen3 answer.
"""

from __future__ import annotations

from collections.abc import Iterator

from rufus.llm import OllamaClient
from rufus.retriever import Product, ProductRetriever

SYSTEM_PROMPT = """\
You are Rufus, an AI shopping assistant. You help customers find products, \
answer questions about items, and make recommendations based on their needs.

You are given a list of relevant products retrieved from the catalog. \
Use ONLY the provided product information to answer — do not invent specs, \
prices, or features. If none of the products match the customer's need, \
say so clearly and suggest what to search for instead.

Be concise, helpful, and direct. Format product names in **bold**.\
"""


def _format_context(products: list[Product]) -> str:
    from rufus.reviews import get_meta
    lines: list[str] = []
    for i, p in enumerate(products, 1):
        line = f"{i}. **{p.title}**"
        if p.brand:
            line += f" — Brand: {p.brand}"
        if p.color:
            line += f", Color: {p.color}"

        # Enrich with Amazon Reviews metadata if available
        meta = get_meta(p.product_id)
        if meta:
            if meta.get("price"):
                line += f", Price: ${meta['price']:.2f}"
            if meta.get("avg_rating") and meta.get("rating_count"):
                line += f", Rating: {meta['avg_rating']:.1f}★ ({meta['rating_count']:,} reviews)"
            if meta.get("store") and meta["store"] != p.brand:
                line += f", Store: {meta['store']}"

        line += f" (relevance: {p.score:.2f})"

        # Features: prefer reviews metadata features over ESCI bullet points
        features = None
        if meta and meta.get("features"):
            features = " • ".join(meta["features"][:4])
        elif p.bullet_point:
            features = p.bullet_point.replace("\n", " ").strip()[:300]
        if features:
            line += f"\n   Features: {features}"

        # Description: from reviews metadata (richer than ESCI)
        desc = (meta or {}).get("description") or p.description
        if desc:
            line += f"\n   Description: {str(desc)[:300]}"

        if meta and meta.get("categories"):
            line += f"\n   Category: {' > '.join(meta['categories'][:3])}"

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

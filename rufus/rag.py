"""
Week 1 RAG pipeline: retrieve relevant products → ground Qwen3 answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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
    lines: list[str] = []
    for i, p in enumerate(products, 1):
        line = f"{i}. **{p.title}**"
        if p.brand:
            line += f" — Brand: {p.brand}"
        if p.color:
            line += f", Color: {p.color}"
        line += f" (relevance: {p.score:.2f})"
        if p.bullet_point:
            bullets = p.bullet_point.replace("\n", " ").strip()[:300]
            line += f"\n   Features: {bullets}"
        lines.append(line)
    return "\n".join(lines)


class RufusRAG:
    def __init__(
        self,
        ollama_model: str = "qwen3.5:latest",
        qdrant_path: Path = Path("data/qdrant_storage"),
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k
        self.retriever = ProductRetriever(qdrant_path=qdrant_path)
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

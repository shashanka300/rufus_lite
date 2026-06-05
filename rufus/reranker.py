"""
Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

After bi-encoder retrieval returns a candidate pool (typically 40–80 products),
the cross-encoder scores each (query, product) pair jointly and re-ranks them.
This is slower per item but far more accurate than dot-product similarity alone.

Model is lazy-loaded on first call and stays in GPU VRAM.
"""

from __future__ import annotations

import dataclasses
import time

import torch

import rufus.hardware  # apply TF32 + cuDNN benchmark globally
from rufus.retriever import Product

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def _product_passage(p: Product) -> str:
    parts = [p.title]
    if p.brand:
        parts.append(f"Brand: {p.brand}")
    if p.color:
        parts.append(f"Color: {p.color}")
    if p.bullet_point:
        parts.append(p.bullet_point[:400])
    if p.description:
        parts.append(p.description[:300])
    return " | ".join(parts)


class ProductReranker:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                device=self._device,
                max_length=512,
                model_kwargs={"torch_dtype": torch.float16},  # fp16 on RTX 5090
            )
        return self._model

    def rerank(self, query: str, products: list[Product], top_k: int = 5) -> list[Product]:
        if not products:
            return products

        from rufus.cache import rerank_cache
        cache_key = (query, top_k, tuple(p.product_id for p in products))
        cached = rerank_cache.fetch(cache_key)
        if cached is not None:
            print(f"[rerank] cache hit")
            return cached

        t0 = time.perf_counter()
        pairs = [(query, _product_passage(p)) for p in products]
        scores = self.model.predict(pairs, show_progress_bar=False, batch_size=128)  # 32 GB VRAM
        print(f"[rerank] {time.perf_counter()-t0:.2f}s  "
              f"({len(products)} -> {top_k} products, device={self._device})")

        ranked = sorted(zip(scores, products), key=lambda x: x[0], reverse=True)
        result = [dataclasses.replace(p, score=float(s)) for s, p in ranked[:top_k]]
        rerank_cache.put(cache_key, result)
        return result

"""
Product retriever: BGE-M3 dense embedding + Qdrant nearest-neighbour search.
Model and Qdrant client are both lazy-loaded on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

import rufus.hardware  # apply TF32 + cuDNN benchmark globally

EMBED_MODEL = "BAAI/bge-m3"
COLLECTION = "rufus_products"
QDRANT_PATH = Path("data/qdrant_storage")


@dataclass
class Product:
    product_id: str
    title: str
    brand: str | None
    color: str | None
    bullet_point: str | None
    description: str | None
    locale: str
    score: float
    image_url: str | None = None


class ProductRetriever:
    def __init__(
        self,
        collection: str = COLLECTION,
        model_name: str = EMBED_MODEL,
    ) -> None:
        self.collection = collection
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name, device=self._device,
                model_kwargs={"torch_dtype": torch.float16}  # fp16 ~2x faster on RTX 5090
            )
        return self._model

    @property
    def _client(self):
        from rufus.qdrant import get_client
        return get_client()

    def _embed(self, query: str) -> list[float]:
        from rufus.cache import embedding_cache
        cached = embedding_cache.fetch((query,))
        if cached is not None:
            return cached
        vec = self.model.encode(
            query, normalize_embeddings=True,
            batch_size=64,       # RTX 5090 32 GB — can handle large batches
            convert_to_numpy=True,
        ).tolist()
        embedding_cache.put((query,), vec)
        return vec

    def retrieve(self, query: str, top_k: int = 5) -> list[Product]:
        vec = self._embed(query)
        hits = self._client.query_points(
            collection_name=self.collection,
            query=vec,
            limit=top_k,
        )
        return [
            Product(
                product_id=h.payload.get("product_id", ""),
                title=h.payload.get("product_title", ""),
                brand=h.payload.get("product_brand"),
                color=h.payload.get("product_color"),
                bullet_point=h.payload.get("product_bullet_point"),
                description=h.payload.get("product_description"),
                locale=h.payload.get("product_locale", "us"),
                score=h.score,
                image_url=h.payload.get("image_url"),
            )
            for h in hits.points
        ]

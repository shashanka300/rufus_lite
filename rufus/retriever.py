"""
Product retriever: BGE-M3 dense embedding + Qdrant nearest-neighbour search.
Model is lazy-loaded on first call so import is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

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


class ProductRetriever:
    def __init__(
        self,
        qdrant_path: Path = QDRANT_PATH,
        collection: str = COLLECTION,
        model_name: str = EMBED_MODEL,
    ) -> None:
        self.collection = collection
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._client = QdrantClient(path=str(qdrant_path))

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def retrieve(self, query: str, top_k: int = 5) -> list[Product]:
        vec = self.model.encode(query, normalize_embeddings=True).tolist()
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
            )
            for h in hits.points
        ]

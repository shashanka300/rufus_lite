"""
CLIP-based product retriever using openai/clip-vit-large-patch14.
Both the CLIP model and the Qdrant client are lazy-loaded on first use.
"""

from __future__ import annotations

import torch

from rufus.retriever import Product

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
COLLECTION = "rufus_clip"


class CLIPRetriever:
    def __init__(
        self,
        collection: str = COLLECTION,
        model_name: str = CLIP_MODEL_NAME,
    ) -> None:
        self.collection = collection
        self.model_name = model_name
        self._model = None
        self._processor = None

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

    @property
    def _client(self):
        from rufus.qdrant import get_client
        return get_client()

    def _load_model(self) -> None:
        from transformers import CLIPModel, CLIPProcessor
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()

    def encode_text(self, text: str) -> list[float]:
        if self._model is None:
            self._load_model()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            embeds = self._model.text_projection(out.pooler_output)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds[0].cpu().tolist()

    def encode_image(self, image) -> list[float]:
        if self._model is None:
            self._load_model()
        inputs = self._processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with torch.no_grad():
            out = self._model.vision_model(pixel_values=pixel_values)
            embeds = self._model.visual_projection(out.pooler_output)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds[0].cpu().tolist()

    def available(self) -> bool:
        try:
            return (
                self._client.collection_exists(self.collection)
                and (self._client.get_collection(self.collection).points_count or 0) > 0
            )
        except Exception:
            return False

    def _hits_to_products(self, hits) -> list[Product]:
        return [
            Product(
                product_id=h.payload.get("product_id", ""),
                title=h.payload.get("product_title", ""),
                brand=h.payload.get("product_brand"),
                color=h.payload.get("product_color"),
                bullet_point=h.payload.get("product_bullet_point"),
                description=None,
                locale=h.payload.get("product_locale", "us"),
                score=h.score,
                image_url=h.payload.get("image_url"),
            )
            for h in hits.points
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[Product]:
        if not self.available():
            return []
        vec = self.encode_text(query)
        hits = self._client.query_points(collection_name=self.collection, query=vec, limit=top_k)
        return self._hits_to_products(hits)

    def retrieve_by_image(self, image, top_k: int = 10) -> list[Product]:
        if not self.available():
            return []
        vec = self.encode_image(image)
        hits = self._client.query_points(collection_name=self.collection, query=vec, limit=top_k)
        return self._hits_to_products(hits)

"""
Amazon Reviews 2023 — Electronics metadata loader.

Loads the 10-shard raw_meta_Electronics parquet files into an in-memory
lookup dict keyed by ASIN (parent_asin).  Provides price, average rating,
review count, full features list, and categories — all absent from ESCI.

The loader is cached as a module-level singleton; call get_meta() from
anywhere to get enriched product attributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

_META: dict[str, dict] | None = None
_DATA_DIR = Path("data/amazon_reviews/raw_meta_Electronics")


class ReviewMeta(TypedDict, total=False):
    price: float | None
    avg_rating: float | None
    rating_count: int | None
    features: list[str]
    description: str | None
    categories: list[str]
    store: str | None


def _load() -> dict[str, ReviewMeta]:
    import pandas as pd

    parquets = sorted(_DATA_DIR.glob("*.parquet"))
    if not parquets:
        return {}

    frames = []
    for p in parquets:
        try:
            df = pd.read_parquet(
                p,
                columns=["parent_asin", "price", "average_rating",
                         "rating_number", "features", "description",
                         "categories", "store"],
            )
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True).drop_duplicates("parent_asin")

    meta: dict[str, ReviewMeta] = {}
    for _, row in df.iterrows():
        asin = row["parent_asin"]
        if not isinstance(asin, str) or not asin:
            continue

        # price: may be None or "$X.XX" string in some versions
        price = row.get("price")
        if isinstance(price, str):
            try:
                price = float(price.replace("$", "").replace(",", ""))
            except ValueError:
                price = None

        # features: list of strings
        feats = row.get("features")
        if not isinstance(feats, list):
            feats = []

        # description: list or string
        desc = row.get("description")
        if isinstance(desc, list):
            desc = " ".join(str(d) for d in desc if d)
        elif not isinstance(desc, str):
            desc = None
        if desc:
            desc = desc[:600]

        # categories: numpy array or list
        cats = row.get("categories")
        if hasattr(cats, "tolist"):
            cats = cats.tolist()
        if not isinstance(cats, list):
            cats = []

        meta[asin] = ReviewMeta(
            price=price if isinstance(price, (int, float)) else None,
            avg_rating=float(row["average_rating"]) if row.get("average_rating") is not None else None,
            rating_count=int(row["rating_number"]) if row.get("rating_number") is not None else None,
            features=[str(f) for f in feats[:5]],
            description=desc or None,
            categories=[str(c) for c in cats if c],
            store=str(row["store"]) if row.get("store") else None,
        )

    return meta


def get_meta(product_id: str) -> ReviewMeta | None:
    """Return enriched metadata for a product_id/ASIN, or None if not found."""
    global _META
    if _META is None:
        _META = _load()
    return _META.get(product_id)


def meta_available() -> bool:
    global _META
    if _META is None:
        _META = _load()
    return bool(_META)

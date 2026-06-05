"""
Product metadata enrichment from Amazon Reviews 2023.

Priority lookup order:
  1. rufus_reviews.db  — 5.5 M ASINs across 33 categories (all downloaded categories)
  2. raw_meta_Electronics parquets — fallback if DB not yet built (Electronics only)

Both sources provide: price, avg_rating, rating_count, features, description, categories.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypedDict

DB_PATH   = Path("data/rufus_reviews.db")
_DATA_DIR = Path("data/amazon_reviews/raw_meta_Electronics")

# Module-level cache: ASIN -> ReviewMeta
_META: dict[str, "ReviewMeta"] | None = None
_USE_DB: bool | None = None   # set on first call


class ReviewMeta(TypedDict, total=False):
    price:        float | None
    avg_rating:   float | None
    rating_count: int   | None
    features:     list[str]
    description:  str   | None
    categories:   list[str]
    store:        str   | None


# ── DB-backed path (preferred) ────────────────────────────────────────────────

def _load_from_db() -> dict[str, ReviewMeta]:
    if not DB_PATH.exists():
        return {}
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT asin, price, avg_rating, rating_count, features, description, category, store "
        "FROM product_meta"
    ).fetchall()
    conn.close()

    import json
    meta: dict[str, ReviewMeta] = {}
    for r in rows:
        feats = []
        try:
            feats = json.loads(r["features"] or "[]")
            if not isinstance(feats, list):
                feats = []
        except Exception:
            pass
        cats = [r["category"]] if r["category"] else []
        meta[r["asin"]] = ReviewMeta(
            price        = float(r["price"]) if r["price"] is not None else None,
            avg_rating   = float(r["avg_rating"]) if r["avg_rating"] is not None else None,
            rating_count = int(r["rating_count"]) if r["rating_count"] is not None else None,
            features     = [str(f) for f in feats[:5]],
            description  = (r["description"] or "")[:600] or None,
            categories   = cats,
            store        = r["store"] or None,
        )
    return meta


# ── Parquet fallback (Electronics only) ──────────────────────────────────────

def _load_from_parquets() -> dict[str, ReviewMeta]:
    import pandas as pd

    parquets = sorted(_DATA_DIR.glob("*.parquet"))
    if not parquets:
        return {}

    frames = []
    for p in parquets:
        try:
            frames.append(pd.read_parquet(
                p, columns=["parent_asin", "price", "average_rating",
                             "rating_number", "features", "description",
                             "categories", "store"],
            ))
        except Exception:
            continue

    if not frames:
        return {}

    import pandas as pd
    df = pd.concat(frames, ignore_index=True).drop_duplicates("parent_asin")

    meta: dict[str, ReviewMeta] = {}
    for _, row in df.iterrows():
        asin = row["parent_asin"]
        if not isinstance(asin, str) or not asin:
            continue

        price = row.get("price")
        if isinstance(price, str):
            try:
                price = float(price.replace("$", "").replace(",", ""))
            except ValueError:
                price = None

        feats = row.get("features") or []
        if not isinstance(feats, list):
            feats = []

        desc = row.get("description")
        if isinstance(desc, list):
            desc = " ".join(str(d) for d in desc if d)
        elif not isinstance(desc, str):
            desc = None
        if desc:
            desc = desc[:600]

        cats = row.get("categories") or []
        if hasattr(cats, "tolist"):
            cats = cats.tolist()
        if not isinstance(cats, list):
            cats = []

        meta[asin] = ReviewMeta(
            price        = price if isinstance(price, (int, float)) else None,
            avg_rating   = float(row["average_rating"]) if row.get("average_rating") is not None else None,
            rating_count = int(row["rating_number"])    if row.get("rating_number")    is not None else None,
            features     = [str(f) for f in feats[:5]],
            description  = desc or None,
            categories   = [str(c) for c in cats if c],
            store        = str(row["store"]) if row.get("store") else None,
        )
    return meta


# ── Public API ────────────────────────────────────────────────────────────────

def _ensure_loaded() -> None:
    global _META, _USE_DB
    if _META is not None:
        return
    if DB_PATH.exists():
        _META = _load_from_db()
        _USE_DB = True
        print(f"[reviews] loaded {len(_META):,} ASINs from rufus_reviews.db (33 categories)")
    else:
        _META = _load_from_parquets()
        _USE_DB = False
        print(f"[reviews] loaded {len(_META):,} ASINs from Electronics parquets (fallback)")


def get_meta(product_id: str) -> ReviewMeta | None:
    _ensure_loaded()
    return _META.get(product_id)


def get_meta_batch(product_ids: list[str]) -> dict[str, "ReviewMeta"]:
    """Efficient batch lookup without loading all 5.5M rows.

    Queries the DB index directly when DB exists; falls back to the in-memory
    cache (parquet path) otherwise.
    """
    if not product_ids:
        return {}
    if not DB_PATH.exists():
        _ensure_loaded()
        return {pid: m for pid in product_ids if (m := (_META or {}).get(pid))}

    import json as _json
    placeholders = ",".join("?" * len(product_ids))
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT asin, price, avg_rating, rating_count, features, description, category, store "
        f"FROM product_meta WHERE asin IN ({placeholders})",
        product_ids,
    ).fetchall()
    conn.close()

    result: dict[str, ReviewMeta] = {}
    for r in rows:
        feats: list = []
        try:
            feats = _json.loads(r["features"] or "[]")
            if not isinstance(feats, list):
                feats = []
        except Exception:
            pass
        result[r["asin"]] = ReviewMeta(
            price        = float(r["price"]) if r["price"] is not None else None,
            avg_rating   = float(r["avg_rating"]) if r["avg_rating"] is not None else None,
            rating_count = int(r["rating_count"]) if r["rating_count"] is not None else None,
            features     = [str(f) for f in feats[:5]],
            description  = (r["description"] or "")[:600] or None,
            categories   = [r["category"]] if r["category"] else [],
            store        = r["store"] or None,
        )
    return result


def get_reviews(product_id: str, limit: int = 2) -> list[str]:
    """Return top helpful review snippets for a product (150 chars each)."""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT text FROM reviews WHERE asin = ? ORDER BY helpful_votes DESC LIMIT ?",
        (product_id, limit),
    ).fetchall()
    conn.close()
    return [r[0][:150] for r in rows if r[0]]


def get_c4_metadata(product_id: str) -> str | None:
    """Return Amazon C4 rich description for an ASIN (up to 500 chars)."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT metadata FROM c4_metadata WHERE asin = ?", (product_id,)
    ).fetchone()
    conn.close()
    return row[0][:500] if row and row[0] else None


def get_c4_metadata_batch(product_ids: list[str]) -> dict[str, str]:
    """Batch C4 description lookup; returns {asin: metadata_snippet}."""
    if not product_ids or not DB_PATH.exists():
        return {}
    placeholders = ",".join("?" * len(product_ids))
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        f"SELECT asin, metadata FROM c4_metadata WHERE asin IN ({placeholders})",
        product_ids,
    ).fetchall()
    conn.close()
    return {r[0]: r[1][:500] for r in rows if r[1]}


def meta_available() -> bool:
    _ensure_loaded()
    return bool(_META)

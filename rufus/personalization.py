"""
User personalization — real session-based preference tracking + rerank bias.

Three complementary signals:
  1. Brand / category affinity — accumulated from products shown this session
  2. Viewed-product similarity — Qdrant ANN on the user's viewed product vectors
  3. Seed profiles — deterministic fallback for new sessions with no history

Storage: user_profiles table in rufus_sc.db (same DB as inventory).

Ingested supporting data (rufus_personalization.db):
  item_popularity   — 235K items with view/cart/purchase counts (RetailRocket)
  co_view           — 500K co-viewed item pairs (RetailRocket sessions)
  basket_copurchase — 1M co-purchased pairs (Instacart 3.4M orders)
  product_popularity — 49K Instacart product purchase counts
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rufus.inventory import get_db

_DDL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    session_id         TEXT PRIMARY KEY,
    brand_prefs        TEXT DEFAULT '{}',
    cat_prefs          TEXT DEFAULT '{}',
    price_min          REAL DEFAULT 0,
    price_max          REAL DEFAULT 999,
    viewed_product_ids TEXT DEFAULT '[]',
    updated_at         TEXT
);
"""

_MAX_VIEWED = 20   # keep last N viewed product IDs per session

_SEED_PROFILES = [
    {"brand_prefs": {"Sony": 5, "Apple": 3, "Samsung": 2},
     "cat_prefs":   {"electronics": 8, "headphones": 4},
     "price_min": 50, "price_max": 300},
    {"brand_prefs": {"Nike": 6, "Adidas": 4, "Under Armour": 2},
     "cat_prefs":   {"sports_and_outdoors": 7, "shoes": 5},
     "price_min": 30, "price_max": 150},
    {"brand_prefs": {"Anker": 5, "Belkin": 3, "AmazonBasics": 4},
     "cat_prefs":   {"electronics": 6, "phone_accessories": 5},
     "price_min": 10, "price_max": 80},
    {"brand_prefs": {"KitchenAid": 4, "Cuisinart": 3, "Instant Pot": 5},
     "cat_prefs":   {"home_kitchen": 8, "small_appliances": 4},
     "price_min": 25, "price_max": 200},
    {"brand_prefs": {"Lego": 6, "Hasbro": 3, "Mattel": 2},
     "cat_prefs":   {"toys_games": 9, "building_sets": 5},
     "price_min": 15, "price_max": 100},
]


def _init() -> None:
    with get_db() as conn:
        conn.executescript(_DDL)


def get_profile(session_id: str) -> dict:
    _init()
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM user_profiles WHERE session_id = ?", (session_id,)
        ).fetchone()

    if row:
        return {
            "brand_prefs":        json.loads(row["brand_prefs"]        or "{}"),
            "cat_prefs":          json.loads(row["cat_prefs"]          or "{}"),
            "price_min":          row["price_min"],
            "price_max":          row["price_max"],
            "viewed_product_ids": json.loads(row["viewed_product_ids"] or "[]"),
            "is_mock": False,
        }

    seed = _SEED_PROFILES[hash(session_id) % len(_SEED_PROFILES)]
    return {**seed, "viewed_product_ids": [], "is_mock": True}


def update_profile(session_id: str, products: list) -> None:
    """Accumulate brand/category affinity and viewed product IDs from shown results."""
    if not products:
        return
    _init()
    profile  = get_profile(session_id)
    brand_p  = profile.get("brand_prefs", {})
    cat_p    = profile.get("cat_prefs",   {})
    viewed   = profile.get("viewed_product_ids", [])

    for p in products:
        brand = getattr(p, "brand", None) or (p.get("brand") if isinstance(p, dict) else None)
        cat   = getattr(p, "category", None) or (p.get("category") if isinstance(p, dict) else None)
        pid   = getattr(p, "product_id", None) or (p.get("product_id") if isinstance(p, dict) else None)
        if brand:
            brand_p[brand] = brand_p.get(brand, 0) + 1
        if cat:
            cat_p[cat] = cat_p.get(cat, 0) + 1
        if pid and pid not in viewed:
            viewed.append(pid)

    viewed = viewed[-_MAX_VIEWED:]  # keep only most recent

    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO user_profiles
               (session_id, brand_prefs, cat_prefs, price_min, price_max,
                viewed_product_ids, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                session_id,
                json.dumps(brand_p),
                json.dumps(cat_p),
                profile.get("price_min", 0),
                profile.get("price_max", 999),
                json.dumps(viewed),
                datetime.utcnow().isoformat(),
            ),
        )


def apply_preference_bias(products: list, session_id: str, weight: float = 0.15) -> list:
    """
    Boost relevance scores for products matching the user's brand/category profile.
    weight=0.15 → a preferred brand adds up to 15% to the score.
    """
    if not products:
        return products

    profile = get_profile(session_id)
    brand_p = profile.get("brand_prefs", {})
    cat_p   = profile.get("cat_prefs",   {})

    if not brand_p and not cat_p:
        return products

    max_brand = max(brand_p.values(), default=1)
    max_cat   = max(cat_p.values(),   default=1)

    boosted = []
    for p in products:
        brand = getattr(p, "brand", "") or ""
        cat   = getattr(p, "category", "") or ""
        boost = 0.0
        if brand and brand in brand_p:
            boost += weight * (brand_p[brand] / max_brand)
        if cat and cat in cat_p:
            boost += weight * 0.5 * (cat_p[cat] / max_cat)
        if boost > 0:
            try:
                from dataclasses import replace
                p = replace(p, score=min(p.score + boost, 1.0))
            except Exception:
                pass
        boosted.append(p)

    boosted.sort(key=lambda x: getattr(x, "score", 0), reverse=True)
    return boosted


def get_similar_to_viewed(session_id: str, top_k: int = 5) -> list:
    """
    Return products semantically similar to what this session has viewed.

    Uses BGE-M3 Qdrant to find nearest neighbours of the viewed product
    vectors — a lightweight collaborative filtering signal.
    Requires the Qdrant server to be running.
    """
    profile = get_profile(session_id)
    viewed  = profile.get("viewed_product_ids", [])
    if not viewed:
        return []

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        from rufus.qdrant import get_client
        from rufus.retriever import Product

        client = get_client()

        # Fetch vectors for the viewed products
        results = client.retrieve(
            collection_name="rufus_products",
            ids=[],          # we'll use scroll + filter instead
            with_vectors=True,
            with_payload=True,
        )
        # Use scroll to find the viewed products by payload product_id
        candidate_vecs: list[list[float]] = []
        for pid in viewed[-5:]:   # use last 5 viewed
            pts, _ = client.scroll(
                collection_name="rufus_products",
                scroll_filter=Filter(must=[
                    FieldCondition(key="product_id", match=MatchAny(any=[pid]))
                ]),
                limit=1,
                with_vectors=True,
                with_payload=False,
            )
            if pts and pts[0].vector:
                candidate_vecs.append(pts[0].vector)

        if not candidate_vecs:
            return []

        # Average the viewed vectors → personal taste centroid
        import numpy as np
        centroid = np.mean(candidate_vecs, axis=0).tolist()

        hits = client.query_points(
            collection_name="rufus_products",
            query=centroid,
            limit=top_k * 2,
        )

        products = [
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
            if h.payload.get("product_id") not in viewed
        ]
        return products[:top_k]

    except Exception:
        return []


def preference_summary(session_id: str) -> str:
    profile = get_profile(session_id)
    brands  = sorted(profile.get("brand_prefs", {}).items(), key=lambda x: -x[1])[:3]
    cats    = sorted(profile.get("cat_prefs",   {}).items(), key=lambda x: -x[1])[:2]
    p_min   = profile.get("price_min", 0)
    p_max   = profile.get("price_max", 999)
    viewed  = profile.get("viewed_product_ids", [])

    parts = []
    if brands:
        parts.append(f"Preferred brands: {', '.join(b for b, _ in brands)}")
    if cats:
        parts.append(f"Often shops: {', '.join(c for c, _ in cats)}")
    if p_max < 900:
        parts.append(f"Typical price range: ${p_min:.0f}-${p_max:.0f}")
    if viewed:
        parts.append(f"Viewed {len(viewed)} product(s) this session")
    if profile.get("is_mock"):
        parts.append("(inferred preferences)")
    return " | ".join(parts) if parts else ""

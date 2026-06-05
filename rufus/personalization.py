"""
User personalization — preference tracking + rerank bias.

Storage: SQLite table in rufus_sc.db (same DB as inventory).
Profiles accumulate brand/category affinity from click history.
When no real history exists, synthetic seed profiles are used so
the feature works immediately without requiring user interaction.

Real personalization data (RetailRocket, Instacart) is on disk at
data/personalization/ but ingestion is not yet built; this module
provides a working mock that can be replaced with a real signal later.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rufus.inventory import get_db

# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    session_id   TEXT PRIMARY KEY,
    brand_prefs  TEXT DEFAULT '{}',   -- JSON: {brand: click_count}
    cat_prefs    TEXT DEFAULT '{}',   -- JSON: {category: click_count}
    price_min    REAL DEFAULT 0,
    price_max    REAL DEFAULT 999,
    updated_at   TEXT
);
"""

# Synthetic seed profiles — rotate by session hash so different users
# get different personas without requiring real history.
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
    """Return user preference profile, seeding a synthetic one if none exists."""
    _init()
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM user_profiles WHERE session_id = ?", (session_id,)
        ).fetchone()

    if row:
        return {
            "brand_prefs": json.loads(row["brand_prefs"] or "{}"),
            "cat_prefs":   json.loads(row["cat_prefs"]   or "{}"),
            "price_min":   row["price_min"],
            "price_max":   row["price_max"],
            "is_mock":     False,
        }

    # Seed a deterministic synthetic profile based on session hash
    seed = _SEED_PROFILES[hash(session_id) % len(_SEED_PROFILES)]
    return {**seed, "is_mock": True}


def update_profile(session_id: str, products: list) -> None:
    """Increment brand/category affinity for products shown to user."""
    if not products:
        return
    _init()
    profile = get_profile(session_id)
    brand_p = profile.get("brand_prefs", {})
    cat_p   = profile.get("cat_prefs",   {})

    for p in products:
        brand = getattr(p, "brand", None) or (p.get("brand") if isinstance(p, dict) else None)
        cat   = getattr(p, "category", None) or (p.get("category") if isinstance(p, dict) else None)
        if brand:
            brand_p[brand] = brand_p.get(brand, 0) + 1
        if cat:
            cat_p[cat] = cat_p.get(cat, 0) + 1

    from datetime import datetime
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO user_profiles
               (session_id, brand_prefs, cat_prefs, price_min, price_max, updated_at)
               VALUES (?,?,?,?,?,?)""",
            (
                session_id,
                json.dumps(brand_p),
                json.dumps(cat_p),
                profile.get("price_min", 0),
                profile.get("price_max", 999),
                datetime.utcnow().isoformat(),
            ),
        )


def apply_preference_bias(products: list, session_id: str, weight: float = 0.15) -> list:
    """
    Boost relevance scores for products matching user's brand/category preferences.
    weight=0.15 means a preferred brand can add up to 15% to the score.
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
            # Products are dataclass-like — set score via object copy or dict
            try:
                from dataclasses import replace
                p = replace(p, score=min(p.score + boost, 1.0))
            except Exception:
                pass
        boosted.append(p)

    boosted.sort(key=lambda x: getattr(x, "score", 0), reverse=True)
    return boosted


def preference_summary(session_id: str) -> str:
    """Human-readable summary of user preferences for LLM context."""
    profile = get_profile(session_id)
    brands = sorted(profile.get("brand_prefs", {}).items(), key=lambda x: -x[1])[:3]
    cats   = sorted(profile.get("cat_prefs",   {}).items(), key=lambda x: -x[1])[:2]
    p_min  = profile.get("price_min", 0)
    p_max  = profile.get("price_max", 999)

    parts = []
    if brands:
        parts.append(f"Preferred brands: {', '.join(b for b, _ in brands)}")
    if cats:
        parts.append(f"Often shops: {', '.join(c for c, _ in cats)}")
    if p_max < 900:
        parts.append(f"Typical price range: ${p_min:.0f}–${p_max:.0f}")
    if profile.get("is_mock"):
        parts.append("(inferred preferences)")
    return " | ".join(parts) if parts else ""

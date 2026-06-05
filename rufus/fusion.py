"""
Reciprocal Rank Fusion (RRF) for merging multiple ranked product lists.

RRF formula: score(d) = Σ_r  1 / (k + rank_r(d))
where k=60 is the standard constant that dampens the impact of high ranks.

When a product appears in only one list it still gets a score, so the fused
list gracefully handles the partial CLIP coverage (164 900 / 1.2 M products).
"""

from __future__ import annotations

import dataclasses

from rufus.retriever import Product

RRF_K = 60  # standard RRF constant


def rrf_fuse(result_lists: list[list[Product]], top_k: int = 5) -> list[Product]:
    """
    Merge multiple ranked Product lists with RRF.

    Each list is treated as one ranker. Products appearing in multiple lists
    accumulate higher scores. The returned list is sorted by fused score,
    truncated to top_k.
    """
    scores: dict[str, float] = {}
    best: dict[str, Product] = {}  # keep the product with the best individual score

    for results in result_lists:
        for rank, product in enumerate(results):
            pid = product.product_id
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
            if pid not in best or product.score > best[pid].score:
                best[pid] = product
            # Keep image_url from whichever source has one — CLIP has verified URLs
            # for 156K SQID products; BGE-M3 never stores image_url in Qdrant.
            if product.image_url and not best[pid].image_url:
                best[pid] = dataclasses.replace(best[pid], image_url=product.image_url)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        dataclasses.replace(best[pid], score=round(score, 4))
        for pid, score in fused
    ]

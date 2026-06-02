"""
NDCG evaluation utilities for the Rufus retrieval system.

ESCI relevance grades:
  E (Exact)       → gain 3  — the product exactly matches the query
  S (Substitute)  → gain 2  — acceptable substitute
  C (Complement)  → gain 1  — complements the query but doesn't satisfy it
  I (Irrelevant)  → gain 0  — irrelevant
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

ESCI_GAINS: dict[str, float] = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance: dict[str, float],
    k: int = 10,
) -> float:
    """
    Compute NDCG@K.

    retrieved_ids  – ordered list of product_ids from the retrieval system
    relevance      – {product_id: gain} ground-truth dict for this query
                     products not in the dict are treated as gain=0
    """
    gains = [relevance.get(pid, 0.0) for pid in retrieved_ids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class EvalResult:
    """Aggregate evaluation result over a set of queries."""
    system: str
    k: int
    scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def std(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        m = self.mean
        variance = sum((s - m) ** 2 for s in self.scores) / (len(self.scores) - 1)
        return math.sqrt(variance)

    @property
    def n(self) -> int:
        return len(self.scores)

    def __repr__(self) -> str:
        return (
            f"EvalResult({self.system!r}, NDCG@{self.k}="
            f"{self.mean:.4f} ± {self.std:.4f}, n={self.n})"
        )

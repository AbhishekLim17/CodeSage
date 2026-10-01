"""Reciprocal Rank Fusion of vector and keyword result lists."""

from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60  # the constant from the original RRF paper; larger = flatter, smaller = top ranks dominate


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Merge several best-first id lists into one; each id scores ``sum(weight / (k + rank))`` over the lists.

    Ranks are 1-based, so no score calibration between very different searchers (cosine vs BM25) is needed.
    An id repeated inside one list only counts at its first position. Ties are broken by best single rank,
    then by id, so the output is deterministic.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("need exactly one weight per ranking")
    if k <= 0 or any(w < 0 for w in weights):
        raise ValueError("k must be positive and weights non-negative")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[str] = set()
        rank = 0
        for item in ranking:
            if item in seen:
                continue
            seen.add(item)
            rank += 1
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            best_rank[item] = min(rank, best_rank.get(item, rank))
    return sorted(scores.items(), key=lambda pair: (-pair[1], best_rank[pair[0]], pair[0]))

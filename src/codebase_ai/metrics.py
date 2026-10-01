"""Retrieval metrics shared by the embedding benchmark and (from M2) the evaluation harness."""

from __future__ import annotations

import random
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from math import comb

DEFAULT_KS = (1, 3, 5, 10)


def unique_in_order(items: Iterable[str]) -> list[str]:
    """De-duplicate, keeping the first occurrence: chunk results become a ranked list of files."""
    return list(dict.fromkeys(items))


def first_gold_rank(ranked: Sequence[str], gold: Collection[str]) -> int | None:
    """1-based rank of the first item in ``ranked`` that is in ``gold``, or ``None`` if there isn't one."""
    for position, item in enumerate(ranked, start=1):
        if item in gold:
            return position
    return None


def summarize(ranks: Sequence[int | None], ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    """``hit@k`` (share of questions whose first gold item ranks within k) and ``mrr`` (reciprocal rank, cut at max k)."""
    if not ranks:
        return {**{f"hit@{k}": 0.0 for k in ks}, "mrr": 0.0}
    cutoff = max(ks)
    total = len(ranks)
    summary = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / total for k in ks}
    summary["mrr"] = sum(1 / r for r in ranks if r is not None and r <= cutoff) / total
    return summary


def reciprocal_rank(rank: int | None, cutoff: int = 10) -> float:
    """``1 / rank`` if the first gold item ranked within ``cutoff``, else 0."""
    return 1.0 / rank if rank is not None and rank <= cutoff else 0.0


@dataclass(frozen=True)
class PairedResult:
    """Config A compared with config B on the same questions."""

    n: int
    mrr_diff: float  # mean(RR_a - RR_b); positive means A is better
    ci_low: float  # 95% bootstrap confidence interval of that mean
    ci_high: float
    better: int  # questions where A ranks the gold file strictly higher than B
    worse: int
    tied: int
    sign_p: float  # two-sided exact sign test over the questions that differ

    @property
    def significant(self) -> bool:
        """The 95% interval excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0


def sign_test_p(better: int, worse: int) -> float:
    """Two-sided exact binomial (p = 0.5) sign test; ties are ignored."""
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(better, worse) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_comparison(
    ranks_a: Sequence[int | None],
    ranks_b: Sequence[int | None],
    *,
    cutoff: int = 10,
    resamples: int = 10_000,
    seed: int = 0,
) -> PairedResult:
    """Compare two configurations question by question.

    With a few dozen questions most gaps between configurations are noise; the bootstrap interval on the mean
    reciprocal-rank difference and the sign test on per-question wins and losses say whether a gap is more than that.
    """
    if len(ranks_a) != len(ranks_b):
        raise ValueError("both configurations must be scored on the same questions")
    if not ranks_a:
        raise ValueError("need at least one question")
    diffs = [reciprocal_rank(a, cutoff) - reciprocal_rank(b, cutoff) for a, b in zip(ranks_a, ranks_b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    better = sum(1 for d in diffs if d > 0)
    worse = sum(1 for d in diffs if d < 0)
    return PairedResult(
        n=n,
        mrr_diff=sum(diffs) / n,
        ci_low=means[int(0.025 * resamples)],
        ci_high=means[int(0.975 * resamples) - 1],
        better=better,
        worse=worse,
        tied=n - better - worse,
        sign_p=sign_test_p(better, worse),
    )

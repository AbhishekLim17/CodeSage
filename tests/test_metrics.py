from __future__ import annotations

import pytest

from codebase_ai.metrics import first_gold_rank, summarize, unique_in_order


def test_unique_in_order_keeps_first_occurrence():
    assert unique_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert unique_in_order([]) == []


def test_first_gold_rank_is_one_based_and_any_of_gold():
    assert first_gold_rank(["x", "y", "z"], {"z", "y"}) == 2
    assert first_gold_rank(["x"], {"x"}) == 1
    assert first_gold_rank(["x", "y"], {"q"}) is None
    assert first_gold_rank([], {"q"}) is None


def test_summarize_hit_rates_and_mrr():
    result = summarize([1, 2, 4, None])
    assert result["hit@1"] == pytest.approx(0.25)
    assert result["hit@3"] == pytest.approx(0.5)
    assert result["hit@5"] == pytest.approx(0.75)
    assert result["hit@10"] == pytest.approx(0.75)
    assert result["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4 + 0) / 4)


def test_mrr_ignores_ranks_beyond_the_largest_k():
    assert summarize([11], ks=(1, 10))["mrr"] == 0.0
    assert summarize([10], ks=(1, 10))["mrr"] == pytest.approx(0.1)


def test_summarize_of_nothing_is_zero():
    assert summarize([]) == {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "hit@10": 0.0, "mrr": 0.0}


# --- paired comparison --------------------------------------------------------------------------------------


def test_reciprocal_rank():
    from codebase_ai.metrics import reciprocal_rank

    assert reciprocal_rank(1) == 1.0 and reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0 and reciprocal_rank(11) == 0.0 and reciprocal_rank(11, cutoff=20) == pytest.approx(1 / 11)


def test_identical_configs_show_no_difference():
    from codebase_ai.metrics import paired_comparison

    result = paired_comparison([1, 2, None, 5], [1, 2, None, 5])
    assert (result.mrr_diff, result.better, result.worse, result.tied, result.sign_p) == (0.0, 0, 0, 4, 1.0)
    assert (result.ci_low, result.ci_high) == (0.0, 0.0)
    assert not result.significant


def test_a_config_that_always_wins_is_significant():
    from codebase_ai.metrics import paired_comparison

    a = [1] * 12
    b = [3] * 12
    result = paired_comparison(a, b)
    assert result.mrr_diff == pytest.approx(1 - 1 / 3)
    assert (result.better, result.worse, result.tied, result.n) == (12, 0, 0, 12)
    assert result.ci_low > 0 and result.significant
    assert result.sign_p == pytest.approx(2 / 2**12)


def test_a_balanced_split_is_not_significant_and_the_interval_contains_the_mean():
    from codebase_ai.metrics import paired_comparison

    a = [1, 3, 1, 3, 2, 2, 1, 3]
    b = [3, 1, 3, 1, 2, 2, 3, 1]
    result = paired_comparison(a, b)
    assert (result.better, result.worse, result.tied) == (3, 3, 2) and result.sign_p == 1.0
    assert result.ci_low <= result.mrr_diff <= result.ci_high
    assert not result.significant


def test_a_miss_counts_as_zero_and_ties_are_ignored_by_the_sign_test():
    from codebase_ai.metrics import paired_comparison

    result = paired_comparison([1, None, 2], [None, None, 2])
    assert (result.better, result.worse, result.tied) == (1, 0, 2)
    assert result.mrr_diff == pytest.approx(1 / 3)
    assert result.sign_p == 1.0  # a single differing question cannot be significant


def test_paired_comparison_is_deterministic_for_a_seed_and_validates_input():
    from codebase_ai.metrics import paired_comparison

    a, b = [1, 2, 3, None, 1, 4], [2, 2, 1, 1, None, 4]
    assert paired_comparison(a, b, seed=7) == paired_comparison(a, b, seed=7)
    with pytest.raises(ValueError, match="same questions"):
        paired_comparison([1], [1, 2])
    with pytest.raises(ValueError, match="at least one"):
        paired_comparison([], [])


@pytest.mark.parametrize(("better", "worse", "expected"), [(0, 0, 1.0), (5, 5, 1.0), (10, 0, 2 / 2**10), (1, 0, 1.0), (3, 1, 0.625)])
def test_sign_test_p(better, worse, expected):
    from codebase_ai.metrics import sign_test_p

    assert sign_test_p(better, worse) == pytest.approx(expected)

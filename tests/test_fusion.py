from __future__ import annotations

import pytest

from codebase_ai.retrieval.fusion import RRF_K, reciprocal_rank_fusion


def ids(fused):
    return [item for item, _ in fused]


def test_a_single_list_keeps_its_order_with_reciprocal_scores():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert ids(fused) == ["a", "b", "c"]
    assert [score for _, score in fused] == pytest.approx([1 / (RRF_K + r) for r in (1, 2, 3)])


def test_agreement_between_lists_beats_a_single_high_rank():
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "z"]])
    assert ids(fused) == ["y", "x", "z"]  # y is 2nd + 1st, x only 1st, z only 2nd


def test_weights_scale_each_list():
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], weights=[3.0, 1.0]))
    assert fused["a"] == pytest.approx(3 / (RRF_K + 1))
    assert fused["b"] == pytest.approx(1 / (RRF_K + 1))


def test_a_zero_weight_list_is_ignored_for_ordering():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], weights=[1.0, 0.0])
    assert ids(fused) == ["a", "b"]


def test_ties_are_broken_by_best_rank_then_id_so_output_is_deterministic():
    # a and b tie on score (each rank 1 in one list); id order decides
    assert ids(reciprocal_rank_fusion([["b"], ["a"]])) == ["a", "b"]
    assert ids(reciprocal_rank_fusion([["a"], ["b"]])) == ["a", "b"]


def test_repeats_inside_one_list_only_count_once_at_their_first_position():
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]]))
    assert fused["a"] == pytest.approx(1 / (RRF_K + 1))
    assert fused["b"] == pytest.approx(1 / (RRF_K + 2))  # rank 2, not 3


def test_smaller_k_lets_a_single_top_rank_beat_broad_agreement():
    # a: 1st in one list only. q: 5th in both lists. a beats q exactly when 1/(k+1) > 2/(k+5), i.e. k < 3.
    # (g1 is also 1st in its list and ties with a; ties go to the smaller id, so a wins that too.)
    lists = [["a", "f1", "f2", "f3", "q"], ["g1", "g2", "g3", "g4", "q"]]
    assert ids(reciprocal_rank_fusion(lists, k=1))[0] == "a"
    assert ids(reciprocal_rank_fusion(lists, k=RRF_K))[0] == "q"


def test_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert ids(reciprocal_rank_fusion([[], ["a"]])) == ["a"]


@pytest.mark.parametrize(
    ("kwargs", "lists"),
    [
        ({"weights": [1.0]}, [["a"], ["b"]]),
        ({"weights": [-1.0, 1.0]}, [["a"], ["b"]]),
        ({"k": 0}, [["a"]]),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, lists):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(lists, **kwargs)

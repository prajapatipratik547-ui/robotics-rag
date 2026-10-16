"""Retrieval metrics and bootstrap intervals in eval/run_eval.py (needs requirements-eval.txt)."""

import pytest

pytest.importorskip("ragas")

from eval.run_eval import CONFIGS, confidence_intervals, fmt_score, score_retrieval  # noqa: E402


def test_score_retrieval_hit_mrr_recall_precision():
    scores = score_retrieval(["x", "a", "y", "b", "z"], ["a", "b", "c"])
    assert scores["first_rank"] == 2
    assert scores["hit"] == 1.0
    assert scores["mrr"] == 0.5
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["precision"] == pytest.approx(2 / 5)


def test_score_retrieval_miss():
    scores = score_retrieval(["x", "y"], ["a"])
    assert scores["first_rank"] is None and scores["hit"] == 0.0 and scores["mrr"] == 0.0


def fake_retrieval(ranks_by_mode: dict[str, list[int | None]]) -> dict:
    return {
        mode: {f"q{i}": {"hit": 1.0 if r else 0.0, "mrr": 1 / r if r else 0.0} for i, r in enumerate(ranks)}
        for mode, ranks in ranks_by_mode.items()
    }


def test_identical_modes_give_a_zero_difference():
    ranks = [1, 2, None, 1, 3] * 6
    retrieval = fake_retrieval({mode: ranks for mode in CONFIGS})
    testset = [{"id": f"q{i}"} for i in range(len(ranks))]
    cis = confidence_intervals(retrieval, testset)
    for row in cis["differences"]:
        assert row["hit"]["diff"] == 0 and row["hit"]["ci"] == [0.0, 0.0]


def test_a_large_consistent_gain_is_clear():
    n = 40
    retrieval = fake_retrieval({"dense": [None] * n, "hybrid": [1] * n, "hybrid+rerank": [1] * n})
    cis = confidence_intervals(retrieval, [{"id": f"q{i}"} for i in range(n)])
    hybrid_vs_dense = next(r for r in cis["differences"] if r["modes"] == ["hybrid", "dense"])
    assert hybrid_vs_dense["hit"]["ci"][0] > 0
    low, high = cis["modes"]["dense"]["hit"]
    assert low <= 0.0 <= high


def test_fmt_score():
    assert fmt_score(0.95, "hit") == "95%"
    assert fmt_score(0.12, "hit", signed=True) == "+12"
    assert fmt_score(-0.0234, "mrr", signed=True) == "-0.023"

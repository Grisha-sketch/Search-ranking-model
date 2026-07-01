"""
tests/test_evaluate.py
----------------------
Unit tests for all evaluation metrics in evaluate.py.
Run with: pytest tests/test_evaluate.py -v
"""

import pandas as pd
import pytest

from src.evaluate import (
    average_precision,
    compare_rankings,
    dcg_at_k,
    evaluate_ranking,
    ndcg_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# DCG
# ---------------------------------------------------------------------------

class TestDCG:
    def test_perfect_top1(self):
        """Single relevant doc at position 1."""
        assert dcg_at_k([1, 0, 0], k=3) == pytest.approx(1.0, rel=1e-4)

    def test_relevant_at_position2(self):
        """Relevant doc at position 2 gets log2(3) discount."""
        # position 2: gain=1, discount=log2(3)
        dcg = dcg_at_k([0, 1, 0], k=3)
        assert dcg == pytest.approx(1.0 / 1.5849625, rel=1e-4)

    def test_all_zeros(self):
        assert dcg_at_k([0, 0, 0], k=3) == 0.0

    def test_k_truncation(self):
        """DCG@2 should only consider first 2 positions."""
        assert dcg_at_k([1, 0, 1], k=2) == dcg_at_k([1, 0], k=2)

    def test_empty(self):
        assert dcg_at_k([], k=5) == 0.0


# ---------------------------------------------------------------------------
# NDCG
# ---------------------------------------------------------------------------

class TestNDCG:
    def test_perfect_ranking(self):
        """Ideal ordering should give NDCG = 1.0."""
        assert ndcg_at_k([1, 1, 0, 0], k=4) == pytest.approx(1.0, rel=1e-4)

    def test_no_relevant(self):
        """No relevant docs → NDCG = 0."""
        assert ndcg_at_k([0, 0, 0], k=3) == 0.0

    def test_reversed_ordering(self):
        """Worst ordering should give NDCG < 1.0."""
        assert ndcg_at_k([0, 0, 1, 1], k=4) < 1.0

    def test_ndcg_between_0_and_1(self):
        relevances = [0, 1, 0, 1, 0]
        score = ndcg_at_k(relevances, k=5)
        assert 0.0 <= score <= 1.0

    def test_single_relevant_first(self):
        assert ndcg_at_k([1], k=1) == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Average Precision
# ---------------------------------------------------------------------------

class TestAP:
    def test_perfect(self):
        assert average_precision([1, 1, 0, 0]) == pytest.approx(1.0, rel=1e-4)

    def test_no_relevant(self):
        assert average_precision([0, 0, 0]) == 0.0

    def test_single_relevant_at_rank1(self):
        assert average_precision([1, 0, 0]) == pytest.approx(1.0, rel=1e-4)

    def test_single_relevant_at_rank2(self):
        assert average_precision([0, 1, 0]) == pytest.approx(0.5, rel=1e-4)

    def test_two_relevant(self):
        # P@1=1/1 (hit), P@3=2/3 (hit) → AP = (1 + 2/3) / 2
        assert average_precision([1, 0, 1]) == pytest.approx( (1.0 + 2 / 3) / 2, rel=1e-4)


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------

class TestMRR:
    def test_relevant_at_rank1(self):
        assert reciprocal_rank([1, 0, 0]) == pytest.approx(1.0, rel=1e-4)

    def test_relevant_at_rank2(self):
        assert reciprocal_rank([0, 1, 0]) == pytest.approx(0.5, rel=1e-4)

    def test_no_relevant(self):
        assert reciprocal_rank([0, 0, 0]) == 0.0

    def test_takes_first_relevant(self):
        """MRR uses only the first relevant hit."""
        assert reciprocal_rank([0, 0, 1, 1]) == pytest.approx(1 / 3, rel=1e-4)


# ---------------------------------------------------------------------------
# evaluate_ranking (batch)
# ---------------------------------------------------------------------------

class TestEvaluateRanking:
    def _make_df(self, scores, labels, qids=None):
        n = len(scores)
        if qids is None:
            qids = ["q1"] * n
        return pd.DataFrame({"qid": qids, "score": scores, "label": labels})

    def test_perfect_single_query(self):
        df = self._make_df([3, 2, 1, 0], [1, 1, 0, 0])
        metrics = evaluate_ranking(df, k_values=[2])
        assert metrics["ndcg@2"] == pytest.approx(1.0, rel=1e-4)

    def test_keys_present(self):
        df = self._make_df([1, 0], [1, 0])
        metrics = evaluate_ranking(df, k_values=[1, 5])
        assert "ndcg@1" in metrics
        assert "ndcg@5" in metrics
        assert "map" in metrics
        assert "mrr" in metrics

    def test_multi_query(self):
        df = self._make_df(
            scores=[2, 1, 2, 1],
            labels=[1, 0, 0, 1],
            qids=["q1", "q1", "q2", "q2"],
        )
        metrics = evaluate_ranking(df, k_values=[1])
        # q1: rank1 relevant → ndcg@1=1.0; q2: rank1 not relevant → ndcg@1=0.0
        assert metrics["ndcg@1"] == pytest.approx(0.5, rel=1e-4)

    def test_empty_df(self):
        df = pd.DataFrame({"qid": [], "score": [], "label": []})
        assert evaluate_ranking(df) == {}


# ---------------------------------------------------------------------------
# compare_rankings
# ---------------------------------------------------------------------------

class TestCompareRankings:
    def test_shape(self):
        df = pd.DataFrame({
            "qid": ["q1", "q1", "q1"],
            "bm25_score": [3, 2, 1],
            "score": [1, 2, 3],
            "label": [1, 0, 0],
        })
        result = compare_rankings(df, k_values=[1])
        assert "baseline (BM25)" in result.columns
        assert "model (LTR)" in result.columns
        assert "delta" in result.columns
        
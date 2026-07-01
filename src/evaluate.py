"""
evaluate.py
-----------
Ranking evaluation metrics.

Metrics implemented:
    - NDCG@k  (Normalized Discounted Cumulative Gain) — primary metric
    - MAP      (Mean Average Precision)
    - MRR      (Mean Reciprocal Rank)

All functions accept ranked lists of relevance labels and return floats.
The `evaluate_ranking` function is the main entry point used by the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def dcg_at_k(relevances: list[int], k: int) -> float:
    """
    Discounted Cumulative Gain @ k.

    DCG@k = Σ (2^rel_i - 1) / log2(i + 2)   for i in 0..k-1

    Uses the standard gain formula where position 1 has no discount.
    """
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    gains = np.array([(2 ** r - 1) for r in relevances], dtype=float)
    discounts = np.log2(np.arange(2, len(gains) + 2))  # log2(2), log2(3), ...
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """
    Normalized DCG @ k.

    NDCG@k = DCG@k / IDCG@k

    IDCG is the DCG of the ideal (perfectly sorted) ranking.
    Returns 0.0 if there are no relevant documents.
    """
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg


def average_precision(relevances: list[int]) -> float:
    """
    Average Precision for a single query.

    AP = (1 / R) * Σ P@k * rel_k

    where R = total number of relevant docs, P@k = precision at cut k,
    rel_k = 1 if doc at position k is relevant.
    """
    num_relevant = sum(1 for r in relevances if r > 0)
    if num_relevant == 0:
        return 0.0

    ap = 0.0
    hits = 0
    for i, rel in enumerate(relevances):
        if rel > 0:
            hits += 1
            ap += hits / (i + 1)
    return ap / num_relevant


def reciprocal_rank(relevances: list[int]) -> float:
    """
    Reciprocal Rank for a single query.

    RR = 1 / rank_of_first_relevant_doc
    Returns 0.0 if no relevant doc is found.
    """
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Batch evaluation over a DataFrame
# ---------------------------------------------------------------------------

def evaluate_ranking(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    qid_col: str = "qid",
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """
    Evaluate a ranked DataFrame grouped by query.

    Args:
        df:         DataFrame with columns [qid, score, label]
        score_col:  column containing the model's predicted relevance score
        label_col:  column containing ground-truth relevance (0/1 or graded)
        qid_col:    column identifying the query
        k_values:   list of cutoffs for NDCG@k

    Returns:
        Dict of metric_name -> mean value across all queries.
        Example: {'ndcg@1': 0.72, 'ndcg@5': 0.65, 'map': 0.58, 'mrr': 0.70}
    """
    ndcg_sums: dict[int, float] = {k: 0.0 for k in k_values}
    map_sum = 0.0
    mrr_sum = 0.0
    num_queries = 0

    for qid, group in df.groupby(qid_col):
        # Sort by score descending (model's predicted ranking)
        ranked = group.sort_values(score_col, ascending=False)
        relevances = ranked[label_col].tolist()

        for k in k_values:
            ndcg_sums[k] += ndcg_at_k(relevances, k)

        map_sum += average_precision(relevances)
        mrr_sum += reciprocal_rank(relevances)
        num_queries += 1

    if num_queries == 0:
        return {}

    results: dict[str, float] = {}
    for k in k_values:
        results[f"ndcg@{k}"] = round(ndcg_sums[k] / num_queries, 4)
    results["map"] = round(map_sum / num_queries, 4)
    results["mrr"] = round(mrr_sum / num_queries, 4)

    return results


def compare_rankings(
    df: pd.DataFrame,
    baseline_score_col: str = "bm25_score",
    model_score_col: str = "score",
    label_col: str = "label",
    qid_col: str = "qid",
    k_values: list[int] = [1, 5, 10],
) -> pd.DataFrame:
    """
    Side-by-side comparison of baseline vs model metrics.

    Returns a DataFrame with rows = metrics, columns = [baseline, model, delta].
    """
    baseline_metrics = evaluate_ranking(df, baseline_score_col, label_col, qid_col, k_values)
    model_metrics = evaluate_ranking(df, model_score_col, label_col, qid_col, k_values)

    rows = []
    for metric in baseline_metrics:
        b = baseline_metrics[metric]
        m = model_metrics.get(metric, 0.0)
        rows.append({
            "metric": metric,
            "baseline (BM25)": b,
            "model (LTR)": m,
            "delta": round(m - b, 4),
            "delta %": f"{((m - b) / b * 100):+.1f}%" if b > 0 else "N/A",
        })
    return pd.DataFrame(rows).set_index("metric")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate a query with 5 retrieved docs, 2 relevant
    relevances = [1, 0, 1, 0, 0]

    print(f"DCG@5:  {dcg_at_k(relevances, 5):.4f}")
    print(f"NDCG@5: {ndcg_at_k(relevances, 5):.4f}")
    print(f"AP:     {average_precision(relevances):.4f}")
    print(f"RR:     {reciprocal_rank(relevances):.4f}")

    # Perfect ranking: relevant docs first
    perfect = [1, 1, 0, 0, 0]
    print(f"\nPerfect NDCG@5: {ndcg_at_k(perfect, 5):.4f}")  # should be 1.0

    # Worst ranking: relevant docs last
    worst = [0, 0, 0, 1, 1]
    print(f"Worst   NDCG@5: {ndcg_at_k(worst, 5):.4f}")
    
"""
user_signals.py
---------------
Simulates user interaction signals and applies position bias correction.

The core challenge with click data:
    Users click results at the top more often — not because they're more
    relevant, but because they're more visible. A model trained on raw
    clicks will learn to rank popular positions higher, not relevant docs.

    This is called **position bias**.

Solution: Inverse Propensity Scoring (IPS)
    Weight each click by 1/P(examined | position) so that clicks from
    lower positions count more (they're rarer and thus more informative).

Signals simulated:
    click           binary — did the user click this result?
    dwell_time      seconds spent on the page after clicking
    skip            user saw the result but didn't click (soft negative)
    examination     did the user examine this position? (latent variable)

Signal features added to DataFrame:
    click_rate          per (query, passage) empirical CTR across sessions
    avg_dwell_time      mean dwell time for clicked (query, passage) pairs
    skip_rate           fraction of impressions where result was skipped
    ips_weight          inverse propensity score for debiasing
    weighted_click      click * ips_weight  (debiased signal for training)
    dwell_score         normalized dwell time (0-1) capped at 5 minutes
    engagement_score    composite: 0.5*click_rate + 0.3*dwell_score + 0.2*(1-skip_rate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position bias model
# ---------------------------------------------------------------------------

def position_examination_probs(
    num_positions: int = 20,
    eta: float = 0.8,
) -> np.ndarray:
    """
    Probability that a user examines position k (1-indexed).

    Uses the standard cascade model:
        P(examined | position k) = (1 / k) ^ eta

    eta controls how steep the bias is:
        eta=0   → no position bias (uniform examination)
        eta=1   → strong bias (position 2 gets half the attention of position 1)
        eta=0.8 → empirically validated on web search logs (Joachims et al.)

    Args:
        num_positions: how many positions to compute
        eta:           position bias exponent

    Returns:
        np.ndarray of shape (num_positions,) with P(examined | position k)
        index 0 = position 1 (top result)
    """
    positions = np.arange(1, num_positions + 1, dtype=float)
    probs = (1.0 / positions) ** eta
    return probs


# ---------------------------------------------------------------------------
# Session simulator
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Controls how realistic the simulated click data is."""
    num_sessions: int = 200        # number of search sessions per query
    num_positions: int = 10        # results shown per page
    eta: float = 0.8               # position bias strength
    base_ctr_relevant: float = 0.6   # P(click | relevant, examined)
    base_ctr_irrelevant: float = 0.1  # P(click | not relevant, examined)
    mean_dwell_relevant: float = 45.0  # seconds on page if relevant
    mean_dwell_irrelevant: float = 8.0  # seconds on page if not relevant
    dwell_noise_std: float = 10.0  # gaussian noise on dwell time
    random_seed: int = 42


def simulate_sessions(
    df: pd.DataFrame,
    config: SimulationConfig = SimulationConfig(),
) -> pd.DataFrame:
    """
    Simulate user search sessions and generate click/dwell/skip signals.

    For each query in df:
        1. Rank the candidates by their BM25 score (simulates initial ranking)
        2. For each session, decide which positions the user examines
           (using the cascade examination model)
        3. For each examined position, simulate a click based on relevance
        4. For clicked results, simulate dwell time

    Args:
        df:     candidate DataFrame with columns [qid, pid, bm25_score, label]
        config: simulation hyperparameters

    Returns:
        DataFrame with per-(qid, pid) aggregated signal columns appended.
    """
    rng = np.random.default_rng(config.random_seed)
    exam_probs = position_examination_probs(config.num_positions, config.eta)

    logger.info(
        "Simulating %d sessions per query across %d unique queries...",
        config.num_sessions,
        df["qid"].nunique(),
    )

    # Accumulate raw events: list of dicts {qid, pid, clicks, impressions, dwell_total, skips}
    signal_rows: list[dict] = []

    for qid, group in df.groupby("qid"):
        # Rank by BM25 — this is the order users see results
        ranked = group.sort_values("bm25_score", ascending=False).reset_index(drop=True)
        pids = ranked["pid"].tolist()
        labels = ranked["label"].tolist()

        num_shown = min(len(pids), config.num_positions)

        # Per-passage accumulators
        clicks = {pid: 0 for pid in pids}
        impressions = {pid: 0 for pid in pids}
        dwell_totals = {pid: 0.0 for pid in pids}
        skips = {pid: 0 for pid in pids}

        for _ in range(config.num_sessions):
            for pos in range(num_shown):
                pid = pids[pos]
                rel = labels[pos]

                # Does the user examine this position?
                examined = rng.random() < exam_probs[pos]
                if not examined:
                    continue

                impressions[pid] += 1

                # Does the user click?
                p_click = (
                    config.base_ctr_relevant if rel > 0
                    else config.base_ctr_irrelevant
                )
                clicked = rng.random() < p_click

                if clicked:
                    clicks[pid] += 1
                    # Simulate dwell time
                    mean_dwell = (
                        config.mean_dwell_relevant if rel > 0
                        else config.mean_dwell_irrelevant
                    )
                    dwell = float(rng.normal(mean_dwell, config.dwell_noise_std))
                    dwell_totals[pid] += max(0.0, dwell)
                else:
                    skips[pid] += 1

        # Build per-(qid, pid) signal rows
        for pos, pid in enumerate(pids):
            imp = impressions[pid]
            clk = clicks[pid]
            signal_rows.append({
                "qid": qid,
                "pid": pid,
                "position": pos + 1,
                "_clicks": clk,
                "_impressions": imp,
                "_dwell_total": dwell_totals[pid],
                "_skips": skips[pid],
            })

    signals_df = pd.DataFrame(signal_rows)

    # Merge back onto original df
    df = df.merge(signals_df, on=["qid", "pid"], how="left")

    # Fill NaN for passages that never appeared (shouldn't happen, but safe)
    for col in ["_clicks", "_impressions", "_dwell_total", "_skips"]:
        df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# IPS debiasing
# ---------------------------------------------------------------------------

def compute_ips_weights(
    df: pd.DataFrame,
    eta: float = 0.8,
    clip_max: float = 10.0,
) -> pd.Series:
    """
    Compute Inverse Propensity Score (IPS) weights per row.

    IPS corrects for position bias by upweighting clicks from lower positions:

        IPS weight = 1 / P(examined | position)
                   = position ^ eta

    High position number → low examination probability → high IPS weight.
    This means a click at position 8 is worth more than a click at position 1.

    Args:
        df:       DataFrame with a 'position' column (1-indexed)
        eta:      position bias exponent (should match simulation)
        clip_max: cap IPS weights to avoid extreme values at low positions

    Returns:
        pd.Series of IPS weights, same index as df.
    """
    positions = df["position"].clip(lower=1).astype(float)
    weights = positions ** eta
    return weights.clip(upper=clip_max)


# ---------------------------------------------------------------------------
# Aggregate signals into features
# ---------------------------------------------------------------------------

def compute_signal_features(df: pd.DataFrame, eta: float = 0.8) -> pd.DataFrame:
    """
    Convert raw event counts into normalized signal features.

    Adds these columns to df:
        click_rate          clicks / impressions  (0.0 if no impressions)
        avg_dwell_time      total_dwell / clicks  (0.0 if no clicks)
        skip_rate           skips / impressions
        ips_weight          inverse propensity score for this position
        weighted_click      click_rate * ips_weight  (debiased)
        dwell_score         avg_dwell_time normalized to [0, 1], capped at 300s
        engagement_score    composite relevance signal from all signals

    Args:
        df:   output of simulate_sessions() with _clicks, _impressions, etc.
        eta:  position bias exponent for IPS

    Returns:
        df with signal feature columns appended.
    """
    df = df.copy()

    imp = df["_impressions"].clip(lower=0)
    clk = df["_clicks"].clip(lower=0)

    # Raw rates
    df["click_rate"] = np.where(imp > 0, clk / imp, 0.0)
    df["avg_dwell_time"] = np.where(clk > 0, df["_dwell_total"] / clk, 0.0)
    df["skip_rate"] = np.where(imp > 0, df["_skips"] / imp, 0.0)

    # IPS debiasing
    df["ips_weight"] = compute_ips_weights(df, eta=eta)
    df["weighted_click"] = df["click_rate"] * df["ips_weight"]

    # Dwell score: normalize to [0, 1], cap at 5 minutes (300s)
    df["dwell_score"] = (df["avg_dwell_time"].clip(upper=300.0) / 300.0)

    # Composite engagement score
    df["engagement_score"] = (
        0.5 * df["click_rate"]
        + 0.3 * df["dwell_score"]
        + 0.2 * (1.0 - df["skip_rate"])
    )

    # Drop raw accumulators — model doesn't need them
    df = df.drop(columns=["_clicks", "_impressions", "_dwell_total", "_skips"])

    logger.info("Signal features computed. New columns: click_rate, avg_dwell_time, "
                "skip_rate, ips_weight, weighted_click, dwell_score, engagement_score")
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

USER_SIGNAL_COLS = [
    "click_rate",
    "avg_dwell_time",
    "skip_rate",
    "ips_weight",
    "weighted_click",
    "dwell_score",
    "engagement_score",
]


def add_user_signals(
    df: pd.DataFrame,
    config: SimulationConfig = SimulationConfig(),
) -> pd.DataFrame:
    """
    Full pipeline: simulate sessions → compute IPS → return signal features.

    Args:
        df:     candidate DataFrame (output of features.extract_features)
        config: simulation config

    Returns:
        df with USER_SIGNAL_COLS appended.
    """
    df = simulate_sessions(df, config)
    df = compute_signal_features(df, eta=config.eta)
    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal test: 2 queries, 4 candidates each
    sample_df = pd.DataFrame([
        {"qid": "q1", "pid": "p1", "query_text": "machine learning", "bm25_score": 4.2, "label": 1},
        {"qid": "q1", "pid": "p2", "query_text": "machine learning", "bm25_score": 3.1, "label": 0},
        {"qid": "q1", "pid": "p3", "query_text": "machine learning", "bm25_score": 1.8, "label": 1},
        {"qid": "q1", "pid": "p4", "query_text": "machine learning", "bm25_score": 0.5, "label": 0},
        {"qid": "q2", "pid": "p5", "query_text": "python sorting", "bm25_score": 5.0, "label": 1},
        {"qid": "q2", "pid": "p6", "query_text": "python sorting", "bm25_score": 2.0, "label": 0},
        {"qid": "q2", "pid": "p7", "query_text": "python sorting", "bm25_score": 0.8, "label": 1},
        {"qid": "q2", "pid": "p8", "query_text": "python sorting", "bm25_score": 0.2, "label": 0},
    ])

    config = SimulationConfig(num_sessions=500, random_seed=42)
    result = add_user_signals(sample_df, config)

    print("\n--- User signal features ---")
    print(result[["pid", "label", "position"] + USER_SIGNAL_COLS].to_string(index=False))

    print("\n--- Position bias check ---")
    print("Higher positions should have higher IPS weights (more debiasing needed):")
    for _, row in result[result["qid"] == "q1"].iterrows():
        print(f"  pos={int(row['position'])}  ips={row['ips_weight']:.3f}  "
              f"click_rate={row['click_rate']:.3f}  weighted_click={row['weighted_click']:.3f}  "
              f"label={int(row['label'])}")

    print("\n--- Sanity checks ---")
    q1 = result[result["qid"] == "q1"]
    rel = q1[q1["label"] == 1]
    irrel = q1[q1["label"] == 0]
    print(f"Avg click_rate (relevant):   {rel['click_rate'].mean():.3f}")
    print(f"Avg click_rate (irrelevant): {irrel['click_rate'].mean():.3f}")
    assert rel["click_rate"].mean() > irrel["click_rate"].mean(), "Relevant docs shoud've higher CTR!"
    print("Assertion passed: relevant docs have higher click_rate.")

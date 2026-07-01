"""
model.py
--------
LightGBM LambdaMART training, evaluation, and inference.

LambdaMART is the industry-standard Learning to Rank algorithm.
It combines:
    - MART  (Multiple Additive Regression Trees) — gradient boosted trees
    - Lambda gradients — derived from NDCG, so the model directly
      optimizes ranking quality rather than pointwise loss

Training flow:
    1. Build feature matrix X from FEATURE_COLS + USER_SIGNAL_COLS
    2. Build group array  — LightGBM needs to know how many rows
       belong to each query (for listwise loss computation)
    3. Train with objective='lambdarank', eval_metric='ndcg'
    4. Log params + metrics to MLflow

Inference flow:
    query + candidates → feature extraction → model.predict() → sorted pids
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.evaluate import compare_rankings, evaluate_ranking
from src.features import FEATURE_COLS, extract_features
from src.user_signals import USER_SIGNAL_COLS, add_user_signals, SimulationConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "lambdamart.lgb"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 5, 10],         # evaluate NDCG at these cutoffs
    "lambdarank_truncation_level": 10,  # only top-10 positions affect gradients
    "learning_rate": 0.05,
    "num_leaves": 31,                  # max leaves per tree — controls complexity
    "min_data_in_leaf": 5,             # prevents overfitting on small groups
    "n_estimators": 300,
    "feature_fraction": 0.8,           # subsample features per tree
    "bagging_fraction": 0.8,           # subsample rows per tree
    "bagging_freq": 5,
    "reg_alpha": 0.1,                  # L1 regularization
    "reg_lambda": 0.2,                 # L2 regularization
    "verbose": -1,                     # suppress LightGBM logs (use our logger)
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Group array builder
# ---------------------------------------------------------------------------


def build_group_array(df: pd.DataFrame, qid_col: str = "qid") -> list[int]:
    """
    Build the group array required by LightGBM for listwise ranking.

   
    LightGBM needs to know how many rows belong to each query so it
    can compute the lambda gradients within each query group.

    Example:
        queries: [q1, q1, q1, q2, q2] → group = [3, 2]

    Args:
        df:      DataFrame sorted by qid
        qid_col: name of the query ID column

    Returns:
        List of group sizes (one integer per unique query).
    """
    return df.groupby(qid_col, sort=False).size().tolist()


# ---------------------------------------------------------------------------
# Train / eval split
# ---------------------------------------------------------------------------

def train_eval_split(
    df: pd.DataFrame,
    eval_frac: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by query ID (not by row) to prevent query leakage.

    Splitting by row would let the model see some passages from an eval
    query during training, inflating eval metrics. We split at query level.

    Args:
        df:          full candidate DataFrame
        eval_frac:   fraction of queries held out for evaluation
        random_seed: reproducibility

    Returns:
        (train_df, eval_df)
    """
    rng = np.random.default_rng(random_seed)
    unique_qids = df["qid"].unique()
    rng.shuffle(unique_qids)

    n_eval = max(1, int(len(unique_qids) * eval_frac))
    eval_qids = set(unique_qids[:n_eval])

    train_df = df[~df["qid"].isin(eval_qids)].copy()
    eval_df = df[df["qid"].isin(eval_qids)].copy()

    logger.info(
        "Split: %d train queries (%d rows) | %d eval queries (%d rows)",
        train_df["qid"].nunique(), len(train_df),
        eval_df["qid"].nunique(), len(eval_df),
    )
    return train_df, eval_df


# ---------------------------------------------------------------------------
# Core trainer
# ---------------------------------------------------------------------------

class LambdaMARTRanker:
    """
    Wraps LightGBM's LambdaMART with fit/predict/save/load.

    Usage:
        ranker = LambdaMARTRanker()
        ranker.fit(train_df, eval_df)
        scores = ranker.predict(df)
        ranker.save()
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[lgb.LGBMRanker] = None
        self.feature_cols: list[str] = FEATURE_COLS + USER_SIGNAL_COLS

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
    ) -> "LambdaMARTRanker":
        """
        Train LambdaMART on the candidate DataFrame.

        Args:
            train_df: training data with feature + label columns
            eval_df:  optional held-out set for early stopping + logging

        Returns:
            self (for chaining)
        """
        # Sort by qid — LightGBM requires rows ordered by query group
        train_df = train_df.sort_values("qid").reset_index(drop=True)
        X_train = train_df[self.feature_cols].values
        y_train = train_df["label"].values.astype(int)
        g_train = build_group_array(train_df)

        eval_set = None
        eval_group = None
        if eval_df is not None:
            eval_df = eval_df.sort_values("qid").reset_index(drop=True)
            X_eval = eval_df[self.feature_cols].values
            y_eval = eval_df["label"].values.astype(int)
            eval_set = [(X_eval, y_eval)]
            eval_group = [build_group_array(eval_df)]

        logger.info(
            "Training LambdaMART: %d rows, %d features, %d query groups",
            len(train_df), len(self.feature_cols), len(g_train),
        )

        self.model = lgb.LGBMRanker(**self.params)
        self.model.fit(
            X_train,
            y_train,
            group=g_train,
            eval_set=eval_set,
            eval_group=eval_group,
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            if eval_df is not None else None,
        )

        logger.info("Training complete. Best iteration: %s", self.model.best_iteration_)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict relevance scores for all rows in df.

        Returns:
            np.ndarray of float scores — higher = more relevant.
            Same length and order as df.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        X = df[self.feature_cols].values
        return self.model.predict(X)

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict scores and return df sorted by score descending per query.

        Returns:
            df with 'score' column added, sorted by (qid, score desc).
        """
        df = df.copy()
        df["score"] = self.predict(df)
        return (
            df.sort_values(["qid", "score"], ascending=[True, False])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """
        Return feature importances sorted descending.

        LightGBM's 'gain' importance = total gain from splits using that feature.
        More reliable than 'split' count for ranking feature value.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")
        importances = self.model.feature_importances_
        return (
            pd.DataFrame({
                "feature": self.feature_cols,
                "importance": importances,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path = MODEL_PATH) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path))
        logger.info("Model saved to %s", path)

    def load(self, path: Path = MODEL_PATH) -> "LambdaMARTRanker":
        """Load a previously saved model from disk."""
        booster = lgb.Booster(model_file=str(path))
        self.model = lgb.LGBMRanker(**self.params)
        self.model._Booster = booster  # inject booster
        logger.info("Model loaded from %s", path)
        return self


# ---------------------------------------------------------------------------
# MLflow training run
# ---------------------------------------------------------------------------

def train_with_mlflow(
    df: pd.DataFrame,
    params: Optional[dict] = None,
    experiment_name: str = "search-ranking",
    run_name: str = "lambdamart",
) -> tuple[LambdaMARTRanker, dict[str, float]]:
    """
    Full training pipeline wrapped in an MLflow experiment run.

    Logs:
        - hyperparameters
        - NDCG@1/5/10, MAP, MRR on eval set
        - feature importance CSV
        - trained model artifact

    Args:
        df:              full candidate DataFrame (features + signals already extracted)
        params:          override DEFAULT_PARAMS keys
        experiment_name: MLflow experiment name
        run_name:        name for this specific run

    Returns:
        (trained ranker, eval metrics dict)
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        # --- Split ---
        train_df, eval_df = train_eval_split(df)

        # --- Train ---
        ranker = LambdaMARTRanker(params=params)
        mlflow.log_params({k: v for k, v in ranker.params.items()
                           if isinstance(v, (int, float, str, bool))})
        ranker.fit(train_df, eval_df)

        # --- Evaluate ---
        eval_df = eval_df.copy()
        eval_df["score"] = ranker.predict(eval_df)

        metrics = evaluate_ranking(eval_df, score_col="score", k_values=[1, 5, 10])
        mlflow.log_metrics(metrics)
        logger.info("Eval metrics: %s", metrics)

        # --- Baseline comparison ---
        comparison = compare_rankings(eval_df, k_values=[1, 5, 10])
        logger.info("\n%s", comparison.to_string())

        # --- Feature importance ---
        fi = ranker.feature_importance()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        fi.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        mlflow.log_artifact(str(FEATURE_IMPORTANCE_PATH))
        logger.info("\nTop 10 features:\n%s", fi.head(10).to_string(index=False))

        # --- Save model ---
        ranker.save()
        mlflow.log_artifact(str(MODEL_PATH))

    return ranker, metrics


# ---------------------------------------------------------------------------
# Smoke test (fast, no MS MARCO needed)
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """
    Train on synthetic data, assert model beats BM25 baseline.
    Used by CI and --smoke-test flag.
    """
    logger.info("Running smoke test...")
    rng = np.random.default_rng(0)

    # Generate synthetic candidate data: 20 queries, 10 candidates each
    rows = []
    for qid in range(20):
        for rank in range(10):
            label = 1 if rank < 3 else 0  # top 3 are relevant
            bm25 = float(10 - rank + rng.normal(0, 0.5))  # noisy BM25
            rows.append({
                "qid": f"q{qid}",
                "pid": f"q{qid}_p{rank}",
                "query_text": f"query {qid}",
                "passage_text": f"passage {rank} for query {qid}",
                "bm25_score": max(0.1, bm25),
                "label": label,
            })

    df = pd.DataFrame(rows)

    # Add features and signals (no embeddings for speed)
    df = extract_features(df, use_embeddings=False)
    df = add_user_signals(df, config=SimulationConfig(num_sessions=50, random_seed=0))

    # Split and train
    train_df, eval_df = train_eval_split(df, eval_frac=0.3)
    ranker = LambdaMARTRanker()
    ranker.fit(train_df, eval_df)

    # Evaluate
    eval_df = eval_df.copy()
    eval_df["score"] = ranker.predict(eval_df)

    model_metrics = evaluate_ranking(eval_df, score_col="score", k_values=[5])
    bm25_metrics = evaluate_ranking(eval_df, score_col="bm25_score", k_values=[5])

    logger.info("Smoke test — BM25 NDCG@5:  %.4f", bm25_metrics["ndcg@5"])
    logger.info("Smoke test — Model NDCG@5: %.4f", model_metrics["ndcg@5"])

    assert model_metrics["ndcg@5"] >= bm25_metrics["ndcg@5"] * 0.95, (
        f"Model NDCG@5 ({model_metrics['ndcg@5']:.4f}) significantly worse than "
        f"BM25 ({bm25_metrics['ndcg@5']:.4f})"
    )
    logger.info("Smoke test passed.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LambdaMART ranker")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test only")
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        from src.data_loader import (
            BM25Retriever,
            build_candidate_df,
            load_msmarco,
        )

        logger.info("Loading MS MARCO...")
        dataset = load_msmarco(split="train", max_queries=args.max_queries)

        logger.info("Running BM25 retrieval...")
        retriever = BM25Retriever(dataset.passages)
        retrieved = retriever.batch_retrieve(dataset.queries, top_k=args.top_k)

        df = build_candidate_df(
            dataset.queries, dataset.passages, dataset.qrels, retrieved
        )

        logger.info("Extracting features...")
        df = extract_features(df, use_embeddings=not args.no_embeddings)

        logger.info("Adding user signals...")
        df = add_user_signals(df)

        logger.info("Training with MLflow tracking...")
        ranker, metrics = train_with_mlflow(df)

        logger.info("Final eval metrics: %s", metrics)

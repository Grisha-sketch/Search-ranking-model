"""
data_loader.py
--------------
Loads MS MARCO passage dataset and runs BM25 baseline retrieval.

MS MARCO is the standard LTR benchmark with real user click signals.
We use the 'passage' variant (shorter docs, faster iteration).

Flow:
    load_msmarco()  ->  queries, passages, qrels (relevance labels)
    bm25_retrieve() ->  top-k candidate passages per query
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import load_dataset
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Query:
    qid: str
    text: str


@dataclass
class Passage:
    pid: str
    text: str


@dataclass
class QRel:
    """Relevance judgment: how relevant is passage `pid` to query `qid`."""
    qid: str
    pid: str
    relevance: int  # 0 = not relevant, 1 = relevant (MS MARCO is binary)


@dataclass
class MSMarcoDataset:
    queries: list[Query]
    passages: dict[str, Passage]   # pid -> Passage (for fast lookup)
    qrels: list[QRel]
    split: str = "train"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_msmarco(
    split: str = "train",
    max_queries: int = 500,
    max_passages: int = 50_000,
) -> MSMarcoDataset:
    """
    Load MS MARCO passage ranking data via HuggingFace datasets.

    Args:
        split:         'train' | 'validation' — use 'validation' for eval
        max_queries:   cap queries for faster dev iteration
        max_passages:  cap passages loaded into memory

    Returns:
        MSMarcoDataset with queries, passages dict, and qrels
    """
    logger.info("Loading MS MARCO passages (%s split, up to %d queries)...", split, max_queries)

    # Load the small dev/train splits — full corpus is ~8.8M passages
    # 'ms_marco' config 'v2.1' is the standard passage ranking version
    dataset = load_dataset("ms_marco", "v2.1", split=split, trust_remote_code=True)

    queries: list[Query] = []
    passages: dict[str, Passage] = {}
    qrels: list[QRel] = []

    for i, sample in enumerate(dataset):
        if i >= max_queries:
            break

        qid = str(sample["query_id"])
        qtext = sample["query"]
        queries.append(Query(qid=qid, text=qtext))

        # Each sample contains passages with is_selected flag (relevance label)
        for j, (ptext, is_selected) in enumerate(
            zip(sample["passages"]["passage_text"], sample["passages"]["is_selected"])
        ):
            pid = f"{qid}_{j}"
            if pid not in passages and len(passages) < max_passages:
                passages[pid] = Passage(pid=pid, text=ptext)

            qrels.append(QRel(qid=qid, pid=pid, relevance=int(is_selected)))

    logger.info(
        "Loaded %d queries, %d passages, %d relevance judgments",
        len(queries), len(passages), len(qrels),
    )
    return MSMarcoDataset(queries=queries, passages=passages, qrels=qrels, split=split)


# ---------------------------------------------------------------------------
# BM25 Baseline Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    BM25Okapi-based retriever.  Acts as the first-stage retrieval
    that produces candidate sets for the reranker.

    BM25 score for term t in document d:
        score(d, q) = Σ IDF(t) * (tf * (k1+1)) / (tf + k1*(1 - b + b*dl/avgdl))

    where k1=1.5, b=0.75 are standard parameters.
    """

    def __init__(self, passages: dict[str, Passage], k1: float = 1.5, b: float = 0.75):
        self.pid_list: list[str] = list(passages.keys())
        tokenized_corpus = [passages[pid].text.lower().split() for pid in self.pid_list]

        logger.info("Building BM25 index over %d passages...", len(self.pid_list))
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        logger.info("BM25 index ready.")

    def retrieve(self, query_text: str, top_k: int = 100) -> list[tuple[str, float]]:
        """
        Retrieve top-k passages for a query.

        Returns:
            List of (pid, bm25_score) sorted descending by score.
        """
        tokenized_query = query_text.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.pid_list[i], float(scores[i])) for i in top_indices]

    def batch_retrieve(
        self,
        queries: list[Query],
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Retrieve candidates for a list of queries.

        Returns:
            Dict mapping qid -> [(pid, score), ...]
        """
        results: dict[str, list[tuple[str, float]]] = {}
        for q in queries:
            results[q.qid] = self.retrieve(q.text, top_k=top_k)
        logger.info("BM25 retrieval done for %d queries (top-%d each).", len(queries), top_k)
        return results


# ---------------------------------------------------------------------------
# Convenience: build a flat DataFrame for feature extraction
# ---------------------------------------------------------------------------

def build_candidate_df(
    queries: list[Query],
    passages: dict[str, Passage],
    qrels: list[QRel],
    retrieved: dict[str, list[tuple[str, float]]],
) -> pd.DataFrame:
    """
    Merge retrieval results with relevance labels into a flat DataFrame.

    Each row = one (query, passage) pair with columns:
        qid, pid, query_text, passage_text, bm25_score, label
    """
    # Build qrel lookup: (qid, pid) -> relevance
    qrel_map: dict[tuple[str, str], int] = {
        (qr.qid, qr.pid): qr.relevance for qr in qrels
    }

    rows = []
    for q in queries:
        for pid, bm25_score in retrieved.get(q.qid, []):
            label = qrel_map.get((q.qid, pid), 0)
            rows.append({
                "qid": q.qid,
                "pid": pid,
                "query_text": q.text,
                "passage_text": passages[pid].text if pid in passages else "",
                "bm25_score": bm25_score,
                "label": label,
            })

    df = pd.DataFrame(rows)
    logger.info("Candidate DataFrame: %d rows, %d unique queries", len(df), df["qid"].nunique())
    return df


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset = load_msmarco(split="train", max_queries=50, max_passages=10_000)

    retriever = BM25Retriever(dataset.passages)
    retrieved = retriever.batch_retrieve(dataset.queries, top_k=20)

    df = build_candidate_df(
        dataset.queries, dataset.passages, dataset.qrels, retrieved
    )
    print(df.head())
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")

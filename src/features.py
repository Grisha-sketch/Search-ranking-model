"""
features.py
-----------
Feature extraction for Learning to Rank.

Features are grouped into four families:

    1. BM25 features       — retrieval score, normalized score
    2. Document features   — length, unique terms, avg term frequency
    3. Query-doc overlap   — exact match ratio, IDF-weighted overlap
    4. Semantic features   — cosine similarity of sentence embeddings

The main entry point is `extract_features(df)` which takes the candidate
DataFrame from data_loader.build_candidate_df() and returns it with all
feature columns appended.

Feature columns (14 total):
    bm25_score              raw BM25 score from retriever
    bm25_score_norm         BM25 score normalized per query (0-1)
    doc_len                 passage word count
    doc_unique_terms        number of unique terms in passage
    doc_avg_tf              mean term frequency across passage terms
    query_len               query word count
    exact_match_ratio       fraction of query terms found in passage
    idf_weighted_overlap    overlap score weighted by IDF rarity
    title_match             1 if first sentence contains all query terms
    passage_starts_with_q   1 if passage starts with a query term
    embedding_cosine        cosine sim of query + passage embeddings
    embedding_dot           raw dot product of normalized embeddings
    bm25_x_cosine           interaction: bm25_score_norm * embedding_cosine
    len_ratio               query_len / doc_len (query coverage proxy)
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IDF helpers  (corpus-level, computed once)
# ---------------------------------------------------------------------------

def build_idf(corpus: list[str]) -> dict[str, float]:
    """
    Compute IDF for every term across the passage corpus.

        IDF(t) = log((N + 1) / (df(t) + 1))   (smoothed)

    Args:
        corpus: list of raw passage strings

    Returns:
        dict mapping term -> idf score
    """
    N = len(corpus)
    df_counts: Counter = Counter()

    for doc in corpus:
        unique_terms = set(doc.lower().split())
        df_counts.update(unique_terms)

    idf: dict[str, float] = {}
    for term, df in df_counts.items():
        idf[term] = math.log((N + 1) / (df + 1))

    logger.info("Built IDF table: %d unique terms from %d passages.", len(idf), N)
    return idf


# ---------------------------------------------------------------------------
# Individual feature extractors  (operate on single strings)
# ---------------------------------------------------------------------------

def _doc_features(text: str) -> dict[str, float]:
    """
    Passage-level lexical features.

    Returns:
        doc_len            word count
        doc_unique_terms   number of unique lowercased tokens
        doc_avg_tf         mean term frequency (count / doc_len)
    """
    tokens = text.lower().split()
    if not tokens:
        return {"doc_len": 0.0, "doc_unique_terms": 0.0, "doc_avg_tf": 0.0}

    counts = Counter(tokens)
    avg_tf = sum(counts.values()) / len(tokens)  # always 1.0 for BoW; useful after stemming

    return {
        "doc_len": float(len(tokens)),
        "doc_unique_terms": float(len(counts)),
        "doc_avg_tf": float(avg_tf),
    }


def _query_features(query: str) -> dict[str, float]:
    """Query-level lexical features."""
    tokens = query.lower().split()
    return {"query_len": float(len(tokens))}


def _overlap_features(
    query: str,
    passage: str,
    idf: dict[str, float],
) -> dict[str, float]:
    """
    Query-document overlap features.

        exact_match_ratio     fraction of query terms present in the passage
                              e.g. query="what is ML" passage="ML is..." → 2/3 ≈ 0.67

        idf_weighted_overlap  sum of IDF scores for matched query terms,
                              normalized by total query IDF weight.
                              Rare matched terms score higher than common ones.

        title_match           1 if the first sentence of the passage contains
                              all query terms (proxy for a direct answer).

        passage_starts_with_q 1 if the first word of the passage is a query term.
    """
    q_tokens = set(query.lower().split())
    p_tokens = set(passage.lower().split())
    p_words = passage.lower().split()

    matched = q_tokens & p_tokens
    exact_match_ratio = len(matched) / len(q_tokens) if q_tokens else 0.0

    total_idf = sum(idf.get(t, 0.0) for t in q_tokens)
    matched_idf = sum(idf.get(t, 0.0) for t in matched)
    idf_weighted_overlap = matched_idf / total_idf if total_idf > 0 else 0.0

    # First sentence heuristic: split on "." and check first chunk
    first_sentence = set(passage.split(".")[0].lower().split())
    title_match = 1.0 if q_tokens.issubset(first_sentence) else 0.0

    passage_starts_with_q = 1.0 if (p_words and p_words[0] in q_tokens) else 0.0

    return {
        "exact_match_ratio": exact_match_ratio,
        "idf_weighted_overlap": idf_weighted_overlap,
        "title_match": title_match,
        "passage_starts_with_q": passage_starts_with_q,
    }


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if either is zero."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Embedding model  (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_embedding_model = None


def _get_embedding_model():
    """
    Lazy-load the sentence-transformer model.
    Uses 'all-MiniLM-L6-v2': 22M params, 384-dim, fast and accurate.
    Only downloaded once; cached by sentence-transformers.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model ready.")
    return _embedding_model


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of texts into L2-normalized embedding vectors.

    Args:
        texts:      list of strings to encode
        batch_size: number of texts per GPU/CPU batch

    Returns:
        np.ndarray of shape (len(texts), 384), L2-normalized rows
    """
    model = _get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 200,
        normalize_embeddings=True,   # L2-normalize so dot product == cosine sim
    )
    return embeddings


# ---------------------------------------------------------------------------
# BM25 normalization  (per-query)
# ---------------------------------------------------------------------------

def _normalize_bm25_per_query(df: pd.DataFrame) -> pd.Series:
    """
    Min-max normalize BM25 scores within each query group.

        normalized = (score - min) / (max - min + ε)

    This makes BM25 scores comparable across queries with different
    score ranges (short vs. long queries inflate raw BM25).
    """
    def _minmax(group):
        mn = group.min()
        mx = group.max()
        return (group - mn) / (mx - mn + 1e-9)

    return df.groupby("qid")["bm25_score"].transform(_minmax)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_features(
    df: pd.DataFrame,
    use_embeddings: bool = True,
    embedding_batch_size: int = 64,
    idf: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Append all feature columns to the candidate DataFrame.

    Args:
        df:                   output of data_loader.build_candidate_df()
                              must have columns: qid, query_text, passage_text, bm25_score
        use_embeddings:       set False to skip sentence-transformer (faster dev iteration)
        embedding_batch_size: batch size for encoding
        idf:                  pre-built IDF dict; computed from df if not provided

    Returns:
        df with 14 new feature columns appended (in-place copy).
    """
    df = df.copy()

    # --- Build IDF if not supplied ---
    if idf is None:
        logger.info("Computing IDF from candidate passages...")
        idf = build_idf(df["passage_text"].tolist())

    # --- BM25 normalization ---
    logger.info("Computing BM25 normalization...")
    df["bm25_score_norm"] = _normalize_bm25_per_query(df)

    # --- Document + query + overlap features ---
    logger.info("Extracting lexical features (%d rows)...", len(df))

    doc_feats = df["passage_text"].apply(_doc_features).apply(pd.Series)
    query_feats = df["query_text"].apply(_query_features).apply(pd.Series)
    overlap_feats = df.apply(
        lambda row: _overlap_features(row["query_text"], row["passage_text"], idf),
        axis=1,
    ).apply(pd.Series)

    df = pd.concat([df, doc_feats, query_feats, overlap_feats], axis=1)

    # --- Semantic embedding features ---
    if use_embeddings:
        logger.info("Encoding queries and passages with sentence-transformers...")

        unique_queries = df["query_text"].unique().tolist()
        unique_passages = df["passage_text"].unique().tolist()

        q_embeddings_arr = encode_texts(unique_queries, batch_size=embedding_batch_size)
        p_embeddings_arr = encode_texts(unique_passages, batch_size=embedding_batch_size)

        q_emb_map = {text: vec for text, vec in zip(unique_queries, q_embeddings_arr)}
        p_emb_map = {text: vec for text, vec in zip(unique_passages, p_embeddings_arr)}

        logger.info("Computing cosine similarities...")
        cosine_scores = []
        dot_scores = []

        for _, row in df.iterrows():
            q_vec = q_emb_map[row["query_text"]]
            p_vec = p_emb_map[row["passage_text"]]
            cosine_scores.append(_cosine_similarity(q_vec, p_vec))
            dot_scores.append(float(np.dot(q_vec, p_vec)))  # same as cosine since normalized

        df["embedding_cosine"] = cosine_scores
        df["embedding_dot"] = dot_scores
    else:
        logger.info("Skipping embeddings (use_embeddings=False).")
        df["embedding_cosine"] = 0.0
        df["embedding_dot"] = 0.0

    # --- Interaction features ---
    df["bm25_x_cosine"] = df["bm25_score_norm"] * df["embedding_cosine"]
    df["len_ratio"] = df["query_len"] / (df["doc_len"] + 1e-9)

    logger.info("Feature extraction complete. Shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Feature column list  (used by model.py to select X)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "bm25_score",
    "bm25_score_norm",
    "doc_len",
    "doc_unique_terms",
    "doc_avg_tf",
    "query_len",
    "exact_match_ratio",
    "idf_weighted_overlap",
    "title_match",
    "passage_starts_with_q",
    "embedding_cosine",
    "embedding_dot",
    "bm25_x_cosine",
    "len_ratio",
]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal test without MS MARCO — just verify the pipeline runs
    sample_df = pd.DataFrame([
        {
            "qid": "q1",
            "pid": "p1",
            "query_text": "what is machine learning",
            "passage_text": "Machine learning is a subset of artificial intelligence "
            "that enables systems to learn from data.",
            "bm25_score": 4.2,
            "label": 1,
        },
        {
            "qid": "q1",
            "pid": "p2",
            "query_text": "what is machine learning",
            "passage_text": "The weather today is sunny with a high of 75 degrees.",
            "bm25_score": 0.3,
            "label": 0,
        },
        {
            "qid": "q2",
            "pid": "p3",
            "query_text": "python list comprehension",
            "passage_text": "List comprehensions provide a concise way to create lists in Python.",
            "bm25_score": 3.8,
            "label": 1,
        },
    ])

    result = extract_features(sample_df, use_embeddings=True)

    print("\nFeature columns:")
    print(result[FEATURE_COLS].to_string())
    print(f"\nShape: {result.shape}")
 
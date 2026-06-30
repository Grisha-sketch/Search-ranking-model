"""
api.py
------
FastAPI endpoint for serving the LambdaMART search ranker.

Endpoints:
    POST /rank          rerank a list of candidate passages for a query
    GET  /health        liveness check
    GET  /model/info    feature list, model params, load status

Request flow:
    1. Client sends query + list of (pid, passage_text, bm25_score) candidates
    2. API builds a single-query DataFrame
    3. Runs feature extraction  (embeddings optional via config)
    4. Adds simulated user signals  (in production these would be real)
    5. Calls ranker.predict() → scores
    6. Returns candidates sorted by score with metadata

Run locally:
    uvicorn src.api:app --reload --port 8000

Example request:
    curl -X POST http://localhost:8000/rank \\
      -H "Content-Type: application/json" \\
      -d '{
        "query": "what is machine learning",
        "candidates": [
          {"pid": "p1", "text": "Machine learning is a method of data analysis.", "bm25_score": 4.2},
          {"pid": "p2", "text": "The weather is sunny today.", "bm25_score": 0.3}
        ]
      }'
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import FEATURE_COLS, extract_features
from src.model import MODEL_PATH, LambdaMARTRanker
from src.user_signals import USER_SIGNAL_COLS, SimulationConfig, add_user_signals

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "false").lower() == "true"
SIM_SESSIONS = int(os.getenv("SIM_SESSIONS", "50"))   # user signal sim sessions

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    ranker: Optional[LambdaMARTRanker] = None
    model_loaded: bool = False
    load_error: Optional[str] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    logger.info("Loading LambdaMART model from %s...", MODEL_PATH)
    try:
        state.ranker = LambdaMARTRanker()
        state.ranker.load(MODEL_PATH)
        state.model_loaded = True
        logger.info("Model loaded successfully.")
    except Exception as e:
        state.load_error = str(e)
        logger.warning("Model not loaded: %s. /rank will return BM25 fallback.", e)

    yield  # app runs here

    logger.info("Shutting down.")


app = FastAPI(
    title="Search Ranking API",
    description="LambdaMART reranker with user signals",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Candidate(BaseModel):
    pid: str = Field(..., description="Passage ID")
    text: str = Field(..., description="Passage text")
    bm25_score: float = Field(0.0, description="BM25 retrieval score")


class RankRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    candidates: list[Candidate] = Field(
        ..., min_length=1, max_length=200,
        description="Candidate passages to rerank",
    )
    use_embeddings: Optional[bool] = Field(
        None,
        description="Override server-level USE_EMBEDDINGS env var for this request",
    )


class RankedResult(BaseModel):
    pid: str
    text: str
    score: float
    bm25_score: float
    rank: int


class RankResponse(BaseModel):
    query: str
    results: list[RankedResult]
    model_used: str          # "lambdamart" | "bm25_fallback"
    latency_ms: float
    num_candidates: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    load_error: Optional[str]


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_path: str
    feature_cols: list[str]
    signal_cols: list[str]
    total_features: int
    use_embeddings: bool
    sim_sessions: int


# ---------------------------------------------------------------------------
# Helper: build DataFrame for a single request
# ---------------------------------------------------------------------------

def _build_request_df(query: str, candidates: list[Candidate]) -> pd.DataFrame:
    """Convert a RankRequest into the DataFrame format expected by the pipeline."""
    return pd.DataFrame([
        {
            "qid": "req",
            "pid": c.pid,
            "query_text": query,
            "passage_text": c.text,
            "bm25_score": c.bm25_score,
            "label": 0,   # unknown at serving time — only needed for training
        }
        for c in candidates
    ])


def _bm25_fallback(candidates: list[Candidate]) -> list[RankedResult]:
    """Return candidates sorted by BM25 score when model is unavailable."""
    sorted_candidates = sorted(candidates, key=lambda c: c.bm25_score, reverse=True)
    return [
        RankedResult(
            pid=c.pid,
            text=c.text,
            score=c.bm25_score,
            bm25_score=c.bm25_score,
            rank=i + 1,
        )
        for i, c in enumerate(sorted_candidates)
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check. Returns model load status."""
    return HealthResponse(
        status="ok",
        model_loaded=state.model_loaded,
        load_error=state.load_error,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """Returns feature list and server configuration."""
    return ModelInfoResponse(
        model_loaded=state.model_loaded,
        model_path=str(MODEL_PATH),
        feature_cols=FEATURE_COLS,
        signal_cols=USER_SIGNAL_COLS,
        total_features=len(FEATURE_COLS) + len(USER_SIGNAL_COLS),
        use_embeddings=USE_EMBEDDINGS,
        sim_sessions=SIM_SESSIONS,
    )


@app.post("/rank", response_model=RankResponse, tags=["ranking"])
def rank(request: RankRequest) -> RankResponse:
    """
    Rerank candidate passages for a query using LambdaMART.

    If the model failed to load, falls back to BM25 ordering.
    Feature extraction runs synchronously — for high-throughput
    production use, this should be moved to a worker pool.
    """
    t0 = time.perf_counter()

    # --- BM25 fallback if model not loaded ---
    if not state.model_loaded:
        results = _bm25_fallback(request.candidates)
        return RankResponse(
            query=request.query,
            results=results,
            model_used="bm25_fallback",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            num_candidates=len(request.candidates),
        )

    # --- Feature extraction ---
    use_emb = request.use_embeddings if request.use_embeddings is not None else USE_EMBEDDINGS

    try:
        df = _build_request_df(request.query, request.candidates)

        df = extract_features(df, use_embeddings=use_emb)

        sim_config = SimulationConfig(num_sessions=SIM_SESSIONS, random_seed=0)
        df = add_user_signals(df, config=sim_config)

    except Exception as e:
        logger.exception("Feature extraction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    # --- Inference ---
    try:
        scores = state.ranker.predict(df)
    except Exception as e:
        logger.exception("Model inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # --- Build response ---
    df["score"] = scores
    df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Map pid -> original candidate for text lookup
    pid_to_candidate = {c.pid: c for c in request.candidates}

    results = []
    for rank_pos, row in enumerate(df_sorted.itertuples(), start=1):
        c = pid_to_candidate[row.pid]
        results.append(RankedResult(
            pid=row.pid,
            text=c.text,
            score=round(float(row.score), 6),
            bm25_score=c.bm25_score,
            rank=rank_pos,
        ))

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "Ranked %d candidates for query=%r in %.1f ms (model=lambdamart)",
        len(results), request.query, latency_ms,
    )

    return RankResponse(
        query=request.query,
        results=results,
        model_used="lambdamart",
        latency_ms=latency_ms,
        num_candidates=len(request.candidates),
    )
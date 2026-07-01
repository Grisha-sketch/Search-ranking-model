"""
tests/test_api.py
-----------------
Integration tests for the /rank, /health, and /model/info endpoints.

Uses FastAPI's TestClient (no running server needed).
The model is NOT loaded in these tests — we test the BM25 fallback
path and schema validation. The smoke test in model.py covers the
trained model path.

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app, state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient with model explicitly unloaded (tests fallback path)."""
    state.model_loaded = False
    state.load_error = "No model file (test mode)"
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_candidates():
    return [
        {
            "pid": "p1",
            "text": "Machine learning is a subset of artificial intelligence.",
            "bm25_score": 4.2,
        },
        {
            "pid": "p2",
            "text": "The weather today is sunny with a high of 75 degrees.",
            "bm25_score": 0.3,
        },
        {
            "pid": "p3",
            "text": "Deep learning uses neural networks to learn representations.",
            "bm25_score": 3.1,
        },
    ]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "ok"

    def test_model_not_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is False
        assert data["load_error"] is not None


# ---------------------------------------------------------------------------
# /model/info
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_returns_200(self, client):
        assert client.get("/model/info").status_code == 200

    def test_feature_cols_present(self, client):
        data = client.get("/model/info").json()
        assert len(data["feature_cols"]) > 0
        assert len(data["signal_cols"]) > 0

    def test_total_features(self, client):
        data = client.get("/model/info").json()
        assert data["total_features"] == len(data["feature_cols"]) + len(data["signal_cols"])


# ---------------------------------------------------------------------------
# /rank  — BM25 fallback (model not loaded)
# ---------------------------------------------------------------------------

class TestRankFallback:
    def test_returns_200(self, client, sample_candidates):
        r = client.post("/rank", json={
            "query": "machine learning",
            "candidates": sample_candidates,
        })
        assert r.status_code == 200

    def test_response_schema(self, client, sample_candidates):
        data = client.post("/rank", json={
            "query": "machine learning",
            "candidates": sample_candidates,
        }).json()
        assert "results" in data
        assert "model_used" in data
        assert "latency_ms" in data
        assert "num_candidates" in data

    def test_fallback_model_label(self, client, sample_candidates):
        data = client.post("/rank", json={
            "query": "machine learning",
            "candidates": sample_candidates,
        }).json()
        assert data["model_used"] == "bm25_fallback"

    def test_bm25_ordering(self, client, sample_candidates):
        """Fallback should sort by bm25_score descending."""
        data = client.post("/rank", json={
            "query": "machine learning",
            "candidates": sample_candidates,
        }).json()
        scores = [r["bm25_score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self, client, sample_candidates):
        data = client.post("/rank", json={
            "query": "machine learning",
            "candidates": sample_candidates,
        }).json()
        ranks = [r["rank"] for r in data["results"]]
        assert ranks == list(range(1, len(sample_candidates) + 1))

    def test_num_candidates_matches(self, client, sample_candidates):
        data = client.post("/rank", json={
            "query": "machine learning",
            "candidates": sample_candidates,
        }).json()
        assert data["num_candidates"] == len(sample_candidates)
        assert len(data["results"]) == len(sample_candidates)

    def test_query_echoed(self, client, sample_candidates):
        query = "what is machine learning"
        data = client.post("/rank", json={
            "query": query,
            "candidates": sample_candidates,
        }).json()
        assert data["query"] == query

    def test_single_candidate(self, client):
        data = client.post("/rank", json={
            "query": "test",
            "candidates": [{"pid": "p1", "text": "some text", "bm25_score": 1.0}],
        }).json()
        assert len(data["results"]) == 1
        assert data["results"][0]["rank"] == 1

    def test_latency_is_positive(self, client, sample_candidates):
        data = client.post("/rank", json={
            "query": "test",
            "candidates": sample_candidates,
        }).json()
        assert data["latency_ms"] > 0


# ---------------------------------------------------------------------------
# /rank  — input validation
# ---------------------------------------------------------------------------

class TestRankValidation:
    def test_empty_query_rejected(self, client, sample_candidates):
        r = client.post("/rank", json={
            "query": "",
            "candidates": sample_candidates,
        })
        assert r.status_code == 422

    def test_empty_candidates_rejected(self, client):
        r = client.post("/rank", json={
            "query": "machine learning",
            "candidates": [],
        })
        assert r.status_code == 422

    def test_missing_query_rejected(self, client, sample_candidates):
        r = client.post("/rank", json={"candidates": sample_candidates})
        assert r.status_code == 422

    def test_missing_candidates_rejected(self, client):
        r = client.post("/rank", json={"query": "test"})
        assert r.status_code == 422

    def test_candidate_default_bm25(self, client):
        """bm25_score should default to 0.0 if omitted."""
        data = client.post("/rank", json={
            "query": "test",
            "candidates": [{"pid": "p1", "text": "some text"}],
        }).json()
        assert data["results"][0]["bm25_score"] == 0.0

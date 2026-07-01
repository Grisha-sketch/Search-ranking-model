Search Ranking Model

A production-style Learning to Rank (LTR) system built on MS MARCO that reranks BM25 candidates using user interaction signals and LightGBM LambdaMART.

Architecture

Query
  │
  ▼
BM25 Retrieval (rank_bm25)         first-stage, top-100 candidates
  │
  ▼
Feature Extraction (21 features)   BM25 scores, embeddings, lexical overlap
  │
  ▼
User Signal Simulation             click rate, dwell time, IPS debiasing
  │
  ▼
LightGBM LambdaMART Reranker       trained on MS MARCO relevance labels
  │
  ▼
FastAPI  /rank  endpoint           returns sorted passages + scores
  │
  ▼
MLflow Experiment Tracking         NDCG@1/5/10, MAP, MRR per run


Features (21 total)

BM25 (2) — raw retrieval score + per-query min-max normalized score

Document (3) — word count, unique terms, average term frequency

Query (1) — query length

Query-document overlap (4) — exact match ratio, IDF-weighted overlap, title match heuristic, passage start match

Semantic (3) — sentence-transformer cosine similarity, dot product (all-MiniLM-L6-v2, 384-dim), BM25×cosine interaction

Derived (1) — query/doc length ratio

User signals (7) — click rate, avg dwell time, skip rate, IPS weight, IPS-weighted click, dwell score, composite engagement score


User Signal Pipeline

Raw click data is biased — results at position 1 get clicked more often simply because they are visible, not because they are relevant. This project corrects for position bias using Inverse Propensity Scoring (IPS):

P(examine | position k) = (1 / k) ^ η       cascade model, η = 0.8

IPS weight = position ^ η                    upweights clicks from lower positions

weighted_click = click_rate × IPS_weight     debiased training signal

Sessions are simulated with configurable CTR (60% relevant, 10% irrelevant) and dwell time (45s relevant, 8s irrelevant) with Gaussian noise.


Evaluation Metrics

MetricDescriptionNDCG@kPrimary — rewards placing relevant docs at the top; discounts by log positionMAPMean Average Precision — precision averaged over all relevant doc positionsMRRMean Reciprocal Rank — position of the first relevant result


Project Structure

search-ranking-model/
├── src/
│   ├── data_loader.py      MS MARCO loading + BM25Retriever
│   ├── features.py         14 lexical + semantic features, IDF table
│   ├── user_signals.py     session simulation, IPS debiasing, 7 signal features
│   ├── model.py            LambdaMARTRanker (fit/predict/save/load) + MLflow
│   ├── evaluate.py         NDCG, MAP, MRR — from scratch
│   └── api.py              FastAPI /rank /health /model/info
├── tests/
│   ├── test_evaluate.py    24 unit tests — metrics
│   └── test_api.py         20 integration tests — endpoints + validation
├── models/                 saved .lgb model + feature importance CSV
├── .github/workflows/
│   └── ci.yml              lint → test → smoke test
└── requirements.txt


Setup

cmdpython -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


Train

Full training on MS MARCO (downloads ~500MB on first run):

cmdpython -m src.model --max-queries 500 --top-k 50 --no-embeddings

With embeddings (slower, higher quality):

cmdpython -m src.model --max-queries 500 --top-k 50

View results in MLflow:

cmdmlflow ui

Then open http://localhost:5000.


Run the API

cmduvicorn src.api:app --reload --port 8000

Rerank candidates:

cmdcurl -X POST http://localhost:8000/rank ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"what is machine learning\", \"candidates\": [{\"pid\": \"p1\", \"text\": \"Machine learning is a subset of AI.\", \"bm25_score\": 4.2}, {\"pid\": \"p2\", \"text\": \"The weather is sunny today.\", \"bm25_score\": 0.3}]}"

Health check:

cmdcurl http://localhost:8000/health

If the model file is not found, /rank automatically falls back to BM25 ordering — no 500 errors.


Test

cmdpytest tests/ -v

44 tests across metrics and API endpoints. CI also runs a model smoke test:

cmdpython -m src.model --smoke-test

Trains LambdaMART on 20 synthetic queries and asserts NDCG@5 does not regress below BM25.


Results

ModelNDCG@5MAPMRRBM25 Baseline———LambdaMART (no embeddings)———LambdaMART (+ embeddings)———

(fill in after training on MS MARCO)


Tech Stack

LayerToolDatasetMS MARCO v2.1 (HuggingFace datasets)First-stage retrievalrank-bm25Embeddingssentence-transformers all-MiniLM-L6-v2Ranking modelLightGBM LambdaMARTExperiment trackingMLflowAPIFastAPI + UvicornTestingpytest + httpxCIGitHub Actions
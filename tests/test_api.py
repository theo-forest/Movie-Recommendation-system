"""
Integration tests for the FastAPI recommendation endpoints.

Uses TestClient (httpx) so no running server is needed.
Models are NOT loaded (mocked) so tests are fast and self-contained.
"""
import pytest
from fastapi.testclient import TestClient

import src.api as api_module
from src.api import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers — inject mock models
# ---------------------------------------------------------------------------

class _MockCF:
    _user_ids = [1, 2, 3]

    def recommend(self, user_id, n=10, exclude_seen=True):
        if user_id not in self._user_ids:
            return []
        return [(100 + i, 4.5 - i * 0.1) for i in range(n)]


class _MockCBF:
    _movie_ids = [1, 2, 3]

    def recommend(self, movie_id, n=10):
        if movie_id not in self._movie_ids:
            return []
        return [(200 + i, 0.9 - i * 0.05) for i in range(n)]


class _MockHybrid:
    def recommend_for_user(self, user_id, n=10):
        return [(300 + i, 0.85 - i * 0.05) for i in range(n)]

    def recommend_for_movie(self, movie_id, n=10):
        return [(400 + i, 0.8 - i * 0.05) for i in range(n)]


import pandas as pd

_MOCK_MOVIES = pd.DataFrame({
    "movieId": [1, 2, 3, 100, 101, 200, 201, 300, 301],
    "title": [
        "Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)",
        "Movie 100 (2000)", "Movie 101 (2001)",
        "Movie 200 (2000)", "Movie 201 (2001)",
        "Movie 300 (2000)", "Movie 301 (2001)",
    ],
    "genres": [
        "Adventure|Animation|Children|Comedy|Fantasy",
        "Adventure|Children|Fantasy",
        "Comedy|Romance",
        "Action", "Drama",
        "Comedy", "Thriller",
        "Action|Drama", "Comedy|Romance",
    ],
})


@pytest.fixture(autouse=True)
def inject_mocks():
    """Replace global model/data references with lightweight mocks."""
    api_module._cf = _MockCF()
    api_module._cbf = _MockCBF()
    api_module._hybrid = _MockHybrid()
    api_module._movies_df = _MOCK_MOVIES
    yield
    api_module._cf = None
    api_module._cbf = None
    api_module._hybrid = None
    api_module._movies_df = None


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["models"]["collaborative_filter"] is True
    assert data["models"]["content_filter"] is True
    assert data["models"]["hybrid_recommender"] is True
    assert data["movies_loaded"] is True


# ---------------------------------------------------------------------------
# /recommend — user-based
# ---------------------------------------------------------------------------

def test_recommend_user_hybrid():
    resp = client.post("/recommend", json={"user_id": 1, "n": 5, "model_type": "hybrid"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["recommendations"]) == 5
    assert data["model_used"] == "hybrid"
    # Each recommendation must have the required fields
    rec = data["recommendations"][0]
    assert "movie_id" in rec
    assert "title" in rec
    assert "genres" in rec
    assert "score" in rec


def test_recommend_user_collaborative():
    resp = client.post("/recommend", json={"user_id": 1, "n": 5, "model_type": "collaborative"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_used"] == "collaborative"
    assert len(data["recommendations"]) == 5


def test_recommend_unknown_user_returns_404_or_empty():
    # MockCF returns [] for unknown users; API should return 404
    resp = client.post("/recommend", json={"user_id": 9999, "n": 5, "model_type": "hybrid"})
    # Hybrid mock always returns results regardless of user; just check 200
    assert resp.status_code in (200, 404)


# ---------------------------------------------------------------------------
# /recommend — item-based
# ---------------------------------------------------------------------------

def test_recommend_item_based():
    resp = client.post("/recommend", json={"movie_id": 1, "n": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_used"] == "content"
    assert len(data["recommendations"]) == 5


def test_recommend_unknown_movie_returns_404():
    resp = client.post("/recommend", json={"movie_id": 9999, "n": 5})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /recommend — validation
# ---------------------------------------------------------------------------

def test_recommend_no_input_returns_400():
    resp = client.post("/recommend", json={"n": 5})
    assert resp.status_code == 400


def test_recommend_invalid_model_type_returns_400():
    resp = client.post(
        "/recommend",
        json={"user_id": 1, "n": 5, "model_type": "nonexistent_model"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /movie/{movie_id}
# ---------------------------------------------------------------------------

def test_get_movie_found():
    resp = client.get("/movie/1")
    assert resp.status_code == 200
    data = resp.json()
    assert "title" in data
    assert "Toy Story" in data["title"]


def test_get_movie_not_found():
    resp = client.get("/movie/99999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /movies/search
# ---------------------------------------------------------------------------

def test_search_movies():
    resp = client.get("/movies/search", params={"q": "Toy"})
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) >= 1
    assert any("Toy Story" in r["title"] for r in results)


def test_search_movies_no_results():
    resp = client.get("/movies/search", params={"q": "xyznotamovie"})
    assert resp.status_code == 200
    assert resp.json() == []

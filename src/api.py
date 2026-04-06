"""
FastAPI recommendation service.

Endpoints
---------
POST /recommend          — get movie recommendations (user or item based)
GET  /movie/{movie_id}   — fetch metadata for a single movie
GET  /metrics            — last saved evaluation metrics
GET  /health             — liveness check with model load status

Models are loaded once at startup from the models/ directory.
Run training first:  python -m src.train
Then start the API:  uvicorn src.api:app --reload
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_HERE, "..", "models")
DATA_DIR = os.path.join(_HERE, "..", "data", "raw", "ml-latest-small")

# ---------------------------------------------------------------------------
# Global model / data cache (populated at startup)
# ---------------------------------------------------------------------------
_cf = None
_cbf = None
_hybrid = None
_movies_df: Optional[pd.DataFrame] = None
_eval_metrics: dict = {}


def _load_model(name: str):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        logger.warning(f"Model file not found: {path}  (run src/train.py first)")
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        logger.error(f"Failed to load {path}: {exc}")
        return None


def _load_all_models():
    global _cf, _cbf, _hybrid, _movies_df, _eval_metrics

    _cf     = _load_model("collaborative_filter.pkl")
    _cbf    = _load_model("content_filter.pkl")
    _hybrid = _load_model("hybrid_recommender.pkl")

    try:
        _movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
        logger.info(f"Movies catalogue loaded: {len(_movies_df)} entries")
    except Exception as exc:
        logger.error(f"Could not load movies.csv: {exc}")

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _eval_metrics = json.load(f)

    logger.info(
        f"Startup complete — CF:{_cf is not None}, "
        f"CBF:{_cbf is not None}, Hybrid:{_hybrid is not None}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all_models()
    yield


app = FastAPI(
    title="Movie Recommendation API",
    description="SVD collaborative filtering + TF-IDF content-based + hybrid",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RecommendationRequest(BaseModel):
    user_id:    Optional[int] = Field(None, description="Request user-based recommendations")
    movie_id:   Optional[int] = Field(None, description="Request item-based (similar movies) recommendations")
    n:          int            = Field(10, ge=1, le=50, description="Number of results")
    model_type: str            = Field("hybrid", description="'hybrid' | 'collaborative' | 'content'")


class MovieRec(BaseModel):
    movie_id: int
    title:    str
    genres:   str
    year:     Optional[float]
    score:    float


class RecommendationResponse(BaseModel):
    recommendations: List[MovieRec]
    explanation:     str
    model_used:      str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _movie_info(movie_id: int) -> dict:
    """Return metadata dict for movie_id, or minimal fallback."""
    if _movies_df is None:
        return {"title": str(movie_id), "genres": "Unknown", "year": None}
    row = _movies_df[_movies_df["movieId"] == movie_id]
    if row.empty:
        return {"title": str(movie_id), "genres": "Unknown", "year": None}
    r = row.iloc[0]
    import re
    year_match = re.search(r"\((\d{4})\)$", str(r["title"]))
    year = float(year_match.group(1)) if year_match else None
    return {"title": r["title"], "genres": r["genres"], "year": year}


def _build_recs(raw_recs) -> List[MovieRec]:
    return [
        MovieRec(
            movie_id=mid,
            score=round(score, 4),
            **_movie_info(mid),
        )
        for mid, score in raw_recs
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest):
    if req.user_id is None and req.movie_id is None:
        raise HTTPException(400, "Provide either user_id or movie_id.")

    recs = []
    explanation = ""
    model_used = req.model_type

    # ---- User-based -------------------------------------------------------
    if req.user_id is not None:
        if req.model_type == "hybrid":
            if _hybrid is None:
                raise HTTPException(503, "Hybrid model not loaded. Run src/train.py first.")
            raw = _hybrid.recommend_for_user(req.user_id, n=req.n)
            explanation = (
                f"Hybrid (CF 70% + CBF 30%) recommendations for user {req.user_id}. "
                "Based on your rating history plus similar-movie content."
            )
        elif req.model_type == "collaborative":
            if _cf is None:
                raise HTTPException(503, "Collaborative filter not loaded.")
            raw = _cf.recommend(req.user_id, n=req.n)
            explanation = (
                f"SVD collaborative filtering for user {req.user_id}. "
                "Users with similar taste also liked these movies."
            )
        else:
            raise HTTPException(400, f"model_type '{req.model_type}' not supported for user recommendations.")
        recs = _build_recs(raw)

    # ---- Item-based -------------------------------------------------------
    elif req.movie_id is not None:
        if _cbf is None:
            raise HTTPException(503, "Content filter not loaded.")
        raw = _cbf.recommend(req.movie_id, n=req.n)
        model_used = "content"
        title = _movie_info(req.movie_id)["title"]
        explanation = f"Movies with similar genre and title profile to '{title}'."
        recs = _build_recs(raw)

    if not recs:
        raise HTTPException(404, "No recommendations found for this input.")

    return RecommendationResponse(
        recommendations=recs,
        explanation=explanation,
        model_used=model_used,
    )


@app.get("/movie/{movie_id}")
def get_movie(movie_id: int):
    info = _movie_info(movie_id)
    if info["title"] == str(movie_id):
        raise HTTPException(404, f"Movie {movie_id} not found.")
    return info


@app.get("/movies/search")
def search_movies(q: str = Query(..., min_length=2), limit: int = Query(10, ge=1, le=50)):
    """Full-text search on movie titles."""
    if _movies_df is None:
        raise HTTPException(503, "Movies data not loaded.")
    mask = _movies_df["title"].str.contains(q, case=False, na=False)
    results = _movies_df[mask].head(limit)
    return results[["movieId", "title", "genres"]].to_dict(orient="records")


@app.get("/metrics")
def get_metrics():
    return _eval_metrics or {"message": "No metrics available. Run src/train.py first."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "collaborative_filter": _cf is not None,
            "content_filter":       _cbf is not None,
            "hybrid_recommender":   _hybrid is not None,
        },
        "movies_loaded": _movies_df is not None,
    }

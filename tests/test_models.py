"""
Unit tests for recommendation models.

Uses a tiny synthetic dataset so tests run in < 1 second without GPU or
the full MovieLens data.
"""
import numpy as np
import pandas as pd
import pytest

from src.models import CollaborativeFilter, ContentBasedFilter, HybridRecommender


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_ratings():
    """5 users × 10 movies, random ratings between 1 and 5."""
    rng = np.random.default_rng(0)
    rows = []
    import pandas as pd
    from datetime import datetime, timedelta
    base_time = datetime(2020, 1, 1)
    for uid in range(1, 6):
        for mid in range(1, 11):
            if rng.random() > 0.3:   # ~70% density
                rows.append({
                    "userId": uid,
                    "movieId": mid,
                    "rating": float(rng.integers(1, 6)),
                    "timestamp": base_time + timedelta(days=int(uid * 10 + mid)),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def small_movies():
    """10 synthetic movies with titles and genres."""
    genres = [
        "Action|Adventure", "Comedy|Romance", "Drama", "Sci-Fi|Thriller",
        "Animation|Family", "Horror", "Documentary", "Comedy|Drama",
        "Action|Sci-Fi", "Romance|Drama",
    ]
    return pd.DataFrame({
        "movieId": list(range(1, 11)),
        "title": [f"Movie {i} ({2000 + i})" for i in range(1, 11)],
        "genres": genres,
    })


# ---------------------------------------------------------------------------
# CollaborativeFilter
# ---------------------------------------------------------------------------

class TestCollaborativeFilter:
    def test_fit_creates_factors(self, small_ratings):
        cf = CollaborativeFilter(n_factors=3)   # 3 < min(5 users, 10 movies) - 1 = 4, always safe
        cf.fit(small_ratings)
        assert cf._user_factors is not None
        assert cf._item_factors is not None
        assert cf._user_factors.shape[1] == 3

    def test_recommend_returns_correct_length(self, small_ratings):
        cf = CollaborativeFilter(n_factors=5)
        cf.fit(small_ratings)
        recs = cf.recommend(user_id=1, n=3)
        assert len(recs) == 3

    def test_recommend_excludes_seen(self, small_ratings):
        cf = CollaborativeFilter(n_factors=5)
        cf.fit(small_ratings)
        seen = set(
            small_ratings[small_ratings["userId"] == 1]["movieId"].tolist()
        )
        recs = cf.recommend(user_id=1, n=5, exclude_seen=True)
        recommended_ids = {mid for mid, _ in recs}
        assert recommended_ids.isdisjoint(seen)

    def test_predict_rating_in_range(self, small_ratings):
        cf = CollaborativeFilter(n_factors=5)
        cf.fit(small_ratings)
        pred = cf.predict_rating(1, 1)
        assert pred is not None
        assert 0.5 <= pred <= 5.0

    def test_unknown_user_returns_empty(self, small_ratings):
        cf = CollaborativeFilter(n_factors=5)
        cf.fit(small_ratings)
        recs = cf.recommend(user_id=9999, n=5)
        assert recs == []

    def test_recommend_scores_descending(self, small_ratings):
        cf = CollaborativeFilter(n_factors=5)
        cf.fit(small_ratings)
        recs = cf.recommend(user_id=1, n=5)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# ContentBasedFilter
# ---------------------------------------------------------------------------

class TestContentBasedFilter:
    def test_fit_builds_matrix(self, small_movies):
        cbf = ContentBasedFilter(max_features=100)
        cbf.fit(small_movies)
        assert cbf._tfidf_matrix is not None
        assert cbf._tfidf_matrix.shape[0] == len(small_movies)

    def test_recommend_returns_correct_length(self, small_movies):
        cbf = ContentBasedFilter(max_features=100)
        cbf.fit(small_movies)
        recs = cbf.recommend(movie_id=1, n=3)
        assert len(recs) == 3

    def test_recommend_excludes_query_movie(self, small_movies):
        cbf = ContentBasedFilter(max_features=100)
        cbf.fit(small_movies)
        recs = cbf.recommend(movie_id=1, n=5)
        returned_ids = [mid for mid, _ in recs]
        assert 1 not in returned_ids

    def test_unknown_movie_returns_empty(self, small_movies):
        cbf = ContentBasedFilter(max_features=100)
        cbf.fit(small_movies)
        recs = cbf.recommend(movie_id=9999, n=5)
        assert recs == []


# ---------------------------------------------------------------------------
# HybridRecommender
# ---------------------------------------------------------------------------

class TestHybridRecommender:
    def test_fit_and_recommend(self, small_ratings, small_movies):
        hybrid = HybridRecommender(cf_weight=0.7, cbf_weight=0.3)
        hybrid.fit(small_ratings, small_movies)
        recs = hybrid.recommend_for_user(user_id=1, n=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5

    def test_recommend_for_movie(self, small_ratings, small_movies):
        hybrid = HybridRecommender(cf_weight=0.7, cbf_weight=0.3)
        hybrid.fit(small_ratings, small_movies)
        recs = hybrid.recommend_for_movie(movie_id=1, n=3)
        assert len(recs) == 3

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            HybridRecommender(cf_weight=0.5, cbf_weight=0.8)

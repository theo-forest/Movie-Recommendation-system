"""
Recommendation models.

Three strategies are implemented and share a common interface:
  - CollaborativeFilter  : SVD matrix factorisation on user–item ratings
  - ContentBasedFilter   : TF-IDF on genre + title text, cosine similarity
  - HybridRecommender    : weighted combination of the two above
"""
import logging
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Recommendations = List[Tuple[int, float]]   # [(movie_id, score), ...]


# ---------------------------------------------------------------------------
# Collaborative Filtering  —  SVD matrix factorisation
# ---------------------------------------------------------------------------

class CollaborativeFilter:
    """
    Memory-efficient SVD-based collaborative filter.

    The user–item rating matrix is mean-centred per user before decomposition
    so the model learns *preference deviations* rather than raw ratings.
    This improves recommendations for users who rate everything high/low.
    """

    def __init__(self, n_factors: int = 50, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self._svd = TruncatedSVD(n_components=n_factors, random_state=random_state)

        # Populated by fit()
        self._user_factors: Optional[np.ndarray] = None
        self._item_factors: Optional[np.ndarray] = None
        self._user_mean: Optional[pd.Series] = None
        self._user_ids: Optional[List[int]] = None
        self._movie_ids: Optional[List[int]] = None
        self._user_item_matrix: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def fit(self, ratings_df: pd.DataFrame) -> "CollaborativeFilter":
        """
        Fit on a ratings DataFrame with columns [userId, movieId, rating].
        """
        logger.info("CollaborativeFilter.fit — building user-item matrix …")
        matrix = ratings_df.pivot_table(
            index="userId", columns="movieId", values="rating"
        )
        self._user_item_matrix = matrix
        self._user_ids = matrix.index.tolist()
        self._movie_ids = matrix.columns.tolist()

        # Mean-centre per user to remove rating-scale bias
        self._user_mean = matrix.mean(axis=1)
        centred = matrix.sub(self._user_mean, axis=0).fillna(0)

        # Cap n_factors to the maximum allowed by the matrix dimensions
        max_factors = min(centred.shape) - 1
        if self.n_factors > max_factors:
            logger.warning(
                f"n_factors={self.n_factors} exceeds matrix limit; capping to {max_factors}"
            )
            self._svd = TruncatedSVD(n_components=max_factors, random_state=self.random_state)

        sparse = csr_matrix(centred.values.astype(np.float32))
        self._user_factors = self._svd.fit_transform(sparse)           # (n_users, k)
        self._item_factors = self._svd.components_.T                   # (n_movies, k)

        logger.info(
            f"CollaborativeFilter fitted — {len(self._user_ids)} users, "
            f"{len(self._movie_ids)} movies, {self.n_factors} latent factors"
        )
        return self

    # ------------------------------------------------------------------
    def predict_rating(self, user_id: int, movie_id: int) -> Optional[float]:
        """Predict the rating user_id would give movie_id (0.5–5.0 scale)."""
        if user_id not in self._user_ids or movie_id not in self._movie_ids:
            return None
        u = self._user_ids.index(user_id)
        i = self._movie_ids.index(movie_id)
        raw = float(np.dot(self._user_factors[u], self._item_factors[i]))
        return float(np.clip(raw + self._user_mean.iloc[u], 0.5, 5.0))

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> Recommendations:
        """Return top-n (movie_id, predicted_rating) pairs for a user."""
        if user_id not in self._user_ids:
            logger.warning(f"user_id {user_id} not in training data")
            return []
        u = self._user_ids.index(user_id)
        scores = (
            np.dot(self._user_factors[u], self._item_factors.T)
            + self._user_mean.iloc[u]
        )

        if exclude_seen:
            seen = self._user_item_matrix.loc[user_id].dropna().index
            seen_indices = [self._movie_ids.index(m) for m in seen if m in self._movie_ids]
            scores[seen_indices] = -np.inf

        # Sort descending and filter out any -inf entries (seen items that overflowed n)
        top_idx = np.argsort(scores)[::-1]
        top_idx = top_idx[scores[top_idx] > -np.inf][:n]
        return [(self._movie_ids[i], float(scores[i])) for i in top_idx]

    # ------------------------------------------------------------------
    def similar_users(self, user_id: int, n: int = 5) -> List[Tuple[int, float]]:
        """Find the n most similar users to user_id."""
        if user_id not in self._user_ids:
            return []
        u = self._user_ids.index(user_id)
        sims = cosine_similarity(
            self._user_factors[u].reshape(1, -1), self._user_factors
        ).flatten()
        sims[u] = -1  # exclude self
        top_idx = np.argsort(sims)[::-1][:n]
        return [(self._user_ids[i], float(sims[i])) for i in top_idx]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"CollaborativeFilter saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "CollaborativeFilter":
        model = joblib.load(path)
        logger.info(f"CollaborativeFilter loaded <- {path}")
        return model


# ---------------------------------------------------------------------------
# Content-Based Filtering  —  TF-IDF + cosine similarity
# ---------------------------------------------------------------------------

class ContentBasedFilter:
    """
    Item–item content-based recommender.

    Features: genres (pipe-separated) concatenated with the cleaned movie title.
    A TF-IDF matrix is computed once at fit-time; similarity is retrieved via
    cosine distance at query time.
    """

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._movie_ids: Optional[List[int]] = None
        self._movies_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def fit(self, movies_df: pd.DataFrame) -> "ContentBasedFilter":
        """
        Fit on a movies DataFrame with columns [movieId, title, genres].
        """
        logger.info("ContentBasedFilter.fit — building TF-IDF matrix …")
        df = movies_df.copy()
        self._movies_df = df
        self._movie_ids = df["movieId"].tolist()

        # Build a rich text feature: genre tokens + cleaned title words
        df["_genres_text"] = df["genres"].str.replace("|", " ", regex=False)
        df["_title_text"] = df["title"].str.replace(r"\(\d{4}\)", "", regex=True)
        df["_features"] = df["_genres_text"] + " " + df["_title_text"]

        self._tfidf = TfidfVectorizer(
            max_features=self.max_features, stop_words="english", ngram_range=(1, 2)
        )
        self._tfidf_matrix = self._tfidf.fit_transform(df["_features"])

        logger.info(
            f"ContentBasedFilter fitted — {len(self._movie_ids)} movies, "
            f"vocab size {len(self._tfidf.vocabulary_)}"
        )
        return self

    # ------------------------------------------------------------------
    def recommend(self, movie_id: int, n: int = 10) -> Recommendations:
        """Return n movies most similar to movie_id."""
        if movie_id not in self._movie_ids:
            logger.warning(f"movie_id {movie_id} not in training data")
            return []
        idx = self._movie_ids.index(movie_id)
        sim_scores = cosine_similarity(
            self._tfidf_matrix[idx], self._tfidf_matrix
        ).flatten()
        sim_scores[idx] = -1  # exclude self
        top_idx = np.argsort(sim_scores)[::-1][:n]
        return [(self._movie_ids[i], float(sim_scores[i])) for i in top_idx]

    # ------------------------------------------------------------------
    def recommend_by_genre(self, genre_query: str, n: int = 10) -> Recommendations:
        """
        Return top-n movies whose content best matches a free-text genre query.
        Useful for cold-start scenarios where we have no user history.
        """
        query_vec = self._tfidf.transform([genre_query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:n]
        return [(self._movie_ids[i], float(scores[i])) for i in top_idx]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"ContentBasedFilter saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "ContentBasedFilter":
        model = joblib.load(path)
        logger.info(f"ContentBasedFilter loaded <- {path}")
        return model


# ---------------------------------------------------------------------------
# Hybrid Recommender
# ---------------------------------------------------------------------------

class HybridRecommender:
    """
    Weighted linear combination of collaborative and content-based scores.

    For a given user:
      1. CF produces predicted ratings for unseen movies.
      2. The user's top-rated seen movie seeds the CBF for similar items.
      3. Both score lists are min-max normalised and blended.

    For a given movie (no user context):
      - Falls back to pure content-based similarity.
    """

    def __init__(self, cf_weight: float = 0.7, cbf_weight: float = 0.3):
        if abs(cf_weight + cbf_weight - 1.0) > 1e-6:
            raise ValueError("cf_weight + cbf_weight must equal 1.0")
        self.cf_weight = cf_weight
        self.cbf_weight = cbf_weight
        self.cf = CollaborativeFilter()
        self.cbf = ContentBasedFilter()

    # ------------------------------------------------------------------
    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        cf_kwargs: dict = None,
        cbf_kwargs: dict = None,
    ) -> "HybridRecommender":
        cf_kwargs = cf_kwargs or {}
        cbf_kwargs = cbf_kwargs or {}
        if cf_kwargs:
            self.cf = CollaborativeFilter(**cf_kwargs)
        if cbf_kwargs:
            self.cbf = ContentBasedFilter(**cbf_kwargs)
        self.cf.fit(ratings_df)
        self.cbf.fit(movies_df)
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(score_dict: dict) -> dict:
        """Min-max normalise a {id: score} dict to [0, 1]."""
        if not score_dict:
            return {}
        vals = np.array(list(score_dict.values()), dtype=float)
        lo, hi = vals.min(), vals.max()
        if hi == lo:
            return {k: 0.5 for k in score_dict}
        return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}

    # ------------------------------------------------------------------
    def recommend_for_user(self, user_id: int, n: int = 10) -> Recommendations:
        cf_recs = dict(self.cf.recommend(user_id, n=n * 3))

        # Seed CBF with the user's highest-rated seen movie
        cbf_recs: dict = {}
        if hasattr(self.cf, "_user_item_matrix") and user_id in self.cf._user_ids:
            user_ratings = self.cf._user_item_matrix.loc[user_id].dropna()
            if len(user_ratings) > 0:
                seed_movie = int(user_ratings.idxmax())
                cbf_recs = dict(self.cbf.recommend(seed_movie, n=n * 3))

        cf_norm = self._normalize(cf_recs)
        cbf_norm = self._normalize(cbf_recs)

        all_movies = set(cf_norm) | set(cbf_norm)
        hybrid_scores = {
            mid: self.cf_weight * cf_norm.get(mid, 0.0)
                 + self.cbf_weight * cbf_norm.get(mid, 0.0)
            for mid in all_movies
        }
        top = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return top

    # ------------------------------------------------------------------
    def recommend_for_movie(self, movie_id: int, n: int = 10) -> Recommendations:
        """Pure content-based fallback when there is no user context."""
        return self.cbf.recommend(movie_id, n=n)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"HybridRecommender saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "HybridRecommender":
        model = joblib.load(path)
        logger.info(f"HybridRecommender loaded <- {path}")
        return model

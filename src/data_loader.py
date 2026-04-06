"""
Data loading and preprocessing pipeline for the MovieLens dataset.
"""
import os
import re
import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

# Default data directory relative to project root
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "data", "raw", "ml-latest-small")


def load_movies(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load and enrich the movies CSV.

    Adds:
    - year        : int (extracted from title)
    - title_clean : str (title without year suffix)
    - genres_list : list[str]
    """
    path = os.path.join(data_dir, "movies.csv")
    movies = pd.read_csv(path)

    # Extract release year from title string like "Toy Story (1995)"
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype(float)
    movies["title_clean"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
    movies["genres_list"] = movies["genres"].apply(
        lambda g: g.split("|") if g != "(no genres listed)" else []
    )

    logger.info(f"Loaded {len(movies)} movies from {path}")
    return movies


def load_ratings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load the ratings CSV.

    Converts timestamp (Unix seconds) to datetime.
    """
    path = os.path.join(data_dir, "ratings.csv")
    ratings = pd.read_csv(path)
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    logger.info(
        f"Loaded {len(ratings)} ratings — "
        f"{ratings['userId'].nunique()} users, {ratings['movieId'].nunique()} movies"
    )
    return ratings


def load_tags(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load the tags CSV."""
    path = os.path.join(data_dir, "tags.csv")
    tags = pd.read_csv(path)
    tags["timestamp"] = pd.to_datetime(tags["timestamp"], unit="s")
    logger.info(f"Loaded {len(tags)} tags")
    return tags


def load_links(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load IMDB / TMDB cross-reference links."""
    path = os.path.join(data_dir, "links.csv")
    return pd.read_csv(path)


def load_all(data_dir: str = DATA_DIR):
    """Return movies, ratings, tags as a tuple."""
    return load_movies(data_dir), load_ratings(data_dir), load_tags(data_dir)


# ---------------------------------------------------------------------------
# User-item matrix helpers
# ---------------------------------------------------------------------------

def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Build a dense user × movie rating matrix.

    Rows = userId, Columns = movieId, Values = rating (NaN if not rated).
    """
    return ratings.pivot_table(index="userId", columns="movieId", values="rating")


def build_sparse_matrix(ratings: pd.DataFrame):
    """
    Build a CSR sparse matrix and return it together with the index arrays.

    Returns
    -------
    matrix   : csr_matrix  (n_users × n_movies)
    user_ids : list[int]
    movie_ids: list[int]
    """
    dense = build_user_item_matrix(ratings)
    matrix = csr_matrix(dense.fillna(0).values)
    return matrix, dense.index.tolist(), dense.columns.tolist()


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def train_test_split_temporal(
    ratings: pd.DataFrame, test_ratio: float = 0.2
) -> tuple:
    """
    Temporal train/test split — the last *test_ratio* fraction of interactions
    (sorted by timestamp) form the test set.

    This is the most realistic strategy for recommendation systems because it
    simulates production: the model is trained on past data and evaluated on
    future interactions.
    """
    sorted_ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(sorted_ratings) * (1 - test_ratio))
    train = sorted_ratings.iloc[:split_idx].copy()
    test = sorted_ratings.iloc[split_idx:].copy()
    logger.info(f"Train set: {len(train)} ratings | Test set: {len(test)} ratings")
    return train, test


def filter_min_interactions(
    ratings: pd.DataFrame, min_user_ratings: int = 5, min_movie_ratings: int = 5
) -> pd.DataFrame:
    """
    Remove users and movies with fewer than the minimum number of interactions.

    This is standard practice to avoid cold-start noise in evaluation.
    """
    before = len(ratings)
    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()

    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index

    ratings = ratings[
        ratings["userId"].isin(valid_users) & ratings["movieId"].isin(valid_movies)
    ]
    logger.info(
        f"filter_min_interactions: {before} -> {len(ratings)} ratings "
        f"({len(valid_users)} users, {len(valid_movies)} movies)"
    )
    return ratings.reset_index(drop=True)


def get_movie_info(movie_id: int, movies: pd.DataFrame) -> dict:
    """Return a single movie's metadata as a dict, or empty dict if not found."""
    row = movies[movies["movieId"] == movie_id]
    return row.iloc[0].to_dict() if not row.empty else {}

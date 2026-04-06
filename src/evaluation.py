"""
Evaluation metrics for recommendation systems.

Two layers of evaluation are provided:
  1. Rating-prediction quality  — RMSE, MAE  (measures how well the model
     predicts the exact rating a user would give).
  2. Ranking quality            — Precision@K, Recall@K, NDCG@K  (measures
     whether the top-K recommended items are actually relevant to the user).

Ranking metrics are more meaningful for a deployed system because users care
about the order of results, not the raw score.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rating-prediction metrics
# ---------------------------------------------------------------------------

def rmse(y_true: List[float], y_pred: List[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: List[float], y_pred: List[float]) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def evaluate_rating_prediction(
    model,
    test_ratings: pd.DataFrame,
    sample_size: int = 2000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Sample up to *sample_size* test interactions and compute RMSE / MAE.

    The model must implement:
        predict_rating(user_id: int, movie_id: int) -> Optional[float]
    """
    sample = test_ratings.sample(
        min(sample_size, len(test_ratings)), random_state=random_state
    )
    y_true, y_pred = [], []

    for _, row in sample.iterrows():
        pred = model.predict_rating(int(row["userId"]), int(row["movieId"]))
        if pred is not None:
            y_true.append(row["rating"])
            y_pred.append(pred)

    if not y_true:
        logger.warning("evaluate_rating_prediction: no valid predictions produced")
        return {}

    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "n_evaluated": len(y_true),
    }
    logger.info(f"Rating metrics -> {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Ranking metrics (per-user, then averaged)
# ---------------------------------------------------------------------------

def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Fraction of top-k recommended items that are relevant."""
    top_k = recommended[:k]
    hits = len(set(top_k) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Fraction of relevant items that appear in the top-k."""
    top_k = recommended[:k]
    hits = len(set(top_k) & set(relevant))
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain.

    Rewards placing relevant items higher in the ranking.
    """
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in set(relevant)
    )
    ideal_len = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """1 if at least one relevant item is in top-k, else 0."""
    return 1.0 if set(recommended[:k]) & set(relevant) else 0.0


def evaluate_ranking(
    model,
    test_ratings: pd.DataFrame,
    k: int = 10,
    n_users: int = 100,
    min_relevant: int = 3,
    rating_threshold: float = 3.5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Evaluate top-K ranking quality averaged across *n_users* sampled users.

    "Relevant" items are those the user rated ≥ *rating_threshold* in the
    test set.  Users with fewer than *min_relevant* such items are skipped.

    The model must implement:
        recommend(user_id: int, n: int) -> List[Tuple[int, float]]
    """
    rng = np.random.default_rng(random_state)
    all_users = test_ratings["userId"].unique()

    # Only evaluate users the model was trained on (avoids cold-start warnings)
    if hasattr(model, "_user_ids") and model._user_ids is not None:
        all_users = np.array([u for u in all_users if u in model._user_ids])

    if len(all_users) == 0:
        logger.warning("evaluate_ranking: no eligible users found")
        return {}

    sampled_users = rng.choice(all_users, size=min(n_users, len(all_users)), replace=False)

    prec_list, rec_list, ndcg_list, hr_list = [], [], [], []

    for uid in sampled_users:
        relevant = test_ratings[
            (test_ratings["userId"] == uid)
            & (test_ratings["rating"] >= rating_threshold)
        ]["movieId"].tolist()

        if len(relevant) < min_relevant:
            continue

        try:
            recs = model.recommend(int(uid), n=k)
        except Exception:
            continue

        recommended = [mid for mid, _ in recs]
        prec_list.append(precision_at_k(recommended, relevant, k))
        rec_list.append(recall_at_k(recommended, relevant, k))
        ndcg_list.append(ndcg_at_k(recommended, relevant, k))
        hr_list.append(hit_rate_at_k(recommended, relevant, k))

    if not prec_list:
        logger.warning("evaluate_ranking: no users with sufficient relevant items found")
        return {}

    metrics = {
        f"Precision@{k}": float(np.mean(prec_list)),
        f"Recall@{k}":    float(np.mean(rec_list)),
        f"NDCG@{k}":      float(np.mean(ndcg_list)),
        f"HitRate@{k}":   float(np.mean(hr_list)),
        "n_users_evaluated": len(prec_list),
    }
    logger.info(f"Ranking metrics (k={k}) -> {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Coverage  (catalogue diversity)
# ---------------------------------------------------------------------------

def catalogue_coverage(
    model,
    user_ids: List[int],
    total_items: int,
    n: int = 10,
) -> float:
    """
    Percentage of the item catalogue that is recommended to at least one user.

    Low coverage means the model over-concentrates on popular items.
    """
    recommended_items = set()
    for uid in user_ids:
        try:
            recs = model.recommend(uid, n=n)
            recommended_items.update(mid for mid, _ in recs)
        except Exception:
            continue
    return len(recommended_items) / total_items if total_items > 0 else 0.0

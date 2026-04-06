"""
Training pipeline.

Run from the project root:
    python -m src.train

This script:
  1. Loads and preprocesses the MovieLens data
  2. Applies a temporal train/test split
  3. Trains all three models (CF, CBF, Hybrid)
  4. Evaluates each model and prints a report
  5. Saves trained models and metrics to the models/ directory
"""
import json
import logging
import os
import sys
import time

# Allow running as `python src/train.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    filter_min_interactions,
    load_movies,
    load_ratings,
    train_test_split_temporal,
)
from src.evaluation import (
    evaluate_ranking,
    evaluate_rating_prediction,
    catalogue_coverage,
)
from src.models import CollaborativeFilter, ContentBasedFilter, HybridRecommender
from src.utils import setup_logging

# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
# ---------------------------------------------------------------------------


def main() -> dict:
    setup_logging()
    logger = logging.getLogger(__name__)

    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"Models will be saved to: {os.path.abspath(MODELS_DIR)}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading MovieLens data …")
    movies = load_movies()
    ratings = load_ratings()

    logger.info(
        f"Raw data: {len(movies)} movies, {len(ratings)} ratings, "
        f"{ratings['userId'].nunique()} users"
    )

    # ------------------------------------------------------------------
    # 2. Filter cold-start items/users and split
    # ------------------------------------------------------------------
    ratings_filtered = filter_min_interactions(ratings, min_user_ratings=5, min_movie_ratings=5)
    train_ratings, test_ratings = train_test_split_temporal(ratings_filtered, test_ratio=0.2)

    # ------------------------------------------------------------------
    # 3. Collaborative Filter (SVD)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training Collaborative Filter (SVD, 50 factors) …")
    t0 = time.time()
    cf = CollaborativeFilter(n_factors=50, random_state=42)
    cf.fit(train_ratings)
    logger.info(f"CF training time: {time.time() - t0:.1f}s")

    cf_rating_metrics = evaluate_rating_prediction(cf, test_ratings, sample_size=2000)
    cf_ranking_metrics = evaluate_ranking(cf, test_ratings, k=10, n_users=100)
    cf_coverage = catalogue_coverage(
        cf, train_ratings["userId"].unique()[:200].tolist(), len(movies)
    )

    cf_metrics = {**cf_rating_metrics, **cf_ranking_metrics, "Coverage@10": cf_coverage}
    logger.info(f"CF metrics: {cf_metrics}")
    cf.save(os.path.join(MODELS_DIR, "collaborative_filter.pkl"))

    # ------------------------------------------------------------------
    # 4. Content-Based Filter (TF-IDF)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training Content-Based Filter (TF-IDF + cosine) …")
    t0 = time.time()
    cbf = ContentBasedFilter(max_features=5000)
    cbf.fit(movies)
    logger.info(f"CBF training time: {time.time() - t0:.1f}s")

    cbf.save(os.path.join(MODELS_DIR, "content_filter.pkl"))
    logger.info("CBF saved (no rating-prediction evaluation for content-only model)")

    # ------------------------------------------------------------------
    # 5. Hybrid Recommender
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training Hybrid Recommender (CF 70% + CBF 30%) …")
    t0 = time.time()
    hybrid = HybridRecommender(cf_weight=0.7, cbf_weight=0.3)
    hybrid.fit(train_ratings, movies)
    logger.info(f"Hybrid training time: {time.time() - t0:.1f}s")

    hybrid_ranking_metrics = evaluate_ranking(
        # The hybrid .recommend() routes through CF internally
        hybrid.cf,
        test_ratings,
        k=10,
        n_users=100,
    )
    hybrid_coverage = catalogue_coverage(
        hybrid.cf, train_ratings["userId"].unique()[:200].tolist(), len(movies)
    )

    hybrid_metrics = {**hybrid_ranking_metrics, "Coverage@10": hybrid_coverage}
    logger.info(f"Hybrid metrics: {hybrid_metrics}")
    hybrid.save(os.path.join(MODELS_DIR, "hybrid_recommender.pkl"))

    # ------------------------------------------------------------------
    # 6. Save metrics report
    # ------------------------------------------------------------------
    all_metrics = {
        "collaborative_filter": cf_metrics,
        "content_filter": {"note": "no rating-prediction; evaluated at query time"},
        "hybrid_recommender": hybrid_metrics,
    }
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved -> {metrics_path}")

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nCollaborative Filter:")
    for k, v in cf_metrics.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    print("\nHybrid Recommender (ranking only):")
    for k, v in hybrid_metrics.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    print(f"\nModels saved in: {os.path.abspath(MODELS_DIR)}")
    return all_metrics


if __name__ == "__main__":
    main()

"""
Model monitoring utilities.

Covers two concerns:
  1. Performance monitoring  — track RMSE / MAE over time and alert on drift
  2. Data drift detection    — Kolmogorov-Smirnov test on rating distributions
  3. Recommendation diversity — intra-list diversity and catalogue coverage

All events are stored in-memory (lists) and can be exported as DataFrames for
dashboarding or logging to an external store.
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Lightweight performance and drift monitor.

    Usage
    -----
    monitor = ModelMonitor("hybrid_v1")
    monitor.log_performance(y_true, y_pred)
    monitor.detect_data_drift(reference_ratings, new_ratings)
    monitor.log_recommendation_event(user_id=1, recommended=[101, 202, 303])
    print(monitor.summary())
    """

    def __init__(self, model_name: str, alert_rmse_threshold: float = 1.0):
        self.model_name = model_name
        self.alert_rmse_threshold = alert_rmse_threshold

        self._perf_log: List[dict] = []
        self._drift_log: List[dict] = []
        self._rec_log: List[dict] = []

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------

    def log_performance(
        self,
        y_true: List[float],
        y_pred: List[float],
        label: str = "eval",
    ) -> Dict[str, float]:
        """
        Compute RMSE and MAE and append them to the performance log.

        Returns the computed metrics dict.
        """
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae_val = float(mean_absolute_error(y_true, y_pred))

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "label": label,
            "RMSE": rmse,
            "MAE": mae_val,
            "n_samples": len(y_true),
        }
        self._perf_log.append(entry)

        if rmse > self.alert_rmse_threshold:
            logger.warning(
                f"[{self.model_name}] RMSE={rmse:.4f} exceeds threshold "
                f"{self.alert_rmse_threshold} — investigate model quality."
            )
        else:
            logger.info(f"[{self.model_name}] RMSE={rmse:.4f}, MAE={mae_val:.4f}")

        return entry

    def get_performance_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._perf_log)

    # ------------------------------------------------------------------
    # Data drift detection
    # ------------------------------------------------------------------

    def detect_data_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
        alpha: float = 0.05,
        label: str = "ratings",
    ) -> Dict[str, object]:
        """
        Kolmogorov-Smirnov two-sample test between reference and current distributions.

        p-value < alpha  ->  distributions differ  ->  drift detected.
        """
        stat, p_value = ks_2samp(reference.dropna(), current.dropna())
        drift_detected = bool(p_value < alpha)

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "label": label,
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "alpha": alpha,
        }
        self._drift_log.append(entry)

        level = logging.WARNING if drift_detected else logging.INFO
        logger.log(
            level,
            f"[{self.model_name}] Drift check '{label}': "
            f"KS={stat:.4f}, p={p_value:.4f}, drift={drift_detected}",
        )
        return entry

    def get_drift_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._drift_log)

    # ------------------------------------------------------------------
    # Recommendation logging & diversity
    # ------------------------------------------------------------------

    def log_recommendation_event(
        self,
        user_id: int,
        recommended: List[int],
        scores: Optional[List[float]] = None,
    ) -> None:
        """Record a recommendation event for coverage / diversity analysis."""
        self._rec_log.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "recommended": recommended,
                "scores": scores or [],
            }
        )

    def catalogue_coverage(self, total_items: int) -> float:
        """Fraction of the catalogue recommended at least once."""
        all_items = {item for event in self._rec_log for item in event["recommended"]}
        return len(all_items) / total_items if total_items > 0 else 0.0

    def average_list_length(self) -> float:
        if not self._rec_log:
            return 0.0
        return float(np.mean([len(e["recommended"]) for e in self._rec_log]))

    def get_recommendation_df(self) -> pd.DataFrame:
        rows = []
        for e in self._rec_log:
            for i, item in enumerate(e["recommended"]):
                rows.append(
                    {
                        "timestamp": e["timestamp"],
                        "user_id": e["user_id"],
                        "movie_id": item,
                        "rank": i + 1,
                        "score": e["scores"][i] if i < len(e["scores"]) else None,
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        perf = self.get_performance_df()
        drift = self.get_drift_df()
        lines = [f"=== Monitor: {self.model_name} ==="]

        if not perf.empty:
            latest = perf.iloc[-1]
            lines.append(
                f"Latest performance — RMSE: {latest['RMSE']:.4f}, "
                f"MAE: {latest['MAE']:.4f}  (n={latest['n_samples']})"
            )
        else:
            lines.append("No performance data logged yet.")

        if not drift.empty:
            n_drift = drift["drift_detected"].sum()
            lines.append(f"Drift checks: {len(drift)} total, {n_drift} drift events")
        else:
            lines.append("No drift checks logged yet.")

        lines.append(f"Recommendation events logged: {len(self._rec_log)}")
        return "\n".join(lines)

    def to_json(self, path: str) -> None:
        """Persist the full monitor state to a JSON file."""
        payload = {
            "model_name": self.model_name,
            "performance_log": self._perf_log,
            "drift_log": self._drift_log,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Monitor state saved -> {path}")

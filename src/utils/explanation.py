"""
Lightweight z-score attribution for CloudDrift anomaly explanations.

Two-track explainability strategy (from CloudDrift technical spec):

    Track 1 (this module) — Production API:
        For each detected anomaly, compute how many standard deviations
        each feature is from its training normal mean. Return the top N
        features with the highest deviation scores. Fast: microseconds,
        no additional model inference.

    Track 2 (notebooks/06_shap_analysis.ipynb) — Evaluation:
        Full SHAP TreeExplainer on Isolation Forest with waterfall charts
        for the top anomaly windows. Implemented on Day 7.

Usage in the API (/detect endpoint, Day 8):
    reference_stats = load_reference_stats()
    snapshot = {"cpu_util_zscore_mid": 0.92, "value_roc": 0.81, ...}
    attribution = compute_feature_deviation_scores(snapshot, feature_cols,
                                                   reference_stats, n_top=3)
    # → [{"feature": "value_zscore_long", "deviation_score": 3.42, ...}, ...]
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
REFERENCE_STATS_PATH = ARTIFACTS_DIR / "reference_stats.json"


def build_reference_stats(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Compute per-feature mean and std from training normal rows.

    The reference statistics define what "normal" looks like for each
    feature. Used at inference time to compute z-score deviation for
    any incoming telemetry reading.

    Args:
        train_df:     Training feature DataFrame (all rows including anomalies).
        feature_cols: Engineered feature column names.

    Returns:
        Dict mapping feature_name → {"mean": float, "std": float}.
    """
    normal_train = train_df[~train_df["is_anomaly"]]
    stats: dict = {}

    for col in feature_cols:
        col_data = normal_train[col].dropna()
        std = float(col_data.std())
        stats[col] = {
            "mean": float(col_data.mean()),
            "std": std if std > 1e-8 else 1.0,  # guard against zero std
        }

    logger.info(
        "Reference stats built from %s normal training rows, %d features",
        f"{len(normal_train):,}",
        len(stats),
    )
    return stats


def save_reference_stats(
    stats: dict,
    path: str | Path = REFERENCE_STATS_PATH,
) -> None:
    """
    Save reference statistics to JSON.

    Args:
        stats: Output of build_reference_stats().
        path:  Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Reference stats saved: %s (%d features)", path, len(stats))


def load_reference_stats(
    path: str | Path = REFERENCE_STATS_PATH,
) -> dict:
    """
    Load reference statistics from JSON.

    Args:
        path: Path to the saved JSON artifact.

    Returns:
        Reference stats dict.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference stats not found: {path}. Run build_reference_stats() first."
        )
    with open(path) as f:
        return json.load(f)


def compute_feature_deviation_scores(
    snapshot: dict,
    feature_cols: list[str],
    reference_stats: dict,
    n_top: int = 5,
) -> list[dict]:
    """
    Compute z-score deviation for each feature relative to training normal.

    For each feature in the snapshot, computes:
        deviation = |value - training_normal_mean| / training_normal_std

    Returns the top n_top features ranked by deviation descending. A high
    deviation score means this feature was unusually far from its typical
    value in normal operation — making it a likely contributor to the
    anomaly detection.

    This is the API's answer to "why was this flagged?" It is fast
    (arithmetic only, no model inference) and always available regardless
    of which ensemble component drove the score.

    Args:
        snapshot:        Dict mapping feature_name → feature_value.
                         Typically one row of the feature DataFrame as dict.
        feature_cols:    Ordered list of feature column names.
        reference_stats: Output of build_reference_stats() or load_reference_stats().
        n_top:           Number of top contributing features to return.

    Returns:
        List of dicts sorted by deviation_score descending, each containing:
            feature:         Column name.
            value:           Current value in the snapshot.
            deviation_score: |z-score| relative to training normal.
            mean:            Training normal mean.
            std:             Training normal std.
    """
    deviations = []

    for col in feature_cols:
        if col not in reference_stats:
            continue
        value = snapshot.get(col)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue

        stats = reference_stats[col]
        z_score = abs(float(value) - stats["mean"]) / stats["std"]
        deviations.append(
            {
                "feature": col,
                "value": float(value),
                "deviation_score": round(z_score, 4),
                "mean": round(stats["mean"], 4),
                "std": round(stats["std"], 4),
            }
        )

    return sorted(deviations, key=lambda x: x["deviation_score"], reverse=True)[:n_top]


def explain_anomaly_row(
    row: pd.Series,
    feature_cols: list[str],
    reference_stats: dict,
    n_top: int = 5,
) -> dict:
    """
    Produce a complete anomaly explanation for one DataFrame row.

    Convenience wrapper around compute_feature_deviation_scores that accepts
    a pandas Series (one row of the feature DataFrame) instead of a dict.

    Args:
        row:             Single row as pd.Series (from df.iloc[i] or df.loc[i]).
        feature_cols:    Feature column names.
        reference_stats: Output of build_reference_stats().
        n_top:           Number of top features to return.

    Returns:
        Dict with keys:
            top_features:           List from compute_feature_deviation_scores.
            top_feature_names:      Just the feature names (for API response).
            top_deviation_scores:   Feature → deviation_score dict.
    """
    snapshot = {col: row[col] for col in feature_cols if col in row.index}
    top = compute_feature_deviation_scores(
        snapshot, feature_cols, reference_stats, n_top=n_top
    )
    return {
        "top_features": top,
        "top_feature_names": [d["feature"] for d in top],
        "top_deviation_scores": {d["feature"]: d["deviation_score"] for d in top},
    }

"""
Feature engineering pipeline for CloudDrift telemetry data.
Transforms raw time-series into predictive features for anomaly detection.
Implemented: Day 3
"""


def build_rolling_features(df, windows: list[int] = [5, 15, 30]):
    """Compute rolling mean, std, z-score, rate-of-change per window."""
    raise NotImplementedError("Implemented Day 3")


def build_cross_metric_features(df):
    """Compute CPU-memory correlation, interaction ratios, composite volatility."""
    raise NotImplementedError("Implemented Day 3")


def apply_percentile_rank_normalization(df):
    """Normalize all features to [0,1] using percentile rank."""
    raise NotImplementedError("Implemented Day 3")


def build_feature_pipeline():
    """Return a fitted scikit-learn Pipeline with all feature transforms."""
    raise NotImplementedError("Implemented Day 3")

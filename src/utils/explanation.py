"""
Lightweight feature attribution using z-score deviation ranking.
Used in the production API (/detect endpoint).
Full SHAP analysis is in notebooks/06_shap_analysis.ipynb.
"""


def compute_feature_deviation_scores(snapshot: dict, reference_stats: dict) -> list:
    """
    Compute z-score deviation for each metric in the telemetry snapshot
    relative to the reference distribution.

    Returns a ranked list of (feature_name, deviation_score) tuples,
    sorted by deviation_score descending.
    """
    raise NotImplementedError("Implemented Day 7")

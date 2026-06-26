"""
Ensemble scoring: combines Isolation Forest and Autoencoder scores.
Implemented: Day 6
"""


def normalize_score(scores, method: str = "minmax"):
    """Normalize anomaly scores to [0, 1]."""
    raise NotImplementedError("Implemented Day 6")


def compute_ensemble_score(if_score, ae_score, if_weight: float = 0.4):
    """Weighted ensemble: (if_weight * IF) + ((1-if_weight) * AE)."""
    raise NotImplementedError("Implemented Day 6")

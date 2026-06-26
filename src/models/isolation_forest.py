"""
Isolation Forest anomaly detector with TimeSeriesSplit cross-validation.
Implemented: Day 4
"""


def train_isolation_forest(X_train):
    """Train Isolation Forest on normal-behavior windows."""
    raise NotImplementedError("Implemented Day 4")


def run_timeseries_cross_validation(X, y, n_splits: int = 5):
    """Run TimeSeriesSplit cross-validation and return fold metrics."""
    raise NotImplementedError("Implemented Day 4")


def calibrate_threshold(model, X_val, y_val):
    """Calibrate anomaly score threshold using precision-recall curve."""
    raise NotImplementedError("Implemented Day 4")

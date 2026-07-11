"""
Ensemble scoring for CloudDrift anomaly detection.

Combines Isolation Forest anomaly scores and TCN Autoencoder reconstruction
errors into a single weighted ensemble score.

Design:
    Each model provides an independent anomaly signal:
    - Isolation Forest:   point-wise feature-space separation (AUC-ROC=0.801)
    - TCN Autoencoder:    temporal sequence reconstruction error (AUC-ROC=0.869)

Normalization:
    Both raw scores are normalized to [0, 1] before combining using
    percentile bounds fitted on training normal rows. Without normalization,
    scale differences between models distort the weighted average.

NaN handling:
    TCN reconstruction errors are NaN for rows in series shorter than
    seq_length=30. For these rows the ensemble falls back to the IF score
    alone rather than dropping those rows from evaluation.

Severity labels:
    Critical: ensemble_score >= 0.8
    Warning:  ensemble_score >= 0.5
    Normal:   ensemble_score <  0.5
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")

# Ensemble hyperparameters
IF_WEIGHT = 0.40
TCN_WEIGHT = 0.60


# Severity thresholds
SEVERITY_CRITICAL = 0.75
SEVERITY_WARNING = 0.5

# Normalization percentile bounds
NORM_LOWER_PCT = 1.0
NORM_UPPER_PCT = 99.0


# ---------------------------------------------------------------------------
# Score computation helpers
# ---------------------------------------------------------------------------


def compute_if_scores(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """
    Compute Isolation Forest anomaly scores for all rows in df.

    Args:
        model:        Fitted IsolationForest.
        df:           Feature DataFrame with feature_cols columns.
        feature_cols: Feature column names.

    Returns:
        pd.Series of anomaly scores (higher = more anomalous).
    """
    from src.models.isolation_forest import compute_anomaly_scores

    scores = compute_anomaly_scores(model, df[feature_cols])
    return pd.Series(scores, index=df.index, name="if_score")


def compute_tcn_scores(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """
    Compute TCN Autoencoder reconstruction errors for all rows in df.

    Args:
        model:        Fitted TCNAutoencoder.
        df:           Feature DataFrame with source_file, timestamp, feature_cols.
        feature_cols: Feature column names.

    Returns:
        pd.Series of reconstruction errors (higher = more anomalous).
        NaN for rows in series shorter than seq_length=30.
    """
    from src.models.tcn_autoencoder import compute_reconstruction_errors

    return compute_reconstruction_errors(model, df, feature_cols)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def fit_score_bounds(
    scores: pd.Series,
    lower_pct: float = NORM_LOWER_PCT,
    upper_pct: float = NORM_UPPER_PCT,
) -> tuple[float, float]:
    """
    Fit [lower_pct, upper_pct] percentile bounds from a score distribution.

    Fitted on training NORMAL scores so the normalization aligns:
        near 0 → typical normal behavior
        near 1 → typical extreme-normal behavior (p99)
        > 1    → anomaly territory (clipped to 1.0 after normalization)

    Args:
        scores:     Raw score Series (usually from normal training rows).
        lower_pct:  Lower percentile for the normalization floor (default 1st).
        upper_pct:  Upper percentile for the normalization ceiling (default 99th).

    Returns:
        Tuple (lower_bound, upper_bound).
    """
    clean = scores.dropna()
    lo = float(np.nanpercentile(clean, lower_pct))
    hi = float(np.nanpercentile(clean, upper_pct))
    if hi <= lo:
        hi = lo + 1e-8
    return lo, hi


def normalize_scores(
    scores: pd.Series,
    lo: float,
    hi: float,
) -> pd.Series:
    """
    Normalize scores to [0, 1] using pre-fitted percentile bounds.

    Clips values to [lo, hi] then scales linearly so lo→0 and hi→1.
    Values above hi (anomaly territory) are clipped to 1.0.
    NaN values are preserved.

    Args:
        scores: Raw score Series.
        lo:     Lower bound (from fit_score_bounds).
        hi:     Upper bound (from fit_score_bounds).

    Returns:
        Normalized Series with values in [0, 1] (NaN preserved).
    """
    clipped = scores.clip(lo, hi)
    return (clipped - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Ensemble combination
# ---------------------------------------------------------------------------


def compute_ensemble_score(
    if_norm: pd.Series,
    tcn_norm: pd.Series,
    if_weight: float = IF_WEIGHT,
) -> pd.Series:
    """
    Combine normalized IF and TCN scores into a weighted ensemble score.

    NaN handling: rows where TCN score is NaN (series too short for windowing)
    receive the IF score only rather than being dropped from evaluation.

    Args:
        if_norm:   Normalized IF anomaly scores (all values in [0, 1]).
        tcn_norm:  Normalized TCN reconstruction errors ([0, 1], NaN allowed).
        if_weight: Weight for the IF component (default 0.4).

    Returns:
        Ensemble score Series with values in [0, 1], no NaN.
    """
    tcn_weight = 1.0 - if_weight
    combined = if_weight * if_norm + tcn_weight * tcn_norm
    # Fallback to IF score where TCN is NaN
    combined = combined.fillna(if_norm)
    return combined.rename("ensemble_score")


def get_severity_label(score: float) -> str:
    """
    Convert a continuous ensemble score to a severity label.

    Args:
        score: Ensemble score in [0, 1].

    Returns:
        "Critical", "Warning", or "Normal".
    """
    if score >= SEVERITY_CRITICAL:
        return "Critical"
    if score >= SEVERITY_WARNING:
        return "Warning"
    return "Normal"


# ---------------------------------------------------------------------------
# Threshold calibration and evaluation
# ---------------------------------------------------------------------------


def calibrate_ensemble_threshold(
    ensemble_scores: pd.Series,
    anomaly_rate: float,
) -> float:
    """
    Calibrate the ensemble score threshold using contamination-based strategy.

    Flags the top (anomaly_rate × 1.5) percent of ensemble scores.
    Consistent with the threshold strategy used for IF and TCN individually.

    Args:
        ensemble_scores: Ensemble score Series on the validation set.
        anomaly_rate:    Observed anomaly rate in the validation set.

    Returns:
        Float threshold.
    """
    contamination = min(max(anomaly_rate * 1.5, 0.01), 0.10)
    threshold = float(np.percentile(ensemble_scores.values, (1 - contamination) * 100))
    flagged = int((ensemble_scores >= threshold).sum())
    logger.info(
        "Ensemble threshold (contamination=%.1f%%): %.6f → flags %d rows (%.1f%%)",
        contamination * 100,
        threshold,
        flagged,
        flagged / len(ensemble_scores) * 100,
    )
    return threshold


def evaluate_ensemble(
    ensemble_scores: pd.Series,
    threshold: float,
    y_true: pd.Series,
    split_name: str = "split",
) -> dict:
    """
    Evaluate ensemble performance on a labelled split.

    Args:
        ensemble_scores: Ensemble score Series for all rows in the split.
        threshold:       Calibrated threshold.
        y_true:          True anomaly labels (bool or int Series).
        split_name:      Label used in logging.

    Returns:
        Metrics dict with precision, recall, F1, F2, AUC-ROC.
    """
    from sklearn.metrics import (
        f1_score,
        fbeta_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_int = y_true.astype(int).values
    scores = ensemble_scores.values
    y_pred = (scores >= threshold).astype(int)

    precision = float(precision_score(y_int, y_pred, zero_division=0))
    recall = float(recall_score(y_int, y_pred, zero_division=0))
    f1 = float(f1_score(y_int, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_int, y_pred, beta=2, zero_division=0))

    try:
        auc_roc = float(roc_auc_score(y_int, scores))
    except ValueError:
        auc_roc = float("nan")

    metrics = {
        "split": split_name,
        "n_rows": int(len(y_int)),
        "n_anomaly_true": int(y_int.sum()),
        "n_anomaly_predicted": int(y_pred.sum()),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "auc_roc": auc_roc,
    }

    logger.info(
        "%s | P=%.3f R=%.3f F1=%.3f F2=%.3f AUC-ROC=%.3f",
        split_name.upper(),
        precision,
        recall,
        f1,
        f2,
        auc_roc if not np.isnan(auc_roc) else 0,
    )
    return metrics


# ---------------------------------------------------------------------------
# Anomaly ranking
# ---------------------------------------------------------------------------


def rank_top_anomalies(
    df: pd.DataFrame,
    ensemble_scores: pd.Series,
    threshold: float,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the top N rows by ensemble score above the threshold.

    Provides the ranked anomaly list that drives the Streamlit dashboard
    and the /batch_detect API response.

    Args:
        df:               Feature DataFrame with timestamp, source_file, is_anomaly.
        ensemble_scores:  Ensemble score Series aligned to df.index.
        threshold:        Calibrated threshold.
        n:                Number of top anomalies to return.

    Returns:
        DataFrame with top N rows sorted by ensemble_score descending,
        containing: timestamp, source_file, ensemble_score, severity_label,
        is_anomaly (ground truth if available).
    """
    flagged_mask = ensemble_scores >= threshold
    flagged_df = df[flagged_mask].copy()
    flagged_df["ensemble_score"] = ensemble_scores[flagged_mask]
    flagged_df["severity_label"] = flagged_df["ensemble_score"].apply(
        get_severity_label
    )

    cols = ["timestamp", "source_file", "ensemble_score", "severity_label"]
    if "is_anomaly" in flagged_df.columns:
        cols.append("is_anomaly")

    top = (
        flagged_df[cols]
        .sort_values("ensemble_score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    logger.info(
        "Top %d anomalies ranked | highest score: %.4f | severity breakdown: %s",
        len(top),
        float(top["ensemble_score"].max()) if len(top) > 0 else 0,
        top["severity_label"].value_counts().to_dict() if len(top) > 0 else {},
    )
    return top


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_ensemble_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    if_weight: float = IF_WEIGHT,
) -> dict:
    """
    Full ensemble pipeline: load models, normalize, combine, evaluate.

    Loads IF and TCN artifacts from the standard artifacts directory,
    fits normalization bounds on training normal rows, computes ensemble
    scores on validation and test sets, calibrates threshold, and
    evaluates performance.

    Args:
        train_df:     Training feature DataFrame.
        val_df:       Validation feature DataFrame.
        test_df:      Test feature DataFrame.
        feature_cols: Engineered feature column names.
        if_weight:    IF model weight in the ensemble (default 0.4).

    Returns:
        Dict with keys: val_metrics, test_metrics, threshold, if_bounds,
        tcn_bounds, if_weight, tcn_weight, top_val_anomalies.
    """
    from src.models.isolation_forest import load_isolation_forest
    from src.models.tcn_autoencoder import load_tcn_autoencoder

    logger.info("Loading model artifacts...")
    if_model = load_isolation_forest()
    tcn_model = load_tcn_autoencoder()

    # ── 1. Compute raw scores ─────────────────────────────────────────────
    logger.info("Computing IF scores on all splits...")
    train_normal = train_df[~train_df["is_anomaly"]]
    if_train_normal = compute_if_scores(if_model, train_normal, feature_cols)
    if_val = compute_if_scores(if_model, val_df, feature_cols)
    if_test = compute_if_scores(if_model, test_df, feature_cols)

    logger.info("Computing TCN reconstruction errors on all splits...")
    tcn_train_normal = compute_tcn_scores(tcn_model, train_normal, feature_cols)
    tcn_val = compute_tcn_scores(tcn_model, val_df, feature_cols)
    tcn_test = compute_tcn_scores(tcn_model, test_df, feature_cols)

    # ── 2. Fit normalization bounds on training normal scores ─────────────
    logger.info("Fitting normalization bounds on training normal scores...")
    if_bounds = fit_score_bounds(if_train_normal)
    tcn_bounds = fit_score_bounds(tcn_train_normal)

    logger.info("IF  bounds: [%.4f, %.4f]", if_bounds[0], if_bounds[1])
    logger.info("TCN bounds: [%.6f, %.6f]", tcn_bounds[0], tcn_bounds[1])

    # ── 3. Normalize ──────────────────────────────────────────────────────
    if_val_norm = normalize_scores(if_val, *if_bounds)
    if_test_norm = normalize_scores(if_test, *if_bounds)

    tcn_val_norm = normalize_scores(tcn_val, *tcn_bounds)
    tcn_test_norm = normalize_scores(tcn_test, *tcn_bounds)

    # ── 4. Compute ensemble scores ────────────────────────────────────────
    val_ensemble = compute_ensemble_score(if_val_norm, tcn_val_norm, if_weight)
    test_ensemble = compute_ensemble_score(if_test_norm, tcn_test_norm, if_weight)

    logger.info(
        "Ensemble scores — val: [%.3f, %.3f]  test: [%.3f, %.3f]",
        float(val_ensemble.min()),
        float(val_ensemble.max()),
        float(test_ensemble.min()),
        float(test_ensemble.max()),
    )

    # ── 5. Calibrate threshold on validation ──────────────────────────────
    val_anomaly_rate = float(val_df["is_anomaly"].mean())
    threshold = calibrate_ensemble_threshold(val_ensemble, val_anomaly_rate)

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    val_metrics = evaluate_ensemble(
        val_ensemble, threshold, val_df["is_anomaly"], "validation"
    )
    test_metrics = evaluate_ensemble(
        test_ensemble, threshold, test_df["is_anomaly"], "test"
    )

    # ── 7. Rank top anomalies on validation set ───────────────────────────
    top_anomalies = rank_top_anomalies(val_df, val_ensemble, threshold, n=10)

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "threshold": threshold,
        "if_bounds": {"lower": if_bounds[0], "upper": if_bounds[1]},
        "tcn_bounds": {"lower": tcn_bounds[0], "upper": tcn_bounds[1]},
        "if_weight": if_weight,
        "tcn_weight": 1.0 - if_weight,
        "top_val_anomalies": top_anomalies,
    }


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def save_ensemble_metadata(
    metadata: dict,
    path: str | Path = ARTIFACTS_DIR / "ensemble_metadata.json",
) -> None:
    """
    Save ensemble configuration and normalization bounds to JSON.

    Args:
        metadata: Dict from run_ensemble_pipeline() (excluding DataFrames).
        path:     Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {k: v for k, v in metadata.items() if k != "top_val_anomalies"}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info("Ensemble metadata saved: %s", path)


def load_ensemble_metadata(
    path: str | Path = ARTIFACTS_DIR / "ensemble_metadata.json",
) -> dict:
    """
    Load ensemble configuration from JSON.

    Args:
        path: Path to the saved JSON artifact.

    Returns:
        Ensemble metadata dict.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Ensemble metadata not found: {path}. Run run_ensemble_pipeline() first."
        )
    with open(path) as f:
        return json.load(f)

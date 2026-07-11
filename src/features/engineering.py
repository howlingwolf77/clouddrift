"""
Feature engineering pipeline for CloudDrift.

Transforms raw server telemetry into 68 predictive features for anomaly
detection. Features are computed per-machine to prevent cross-contamination
between independent time-series.

Designed for SMD (Server Machine Dataset) and Alibaba Cluster Trace data:
    cpu_util, mem_util, net_io_in, net_io_out, disk_io

Feature categories:
    Rolling statistics (short=1, mid=3, long=6 steps):
        mean, std, z-score, rate-of-change, range ratio
    Cross-metric features:
        CPU-memory rolling correlation, CPU-network ratio,
        composite volatility score
    Normalization:
        RobustPercentileNormalizer fitted on training data —
        clips to [p1, p99] then scales to [0, 1]

Output: 68 feature columns (5 raw + 60 rolling + 3 cross-metric).
This maps directly to the TCN Autoencoder's input_dim=68.

Pipeline artifact:
    artifacts/feature_pipeline.joblib

Usage:
    # Fit on training data and transform all splits
    train_feat = build_alibaba_features(train_df, group_col="machine_id")
    feature_cols = get_feature_columns(train_feat)
    pipeline = build_feature_pipeline(train_feat, feature_cols)
    train_norm = apply_feature_pipeline(pipeline, train_feat, feature_cols)
    save_feature_pipeline(pipeline)

    # At inference time
    pipeline = load_feature_pipeline()
    snapshot_feat = build_alibaba_features(snapshot_df, group_col="machine_id")
    snapshot_norm = apply_feature_pipeline(pipeline, snapshot_feat, feature_cols)
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rolling window sizes in steps.
# Step-based (not time-based) so the same code works across different
# sampling rates (SMD: 1-minute intervals, Alibaba: irregular).
ROLLING_WINDOWS: dict[str, int] = {"short": 1, "mid": 3, "long": 6}

# Columns that are metadata or target labels — never used as model features
METADATA_COLS: frozenset[str] = frozenset(
    [
        "timestamp",
        "source_file",
        "metric_name",
        "category",
        "is_anomaly",
        "machine_id",
        "time_stamp",
        "time_step",
    ]
)

# CloudDrift standard metric columns for server telemetry.
# Matches both SMD (5 selected columns) and Alibaba (after renaming).
ALIBABA_METRIC_COLS = [
    "cpu_util",
    "mem_util",
    "net_io_in",
    "net_io_out",
    "disk_io",
]

# Forward-fill limit: maximum consecutive missing steps to fill.
# 3 steps at 1-minute intervals = 3 minutes. Larger gaps are left as NaN
# and handled by the normalizer (filled with 0 after normalization).
FORWARD_FILL_LIMIT = 3


# ---------------------------------------------------------------------------
# Public interface — feature engineering
# ---------------------------------------------------------------------------


def build_alibaba_features(
    df: pd.DataFrame,
    group_col: str = "machine_id",
) -> pd.DataFrame:
    """
    Build rolling statistical and cross-metric features for server telemetry.

    Works on both SMD and Alibaba data formats since both share the same
    CloudDrift standard column names (cpu_util, mem_util, net_io_in,
    net_io_out, disk_io). Features are computed within each machine group
    independently to prevent contamination between machines.

    Multi-metric data enables cross-metric features (CPU-memory correlation,
    CPU-network ratio, composite volatility) that are unavailable in
    single-metric datasets.

    Output: 68 feature columns per row:
        5 raw metric columns
        5 metrics × 12 rolling features = 60 rolling features
        3 cross-metric features

    Args:
        df:        DataFrame from load_smd_dataset() or load_alibaba_cluster_trace().
        group_col: Column that identifies independent machines.
                   Default: "machine_id".

    Returns:
        DataFrame with original columns plus 68 engineered feature columns.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found.")

    available_metrics = [c for c in ALIBABA_METRIC_COLS if c in df.columns]
    if not available_metrics:
        raise ValueError(
            f"No metric columns found. Expected one of: {ALIBABA_METRIC_COLS}"
        )

    groups = []
    n_groups = df[group_col].nunique()
    logger.info(
        "Building Alibaba features for %d machines (grouped by %s)...",
        n_groups,
        group_col,
    )

    for machine_id, group_df in df.groupby(group_col):
        group_df = group_df.sort_values("timestamp").copy()

        for col in available_metrics:
            group_df[col] = group_df[col].ffill(limit=FORWARD_FILL_LIMIT)

        for metric_col in available_metrics:
            group_df = _compute_rolling_features(group_df, value_col=metric_col)

        group_df = _compute_cross_metric_features(group_df, available_metrics)

        groups.append(group_df)

    result = pd.concat(groups, ignore_index=True)

    n_features = len(get_feature_columns(result))
    logger.info(
        "Alibaba feature engineering complete: %s rows, %d feature columns",
        f"{len(result):,}",
        n_features,
    )
    return result


# ---------------------------------------------------------------------------
# Public interface — feature columns
# ---------------------------------------------------------------------------


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of engineered feature columns from a feature DataFrame.

    Excludes all metadata columns (timestamp, labels, identifiers).
    The returned list is the input feature set for model training and
    maps to the TCN Autoencoder's input_dim.

    Args:
        df: DataFrame after build_alibaba_features().

    Returns:
        Sorted list of feature column names.
    """
    return sorted([c for c in df.columns if c not in METADATA_COLS])


# ---------------------------------------------------------------------------
# Public interface — feature pipeline
# ---------------------------------------------------------------------------


def build_feature_pipeline(
    train_features: pd.DataFrame,
    feature_cols: list[str],
) -> Pipeline:
    """
    Fit a normalization pipeline on training feature data.

    The fitted pipeline stores [p1, p99] bounds from the training
    distribution per feature. At inference time, new data is clipped
    to those bounds and scaled to [0, 1].

    Fitting on training data only (not val or test) prevents leakage
    of future distribution information into the normalization step.

    Args:
        train_features: Feature DataFrame from the training split.
        feature_cols:   Feature column names (from get_feature_columns).

    Returns:
        Fitted sklearn Pipeline containing RobustPercentileNormalizer.
    """
    x_train = train_features[feature_cols].copy()
    x_train = x_train.fillna(0.0)

    normalizer = RobustPercentileNormalizer(lower_pct=1.0, upper_pct=99.0)
    pipeline = Pipeline([("normalizer", normalizer)])
    pipeline.fit(x_train)

    logger.info(
        "Feature pipeline fitted on %s training rows, %d features",
        f"{len(x_train):,}",
        len(feature_cols),
    )
    return pipeline


def apply_feature_pipeline(
    pipeline: Pipeline,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Apply a fitted feature pipeline to any data split.

    Returns the full DataFrame with feature columns replaced by their
    normalised values. Metadata columns (timestamp, labels) are preserved
    unchanged.

    Args:
        pipeline:     Fitted sklearn Pipeline from build_feature_pipeline().
        feature_df:   Feature DataFrame to transform.
        feature_cols: Feature column names used when the pipeline was fitted.

    Returns:
        DataFrame with normalised feature columns and unchanged metadata.
    """
    result = feature_df.copy()
    x = result[feature_cols].fillna(0.0)
    x_norm = pipeline.transform(x)
    result[feature_cols] = x_norm
    return result


def save_feature_pipeline(
    pipeline: Pipeline,
    path: str | Path = ARTIFACTS_DIR / "feature_pipeline.joblib",
) -> None:
    """
    Save the fitted feature pipeline to disk as a joblib artifact.

    Args:
        pipeline: Fitted sklearn Pipeline.
        path:     Destination path. Default: artifacts/feature_pipeline.joblib.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    size_kb = path.stat().st_size / 1024
    logger.info("Feature pipeline saved: %s (%.1f KB)", path, size_kb)


def load_feature_pipeline(
    path: str | Path = ARTIFACTS_DIR / "feature_pipeline.joblib",
) -> Pipeline:
    """
    Load a previously fitted feature pipeline from disk.

    Args:
        path: Path to the saved joblib artifact.

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature pipeline artifact not found: {path}. "
            "Run build_feature_pipeline() and save_feature_pipeline() first."
        )
    pipeline = joblib.load(path)
    logger.info("Feature pipeline loaded from: %s", path)
    return pipeline


# ---------------------------------------------------------------------------
# RobustPercentileNormalizer — custom sklearn transformer
# ---------------------------------------------------------------------------


class RobustPercentileNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize features using percentile bounds fitted on training data.

    Algorithm:
        1. Fit: compute p1 and p99 of each feature from training data.
        2. Transform: clip values to [p1, p99] then scale to [0, 1].

    Why this approach:
        - Robust to outliers: extreme values are clipped rather than
          distorting the entire distribution.
        - Consistent: the same bounds apply at training and inference time,
          preventing distribution shift between environments.
        - Interpretable: 0 = lowest seen in training, 1 = highest seen.
        - Handles heterogeneous scales across different server metrics
          (CPU %, memory %, network rates, disk I/O).

    Args:
        lower_pct: Lower percentile for the clip floor (default 1st).
        upper_pct: Upper percentile for the clip ceiling (default 99th).
    """

    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, x: pd.DataFrame, y=None) -> "RobustPercentileNormalizer":
        """
        Compute per-feature percentile bounds from training data.

        Args:
            x: Feature DataFrame (feature columns only, no metadata).
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            self (fitted transformer).
        """
        self.feature_names_in_: list[str] = list(x.columns)
        self.bounds_: dict[str, tuple[float, float]] = {}

        for col in x.columns:
            lo = float(x[col].quantile(self.lower_pct / 100))
            hi = float(x[col].quantile(self.upper_pct / 100))
            # Guard against constant columns, NaN, and Inf bounds.
            # NaN arises in cpu_mem_corr_long when rolling Pearson correlation
            # is undefined (min_periods=2 not met at series start).
            import math

            if math.isnan(lo) or math.isnan(hi) or math.isinf(lo) or math.isinf(hi):
                lo, hi = 0.0, 1.0
            elif hi <= lo:
                hi = lo + 1.0
            self.bounds_[col] = (lo, hi)

        logger.debug(
            "RobustPercentileNormalizer fitted on %d features, %s rows",
            len(self.feature_names_in_),
            f"{len(x):,}",
        )
        return self

    def transform(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Apply clip-and-scale normalization using fitted bounds.

        Args:
            x: Feature DataFrame. Columns not seen during fit receive 0.0.
            y: Ignored.

        Returns:
            Normalized DataFrame with all features in [0, 1].
        """
        result = x.copy()
        for col in self.feature_names_in_:
            if col not in result.columns:
                result[col] = 0.0
                continue
            lo, hi = self.bounds_[col]
            clipped = result[col].clip(lo, hi)
            result[col] = (clipped - lo) / (hi - lo)
        return result

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """sklearn API: return output feature names."""
        return self.feature_names_in_


# ---------------------------------------------------------------------------
# Private helpers — rolling feature computation
# ---------------------------------------------------------------------------


def _compute_rolling_features(
    df: pd.DataFrame,
    value_col: str,
    windows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Compute rolling statistical features for one metric column.

    All rolling operations use min_periods=1 so short series at the start
    of each machine still produce estimates rather than NaN.

    Features computed:
        {col}_mean_{short|mid|long}         rolling mean
        {col}_std_{short|mid|long}          rolling std (0 if too few points)
        {col}_zscore_{short|mid|long}       (value - mean) / std
        {col}_roc                           rate of change (first diff)
        {col}_range_ratio_{mid|long}        rolling_min / rolling_max

    Args:
        df:        DataFrame sorted by timestamp (already sorted by caller).
        value_col: Name of the metric column to compute features on.
        windows:   Dict mapping name → step count. Default: ROLLING_WINDOWS.

    Returns:
        DataFrame with original columns plus the new feature columns.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    df = df.copy()
    series = df[value_col]

    for name, w in windows.items():
        roll = series.rolling(window=w, min_periods=1)

        mean = roll.mean()
        std = roll.std().fillna(0.0)

        df[f"{value_col}_mean_{name}"] = mean
        df[f"{value_col}_std_{name}"] = std

        safe_std = std.replace(0.0, float("nan"))
        df[f"{value_col}_zscore_{name}"] = ((series - mean) / safe_std).fillna(0.0)

    df[f"{value_col}_roc"] = series.diff().fillna(0.0)

    for name, w in [("mid", windows["mid"]), ("long", windows["long"])]:
        roll_w = series.rolling(window=w, min_periods=1)
        rolling_min = roll_w.min()
        rolling_max = roll_w.max()
        safe_max = rolling_max.replace(0.0, float("nan"))
        df[f"{value_col}_range_ratio_{name}"] = (
            (rolling_min / safe_max).fillna(1.0).clip(0.0, 1.0)
        )

    return df


def _compute_cross_metric_features(
    df: pd.DataFrame,
    available_metrics: list[str],
) -> pd.DataFrame:
    """
    Compute cross-metric features for multi-metric server telemetry.

    These features capture relationships between metrics invisible when
    each metric is analyzed in isolation:
    - CPU and memory usually move together; a breakdown in their correlation
      signals an unusual process consuming one but not the other.
    - CPU-to-network ratio: unexpectedly high network traffic without CPU
      load (or vice versa) indicates anomalous application behavior.
    - Composite volatility: all metrics simultaneously more volatile than
      normal is a stronger anomaly signal than any single metric alone.

    Args:
        df:                DataFrame for one machine (sorted by timestamp).
        available_metrics: List of metric columns actually present in df.

    Returns:
        DataFrame with cross-metric feature columns added.
    """
    df = df.copy()
    w = ROLLING_WINDOWS["long"]

    if "cpu_util" in available_metrics and "mem_util" in available_metrics:
        df["cpu_mem_corr_long"] = (
            df["cpu_util"]
            .rolling(window=w, min_periods=2)
            .corr(df["mem_util"])
            .fillna(0.0)
        )

    if "cpu_util" in available_metrics and "net_io_in" in available_metrics:
        safe_net = df["net_io_in"].replace(0.0, float("nan"))
        df["cpu_net_ratio"] = (df["cpu_util"] / safe_net).fillna(0.0).clip(0.0, 100.0)

    std_cols = [
        f"{m}_std_long" for m in available_metrics if f"{m}_std_long" in df.columns
    ]
    if std_cols:
        df["volatility_score"] = df[std_cols].mean(axis=1)

    return df

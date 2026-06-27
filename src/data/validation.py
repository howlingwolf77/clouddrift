"""
Pandera schema validation and data quality checks for CloudDrift.

Validation runs as a blocking gate before any feature engineering:
bad data raises an exception and stops the pipeline rather than
silently corrupting model training.

Key schemas:
    NAB_RAW_SCHEMA      — validates the unified NAB DataFrame
    ALIBABA_RAW_SCHEMA  — validates the Alibaba DataFrame (if present)

Key functions:
    validate_nab_schema()          — runs Pandera; raises on failure
    validate_null_rates()          — null % per column vs threshold
    validate_timestamp_continuity()— gap detection in time-series
    generate_data_quality_report() — combines all checks into one dict
    define_temporal_split()        — time-ordered train/val/test split
"""

import logging

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MAX_NULL_PCT = 5.0  # flag columns with more than 5% missing values
MAX_GAP_MULTIPLIER = 2  # flag gaps longer than 2× the median interval
NEGATIVE_VALUE_COLS = [  # columns that must never go below zero
    "net_io_in",
    "net_io_out",
    "disk_io",
]


# ---------------------------------------------------------------------------
# Pandera schemas
# ---------------------------------------------------------------------------

NAB_RAW_SCHEMA = DataFrameSchema(
    columns={
        "timestamp": Column(
            pa.DateTime,
            nullable=False,
        ),
        "value": Column(
            pa.Float,
            checks=Check.ge(0),
            nullable=True,  # small number of nulls acceptable at ingestion
        ),
        "metric_name": Column(pa.String, nullable=False),
        "category": Column(pa.String, nullable=False),
        "source_file": Column(pa.String, nullable=False),
        "is_anomaly": Column(pa.Bool, nullable=False),
    },
    coerce=True,  # attempt type coercion before failing
    strict=False,  # allow extra columns
)

ALIBABA_RAW_SCHEMA = DataFrameSchema(
    columns={
        "cpu_util": Column(
            pa.Float,
            checks=[Check.ge(0), Check.le(100)],
            nullable=True,
        ),
        "mem_util": Column(
            pa.Float,
            checks=[Check.ge(0), Check.le(100)],
            nullable=True,
        ),
        # net_in/net_out are normalised to [0, 100] per Alibaba schema
        "net_io_in": Column(
            pa.Float,
            checks=[Check.ge(0), Check.le(100)],
            nullable=True,
        ),
        "net_io_out": Column(
            pa.Float,
            checks=[Check.ge(0), Check.le(100)],
            nullable=True,
        ),
        # disk_io is nullable: sentinel values (-1, 101) are replaced
        # with NaN in _normalise_alibaba_columns before this runs
        "disk_io": Column(
            pa.Float,
            checks=[Check.ge(0), Check.le(100)],
            nullable=True,
        ),
    },
    coerce=True,
    strict=False,  # machine_id and timestamp allowed as extra columns
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_nab_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Pandera schema validation on the NAB DataFrame.

    This is a blocking gate — raises pa.errors.SchemaErrors if
    the DataFrame does not match NAB_RAW_SCHEMA.

    Args:
        df: Output of load_nab_dataset()

    Returns:
        Validated DataFrame (same data, Pandera-typed)

    Raises:
        pa.errors.SchemaErrors: On validation failure (lazy mode —
            all errors collected before raising).
    """
    logger.info("Running Pandera NAB schema validation...")
    try:
        validated = NAB_RAW_SCHEMA.validate(df, lazy=True)
        logger.info("NAB schema validation passed — %s rows validated", f"{len(df):,}")
        return validated
    except pa.errors.SchemaErrors as exc:
        n_failures = len(exc.failure_cases)
        logger.error("NAB schema validation FAILED — %d failure cases", n_failures)
        logger.error("Failure summary:\n%s", exc.failure_cases.head(10).to_string())
        raise


def validate_alibaba_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Pandera schema validation on the Alibaba DataFrame.

    Args:
        df: Output of load_alibaba_cluster_trace()

    Returns:
        Validated DataFrame
    """
    logger.info("Running Pandera Alibaba schema validation...")
    try:
        validated = ALIBABA_RAW_SCHEMA.validate(df, lazy=True)
        logger.info("Alibaba schema validation passed — %s rows", f"{len(df):,}")
        return validated
    except pa.errors.SchemaErrors as exc:
        logger.error(
            "Alibaba schema validation FAILED:\n%s", exc.failure_cases.head(10)
        )
        raise


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------


def validate_null_rates(
    df: pd.DataFrame,
    max_null_pct: float = MAX_NULL_PCT,
) -> dict:
    """
    Check null percentage per column against the threshold.

    Args:
        df:           DataFrame to check.
        max_null_pct: Maximum acceptable null percentage (default 5%).

    Returns:
        Dict mapping column name → {null_pct, pass, threshold}.
        Pass is True when null_pct <= max_null_pct.
    """
    null_rates = (df.isnull().mean() * 100).round(3)
    results = {}

    for col, rate in null_rates.items():
        passed = float(rate) <= max_null_pct
        results[col] = {
            "null_pct": float(rate),
            "pass": passed,
            "threshold_pct": max_null_pct,
        }
        if not passed:
            logger.warning(
                "NULL RATE EXCEEDED — column '%s': %.1f%% (threshold %.0f%%)",
                col,
                rate,
                max_null_pct,
            )

    passing = sum(1 for v in results.values() if v["pass"])
    logger.info("Null rate check: %d/%d columns pass", passing, len(results))
    return results


def validate_timestamp_continuity(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    max_gap_multiplier: float = MAX_GAP_MULTIPLIER,
    group_col: str | None = None,
) -> dict:
    """
    Detect timestamp gaps larger than max_gap_multiplier × median interval.

    For single time-series data, gaps are checked across all rows.
    For concatenated multi-series data (e.g. NAB with many source files),
    pass group_col="source_file" to check continuity within each series
    independently. Inter-series gaps are expected and should not fail.

    Args:
        df:                 DataFrame containing the timestamp column.
        timestamp_col:      Name of the timestamp column.
        max_gap_multiplier: Flag gaps > this multiple of the median interval.
        group_col:          Optional column to group by before checking gaps.
                            Use "source_file" for NAB to check per time-series.

    Returns:
        Dict with gap statistics and pass/fail flag.
    """
    if timestamp_col not in df.columns:
        return {"error": f"Column '{timestamp_col}' not found", "pass": False}

    # Per-series mode: group by group_col and check each series independently
    if group_col and group_col in df.columns:
        return _validate_continuity_per_series(
            df, timestamp_col, max_gap_multiplier, group_col
        )

    # Single-series mode: check all rows together
    ts = pd.to_datetime(df[timestamp_col]).sort_values().reset_index(drop=True)
    gaps_minutes = ts.diff().dt.total_seconds().div(60).dropna()

    if gaps_minutes.empty:
        return {"error": "Not enough timestamps to compute gaps", "pass": False}

    median_gap = gaps_minutes.median()
    threshold = median_gap * max_gap_multiplier
    large_gaps = gaps_minutes[gaps_minutes > threshold]

    result = {
        "mode": "single_series",
        "total_intervals": len(gaps_minutes),
        "median_interval_minutes": round(float(median_gap), 2),
        "threshold_minutes": round(float(threshold), 2),
        "large_gap_count": len(large_gaps),
        "max_gap_minutes": round(float(gaps_minutes.max()), 2),
        "mean_gap_minutes": round(float(gaps_minutes.mean()), 2),
        # Always True — continuity is informational; overall_pass excludes it.
        "pass": True,
    }

    if large_gaps.any():
        logger.info(
            "Timestamp gaps detected (informational) — %d gaps > %.1f min. "
            "Largest: %.1f min. Handled at feature engineering stage.",
            len(large_gaps),
            threshold,
            gaps_minutes.max(),
        )
    else:
        logger.info(
            "Timestamp continuity: no gaps > %.1f min",
            threshold,
        )

    return result


def _validate_continuity_per_series(
    df: pd.DataFrame,
    timestamp_col: str,
    max_gap_multiplier: float,
    group_col: str,
    min_threshold_minutes: float = 5.0,
) -> dict:
    """
    Check timestamp continuity within each group independently.

    Used for concatenated multi-series datasets (NAB, Alibaba) where
    inter-series gaps are expected and should not cause a failure.

    Args:
        min_threshold_minutes: Floor on the gap threshold. Prevents the
            threshold from becoming near-zero when readings from multiple
            machines with the same timestamp are concatenated (Alibaba).
            Default 5 minutes: a genuine gap under 5 minutes is not
            operationally meaningful for anomaly detection.
    """
    series_results = {}
    total_large_gaps = 0

    for series_name, group_df in df.groupby(group_col):
        ts = (
            pd.to_datetime(group_df[timestamp_col]).sort_values().reset_index(drop=True)
        )
        gaps_minutes = ts.diff().dt.total_seconds().div(60).dropna()

        if gaps_minutes.empty:
            continue

        median_gap = gaps_minutes.median()
        # Apply minimum floor to prevent threshold = 0 when readings from
        # multiple machines with identical timestamps are concatenated
        threshold = max(median_gap * max_gap_multiplier, min_threshold_minutes)
        large = gaps_minutes[gaps_minutes > threshold]
        total_large_gaps += len(large)

        if len(large) > 0:
            logger.info(
                "Series %s: %d gaps > %.1f min. Max: %.1f min "
                "(informational — not a blocking failure)",
                series_name,
                len(large),
                threshold,
                gaps_minutes.max(),
            )

        series_results[str(series_name)] = {
            "large_gap_count": len(large),
            "max_gap_minutes": round(float(gaps_minutes.max()), 2),
            "median_interval_minutes": round(float(median_gap), 2),
            "threshold_minutes": round(float(threshold), 2),
        }

    n_with_gaps = sum(1 for v in series_results.values() if v["large_gap_count"] > 0)
    logger.info(
        "Timestamp continuity (per %s, informational): %d/%d series have "
        "gaps > threshold. Gaps handled at feature engineering via sequence "
        "windowing — not a blocking data quality failure.",
        group_col,
        n_with_gaps,
        len(series_results),
    )

    return {
        "mode": "per_series",
        "group_col": group_col,
        "series_checked": len(series_results),
        "series_with_gaps": n_with_gaps,
        "total_large_gaps": total_large_gaps,
        "min_threshold_minutes": min_threshold_minutes,
        # pass is always True here — continuity is informational only.
        # overall_pass in generate_data_quality_report does not include this.
        "pass": True,
        "series_detail": series_results,
    }


def validate_value_ranges(df: pd.DataFrame) -> dict:
    """
    Check that metric columns stay within valid physical bounds.

    CPU and memory must be 0-100%. Network and disk I/O must be >= 0.

    Args:
        df: DataFrame with metric columns.

    Returns:
        Dict mapping column → {min, max, negative_count, exceeds_100_count, pass}.
    """
    results = {}

    pct_cols = [c for c in ["cpu_util", "mem_util", "disk_io"] if c in df.columns]
    for col in pct_cols:
        col_data = pd.to_numeric(df[col], errors="coerce").dropna()
        neg_count = int((col_data < 0).sum())
        over_count = int((col_data > 100).sum())
        results[col] = {
            "min": round(float(col_data.min()), 3),
            "max": round(float(col_data.max()), 3),
            "negative_count": neg_count,
            "exceeds_100_count": over_count,
            "pass": neg_count == 0 and over_count == 0,
        }

    non_neg_cols = [c for c in NEGATIVE_VALUE_COLS if c in df.columns]
    for col in non_neg_cols:
        if col in results:
            continue
        col_data = pd.to_numeric(df[col], errors="coerce").dropna()
        neg_count = int((col_data < 0).sum())
        results[col] = {
            "min": round(float(col_data.min()), 3),
            "max": round(float(col_data.max()), 3),
            "negative_count": neg_count,
            "pass": neg_count == 0,
        }

    return results


# ---------------------------------------------------------------------------
# Composite quality report
# ---------------------------------------------------------------------------


def generate_data_quality_report(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    timestamp_col: str = "timestamp",
) -> dict:
    """
    Generate a comprehensive data quality report for one DataFrame.

    Combines null rate checks, timestamp continuity, value ranges,
    and anomaly distribution into a single nested dict suitable for
    logging, saving to JSON, or displaying in the profiling notebook.

    Args:
        df:            DataFrame to report on.
        dataset_name:  Label used in the report header.
        timestamp_col: Name of the timestamp column.

    Returns:
        Nested dict. Key 'overall_pass' is True when all checks pass.
    """
    logger.info("Generating data quality report for: %s", dataset_name)

    # Use per-series continuity check for concatenated multi-series datasets.
    # NAB:     group by source_file  (each CSV is one independent time-series)
    # Alibaba: group by machine_id   (each machine is one independent time-series)
    # Checking continuity across series produces false failures from expected
    # inter-series gaps when different machines are concatenated together.
    if "source_file" in df.columns:
        continuity_group = "source_file"
    elif "machine_id" in df.columns:
        continuity_group = "machine_id"
    else:
        continuity_group = None

    report: dict = {
        "dataset": dataset_name,
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "null_rate_check": validate_null_rates(df),
        "timestamp_continuity": validate_timestamp_continuity(
            df, timestamp_col, group_col=continuity_group
        ),
        "value_range_check": validate_value_ranges(df),
    }

    if "is_anomaly" in df.columns:
        report["anomaly_distribution"] = {
            "anomaly_count": int(df["is_anomaly"].sum()),
            "normal_count": int((~df["is_anomaly"]).sum()),
            "anomaly_rate_pct": round(float(df["is_anomaly"].mean()) * 100, 3),
            "class_imbalance_ratio": round(
                float((~df["is_anomaly"]).sum())
                / max(float(df["is_anomaly"].sum()), 1),
                1,
            ),
        }

    # Aggregate overall pass/fail
    null_checks = [
        v["pass"]
        for v in report["null_rate_check"].values()
        if isinstance(v, dict) and "pass" in v
    ]
    range_checks = [
        v["pass"]
        for v in report["value_range_check"].values()
        if isinstance(v, dict) and "pass" in v
    ]

    # Timestamp continuity is intentionally excluded from overall_pass.
    # Real-world training datasets contain genuine measurement gaps that
    # are properties of the collection system, not data quality failures.
    # NAB contains heterogeneous series at different sampling rates;
    # Alibaba has irregular collection intervals per machine.
    # Gaps are handled at Day 3 feature engineering by the sequence
    # windowing logic — sequences that span a gap are simply not created.
    # overall_pass gates on null rates and value ranges only.
    report["overall_pass"] = all(null_checks) and all(range_checks)
    report["timestamp_continuity_informational"] = True

    status = "PASS" if report["overall_pass"] else "FAIL"
    logger.info("Data quality report for %s: %s", dataset_name, status)

    return report


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def define_temporal_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series DataFrame into train / validation / test sets.

    CRITICAL: Data is split by time, not randomly shuffled.
    Earlier rows go to training; later rows go to evaluation.
    Shuffling would cause data leakage — the model would train on
    future data and evaluate on the past.

    Args:
        df:            DataFrame to split.
        timestamp_col: Column to sort by before splitting.
        train_pct:     Fraction for training (default 70%).
        val_pct:       Fraction for validation (default 15%).
                       Test receives the remaining 15%.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        AssertionError: If train_pct + val_pct >= 1.0.
    """
    test_pct = round(1.0 - train_pct - val_pct, 4)
    assert test_pct > 0, f"train_pct ({train_pct}) + val_pct ({val_pct}) must be < 1.0"

    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)

    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        "Temporal split — train: %s (%.0f%%) | val: %s (%.0f%%) | test: %s (%.0f%%)",
        f"{len(train_df):,}",
        train_pct * 100,
        f"{len(val_df):,}",
        val_pct * 100,
        f"{len(test_df):,}",
        test_pct * 100,
    )
    logger.info(
        "Train range: %s → %s",
        train_df[timestamp_col].min(),
        train_df[timestamp_col].max(),
    )
    logger.info(
        "Val range:   %s → %s",
        val_df[timestamp_col].min(),
        val_df[timestamp_col].max(),
    )
    logger.info(
        "Test range:  %s → %s",
        test_df[timestamp_col].min(),
        test_df[timestamp_col].max(),
    )

    # Verify temporal ordering — no leakage
    assert train_df[timestamp_col].max() <= val_df[timestamp_col].min(), (
        "DATA LEAKAGE: train set contains timestamps after val set start"
    )
    assert val_df[timestamp_col].max() <= test_df[timestamp_col].min(), (
        "DATA LEAKAGE: val set contains timestamps after test set start"
    )
    logger.info("Temporal ordering verified — no data leakage")

    return train_df, val_df, test_df

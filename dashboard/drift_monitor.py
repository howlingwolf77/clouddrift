"""
Evidently AI drift monitoring for CloudDrift.

Compares the training reference distribution against a rolling window
of recent inference inputs collected during a Streamlit dashboard session.

Reference data:
    Loaded from artifacts/api_reference_stats.json (per-metric mean/std)
    or from data/processed/alibaba_machine_usage.parquet if available.
    A sample of 500 normal rows from the Alibaba data provides richer
    statistical power than synthetic data from the mean/std stats alone.

Current data:
    The last N telemetry snapshots submitted via the Streamlit dashboard,
    passed in as a list of dicts by the caller (dashboard/app.py).

Drift detection:
    Uses Evidently's DataDriftPreset which applies the Wasserstein distance
    for numerical features by default. A drift_score per column is returned
    alongside the HTML report for inline display in the dashboard.

Output:
    HTML report saved to logs/drift/ with a timestamp filename.
    Summary dict returned for inline display in Streamlit.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
PROCESSED_DIR = Path("data/processed")
DRIFT_LOG_DIR = Path("logs/drift")

METRIC_COLS = ["cpu_util", "mem_util", "net_io_in", "net_io_out", "disk_io"]

# Minimum rows needed in the current window to generate a meaningful report
MIN_CURRENT_ROWS = 30


# ---------------------------------------------------------------------------
# Reference data loading
# ---------------------------------------------------------------------------


def load_reference_data(n_rows: int = 500) -> pd.DataFrame:
    """
    Load the training reference distribution for drift comparison.

    Tries two sources in order:
        1. Alibaba processed parquet (real production telemetry)
        2. Synthetic data generated from api_reference_stats.json
           (fallback when Alibaba data is unavailable)

    Args:
        n_rows: Maximum rows to sample from Alibaba data.

    Returns:
        DataFrame with columns: cpu_util, mem_util, net_io_in,
        net_io_out, disk_io (NaN allowed for disk_io).
    """
    alibaba_path = PROCESSED_DIR / "alibaba_machine_usage.parquet"

    if alibaba_path.exists():
        df = pd.read_parquet(alibaba_path)
        available_cols = [c for c in METRIC_COLS if c in df.columns]
        df = df[available_cols].dropna(subset=["cpu_util", "mem_util"]).copy()
        if len(df) > n_rows:
            df = df.sample(n=n_rows, random_state=42)
        logger.info(
            "Reference data loaded from Alibaba parquet: %d rows, %d cols",
            len(df),
            len(available_cols),
        )
        return df.reset_index(drop=True)

    # Fallback: generate from reference stats
    ref_stats_path = ARTIFACTS_DIR / "api_reference_stats.json"
    if not ref_stats_path.exists():
        raise FileNotFoundError(
            "Neither Alibaba parquet nor api_reference_stats.json found. "
            "Run Day 8 Step 2 (build API reference stats) first."
        )

    with open(ref_stats_path) as f:
        stats = json.load(f)

    rng = np.random.default_rng(42)
    rows: dict[str, np.ndarray] = {}
    for col in METRIC_COLS:
        if col in stats:
            mean = stats[col]["mean"]
            std = stats[col]["std"]
            rows[col] = rng.normal(mean, std, n_rows).clip(
                stats[col].get("p1", 0),
                stats[col].get("p99", 100),
            )
        else:
            rows[col] = np.full(n_rows, np.nan)

    df = pd.DataFrame(rows)
    logger.info(
        "Reference data generated from api_reference_stats.json: %d rows", n_rows
    )
    return df


# ---------------------------------------------------------------------------
# Drift report generation
# ---------------------------------------------------------------------------


def generate_drift_report(
    current_data: list[dict],
    reference_df=None,
) -> tuple[str, dict]:
    """
    Generate an Evidently data drift report comparing training reference
    against the current rolling window of inference inputs.

    Evidently 0.7.x API: Report.run() returns a Snapshot object.
    All export methods (save_html, get_html_str, dict) are on the
    Snapshot, not on the Report.

    Args:
        current_data: List of snapshot dicts from the Streamlit session.
        reference_df: Pre-loaded reference DataFrame. If None, loads
                      via load_reference_data().

    Returns:
        Tuple of (html_report_path, summary_dict).

    Raises:
        ValueError: If current_data has fewer than MIN_CURRENT_ROWS rows.
    """
    if len(current_data) < MIN_CURRENT_ROWS:
        raise ValueError(
            f"Need at least {MIN_CURRENT_ROWS} current readings for a "
            f"meaningful drift report — only {len(current_data)} provided. "
            f"Submit more telemetry readings and try again."
        )

    if reference_df is None:
        reference_df = load_reference_data()

    current_df = pd.DataFrame(current_data)
    available_cols = [c for c in METRIC_COLS if c in current_df.columns]
    current_df = current_df[available_cols].copy()

    shared_cols = [
        c for c in METRIC_COLS if c in reference_df.columns and c in current_df.columns
    ]
    ref_aligned = reference_df[shared_cols].copy()
    curr_aligned = current_df[shared_cols].copy()

    logger.info(
        "Running Evidently drift report: ref=%d rows, current=%d rows, cols=%s",
        len(ref_aligned),
        len(curr_aligned),
        shared_cols,
    )

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        try:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
        except ImportError as e:
            raise ImportError(
                "evidently is required for drift reports. Run: uv add evidently"
            ) from e

    # Evidently 0.7+: run() returns a Snapshot — that is the object
    # with export methods. Report itself has none.
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_aligned, current_data=curr_aligned)

    DRIFT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    html_path = DRIFT_LOG_DIR / f"drift_report_{timestamp}.html"

    # Use save_html if available (Snapshot.save_html writes to disk directly)
    # Fall back to get_html_str which returns the HTML as a string
    if hasattr(snapshot, "save_html"):
        snapshot.save_html(str(html_path))
    elif hasattr(snapshot, "get_html_str"):
        html_path.write_text(snapshot.get_html_str(), encoding="utf-8")
    else:
        raise RuntimeError(
            f"Cannot export Evidently report — no save_html or get_html_str "
            f"on {type(snapshot)}. Available: {[m for m in dir(snapshot) if not m.startswith('_')]}"
        )

    logger.info("Drift report saved: %s", html_path)

    # Extract per-column drift summary from the snapshot
    summary = _extract_drift_summary(snapshot, shared_cols)
    return str(html_path), summary


def _extract_drift_summary(snapshot, cols: list[str]) -> dict:
    """
    Extract per-column drift detection results from an Evidently Snapshot.

    Evidently 0.7+: the Snapshot object has a .dict() method.
    Falls back gracefully if the structure is unexpected.
    """
    summary: dict = {"columns": {}, "n_drifted": 0, "dataset_drifted": False}

    try:
        if hasattr(snapshot, "dict"):
            report_dict = snapshot.dict()
        elif hasattr(snapshot, "as_dict"):
            report_dict = snapshot.as_dict()
        else:
            logger.warning(
                "Snapshot has no dict/as_dict method — returning empty summary"
            )
            return summary

        metrics = report_dict.get("metrics", [])
        for metric in metrics:
            result = metric.get("result", {})
            drift_by_col = result.get("drift_by_columns", {})
            for col, col_result in drift_by_col.items():
                summary["columns"][col] = {
                    "drifted": bool(col_result.get("drift_detected", False)),
                    "drift_score": float(col_result.get("drift_score", 0.0)),
                    "stattest": col_result.get("stattest_name", "unknown"),
                }
            if "dataset_drift" in result:
                summary["dataset_drifted"] = bool(result["dataset_drift"])
            if "number_of_drifted_columns" in result:
                summary["n_drifted"] = int(result["number_of_drifted_columns"])
    except Exception:
        logger.exception("Could not parse Evidently snapshot summary")

    return summary


# ---------------------------------------------------------------------------
# Lightweight per-metric z-score drift (no Evidently required)
# ---------------------------------------------------------------------------


def compute_zscore_drift(
    current_data: list[dict],
    reference_stats_path: Path = ARTIFACTS_DIR / "api_reference_stats.json",
) -> dict[str, float]:
    """
    Compute simple z-score drift for each metric.

    Compares the mean of recent current readings against the training
    distribution using a z-score: |current_mean - ref_mean| / ref_std.
    Does not require Evidently — useful for inline display even when
    there are fewer than MIN_CURRENT_ROWS readings.

    Args:
        current_data: List of snapshot dicts from the Streamlit session.
        reference_stats_path: Path to api_reference_stats.json.

    Returns:
        Dict mapping metric name → absolute z-score (higher = more drift).
    """
    if not current_data or not reference_stats_path.exists():
        return {}

    with open(reference_stats_path) as f:
        ref_stats = json.load(f)

    current_df = pd.DataFrame(current_data)
    drift_scores: dict[str, float] = {}

    for col in METRIC_COLS:
        if col not in current_df.columns or col not in ref_stats:
            continue
        col_data = current_df[col].dropna()
        if len(col_data) == 0:
            continue
        current_mean = float(col_data.mean())
        ref_mean = ref_stats[col]["mean"]
        ref_std = ref_stats[col]["std"]
        drift_scores[col] = round(abs(current_mean - ref_mean) / ref_std, 3)

    return drift_scores

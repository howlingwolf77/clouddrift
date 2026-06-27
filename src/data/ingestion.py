"""
Data ingestion pipeline for CloudDrift.

Loads and unifies:
  - Numenta Anomaly Benchmark (NAB): real AWS CloudWatch time-series
    with verified anomaly window labels
  - Alibaba Cluster Trace 2018: real production cluster telemetry
    (machine_usage.csv, 8.4 GB, no header row)

Both datasets contain real production data, not simulated data.

Alibaba Cluster Trace schema (no header — columns named on read):
    machine_id       string   uid of machine
    time_stamp       double   seconds from trace reference epoch
    cpu_util_percent bigint   [0, 100]
    mem_util_percent bigint   [0, 100]
    mem_gps          double   normalized memory bandwidth [0, 100] — sparse
    mkpi             bigint   cache miss per thousand instructions — sparse
    net_in           double   normalized inbound network traffic [0, 100]
    net_out          double   normalized outbound network traffic [0, 100]
    disk_io_percent  double   [0, 100]; sentinel values -1 and 101 are invalid

Reference: Guo et al., "Limitations and Improvements of Machine Learning Workload
Scheduling in Cloud Computing", 2019.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

NAB_ROOT = Path("data/raw/nab")
ALIBABA_ROOT = Path("data/raw/alibaba")
PROCESSED_DIR = Path("data/processed")

# NAB categories used for CloudDrift
NAB_CATEGORIES = ["realAWSCloudwatch", "realKnownCause"]

# ---------------------------------------------------------------------------
# Alibaba Cluster Trace constants
# ---------------------------------------------------------------------------

# Column names for machine_usage.csv — file has NO header row
ALIBABA_COLUMNS = [
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "mkpi",
    "net_in",
    "net_out",
    "disk_io_percent",
]

# Sentinel values in disk_io_percent that indicate invalid readings
# Per Alibaba schema documentation: -1 and 101 are abnormal values
ALIBABA_DISK_IO_SENTINELS = {-1, 101}

# Reference epoch for converting Alibaba time_stamp (seconds) to datetime
# The 2018 trace starts approximately at this date
ALIBABA_BASE_TIME = pd.Timestamp("2018-01-01 00:00:00")

# CloudDrift standard column names mapped from Alibaba names
ALIBABA_RENAME_MAP = {
    "cpu_util_percent": "cpu_util",
    "mem_util_percent": "mem_util",
    "net_in": "net_io_in",
    "net_out": "net_io_out",
    "disk_io_percent": "disk_io",
}


# ---------------------------------------------------------------------------
# Public interface — NAB
# ---------------------------------------------------------------------------


def load_nab_dataset(nab_root: str | Path = NAB_ROOT) -> pd.DataFrame:
    """
    Load Numenta Anomaly Benchmark time-series with anomaly labels.

    Reads all CSV files from NAB_CATEGORIES, attaches anomaly labels
    from combined_windows.json, and returns a unified DataFrame.

    Args:
        nab_root: Path to the cloned NAB repository root.

    Returns:
        DataFrame with columns:
            timestamp    — datetime64[ns]
            value        — float64 (metric reading)
            metric_name  — str (filename stem)
            category     — str (NAB category)
            source_file  — str (category/filename.csv)
            is_anomaly   — bool

    Raises:
        FileNotFoundError: If nab_root or labels file does not exist.
        ValueError: If no data files load successfully.
    """
    nab_root = Path(nab_root)
    data_path = nab_root / "data"
    labels_path = nab_root / "labels" / "combined_windows.json"

    if not data_path.exists():
        raise FileNotFoundError(f"NAB data directory not found: {data_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"NAB labels file not found: {labels_path}")

    with open(labels_path) as f:
        anomaly_windows = json.load(f)

    frames = []
    for category in NAB_CATEGORIES:
        category_path = data_path / category
        if not category_path.exists():
            logger.warning("NAB category not found, skipping: %s", category)
            continue

        csv_files = sorted(category_path.glob("*.csv"))
        logger.info("Loading %d files from %s", len(csv_files), category)

        for csv_file in csv_files:
            df = _load_nab_csv(csv_file, category, anomaly_windows)
            if df is not None:
                frames.append(df)

    if not frames:
        raise ValueError("No NAB data files loaded successfully.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    n_anomaly = combined["is_anomaly"].sum()
    logger.info(
        "NAB loaded: %s rows | %s anomaly rows (%.1f%%)",
        f"{len(combined):,}",
        f"{n_anomaly:,}",
        combined["is_anomaly"].mean() * 100,
    )
    return combined


# ---------------------------------------------------------------------------
# Public interface — Alibaba
# ---------------------------------------------------------------------------


def load_alibaba_cluster_trace(
    alibaba_root: str | Path = ALIBABA_ROOT,
    max_machines: int = 5,
    chunk_size: int = 500_000,
    max_rows: int = 5_000_000,
) -> pd.DataFrame | None:
    """
    Load Alibaba Cluster Trace 2018 machine usage data.

    The file (machine_usage.csv) is 8.4 GB with no header row.
    This function uses chunked reading to avoid loading the full
    file into memory, stopping after max_rows rows have been
    collected for the selected machines.

    Key data characteristics handled here:
      - No header row: columns named via ALIBABA_COLUMNS
      - time_stamp is seconds from reference epoch, not Unix time
      - disk_io_percent uses -1 and 101 as sentinel values for
        invalid readings — replaced with NaN
      - mem_gps and mkpi are genuinely sparse (many nulls expected)

    Args:
        alibaba_root:  Directory containing machine_usage.csv.
        max_machines:  Number of unique machine IDs to retain.
                       Reduces output to a manageable subset.
        chunk_size:    Rows per read chunk (default 500,000).
        max_rows:      Stop reading after this many rows have been
                       collected for the target machines.

    Returns:
        DataFrame with standardised CloudDrift column names,
        or None if the data file is not found.
    """
    alibaba_root = Path(alibaba_root)
    csv_path = alibaba_root / "machine_usage.csv"

    if not csv_path.exists():
        logger.warning("Alibaba machine_usage.csv not found at %s", csv_path)
        return None

    file_size_gb = csv_path.stat().st_size / 1_073_741_824
    logger.info(
        "Reading Alibaba machine_usage.csv (%.1f GB) in chunks of %s rows",
        file_size_gb,
        f"{chunk_size:,}",
    )

    target_machines: set | None = None
    frames: list[pd.DataFrame] = []
    total_rows_collected = 0

    for chunk in pd.read_csv(
        csv_path,
        header=None,  # no header row in the file
        names=ALIBABA_COLUMNS,  # assign column names explicitly
        dtype={
            "machine_id": str,
            "time_stamp": float,
            "cpu_util_percent": float,
            "mem_util_percent": float,
            "mem_gps": float,
            "mkpi": float,
            "net_in": float,
            "net_out": float,
            "disk_io_percent": float,
        },
        chunksize=chunk_size,
        low_memory=False,
    ):
        # On the first chunk, decide which machines to track
        if target_machines is None:
            all_machines = chunk["machine_id"].dropna().unique()
            target_machines = set(all_machines[:max_machines])
            logger.info(
                "Targeting %d machines: %s",
                len(target_machines),
                sorted(target_machines),
            )

        filtered = chunk[chunk["machine_id"].isin(target_machines)].copy()
        if len(filtered) == 0:
            continue

        frames.append(filtered)
        total_rows_collected += len(filtered)

        logger.debug(
            "Chunk processed — %s rows collected so far",
            f"{total_rows_collected:,}",
        )

        if total_rows_collected >= max_rows:
            logger.info(
                "Reached max_rows limit (%s) — stopping early",
                f"{max_rows:,}",
            )
            break

    if not frames:
        logger.warning("No rows collected from Alibaba dataset")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = _normalise_alibaba_columns(combined)

    if combined is None:
        return None

    logger.info(
        "Alibaba loaded: %s rows | %d machines",
        f"{len(combined):,}",
        combined["machine_id"].nunique() if "machine_id" in combined.columns else 0,
    )
    return combined


# ---------------------------------------------------------------------------
# Public utility
# ---------------------------------------------------------------------------


def get_dataset_summary(df: pd.DataFrame, name: str = "dataset") -> dict:
    """
    Return a concise profiling summary of a loaded DataFrame.

    Args:
        df:   Any loaded CloudDrift DataFrame.
        name: Label for the summary header.

    Returns:
        Nested dict covering shape, dtypes, null rates, time range,
        and anomaly distribution if the column is present.
    """
    summary: dict = {
        "name": name,
        "rows": len(df),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "null_rates_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    summary["numeric_stats"] = (
        df[numeric_cols].describe().round(4).to_dict() if numeric_cols else {}
    )

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        summary["time_range"] = {
            "start": str(ts.min()),
            "end": str(ts.max()),
            "duration_days": (ts.max() - ts.min()).days,
            "total_readings": len(ts),
        }

    if "is_anomaly" in df.columns:
        summary["anomaly_distribution"] = {
            "anomaly_count": int(df["is_anomaly"].sum()),
            "normal_count": int((~df["is_anomaly"]).sum()),
            "anomaly_rate_pct": round(df["is_anomaly"].mean() * 100, 3),
        }

    if "category" in df.columns:
        summary["category_counts"] = df["category"].value_counts().to_dict()

    if "metric_name" in df.columns:
        summary["unique_metrics"] = df["metric_name"].nunique()

    return summary


# ---------------------------------------------------------------------------
# Private helpers — NAB
# ---------------------------------------------------------------------------


def _load_nab_csv(
    csv_file: Path,
    category: str,
    anomaly_windows: dict,
) -> pd.DataFrame | None:
    """Load one NAB CSV file and attach anomaly labels."""
    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()

        if "timestamp" not in df.columns or "value" not in df.columns:
            logger.warning(
                "Unexpected columns in %s: %s",
                csv_file.name,
                df.columns.tolist(),
            )
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["metric_name"] = csv_file.stem
        df["category"] = category
        df["source_file"] = f"{category}/{csv_file.name}"

        label_key = f"{category}/{csv_file.name}"
        windows = anomaly_windows.get(label_key, [])
        df["is_anomaly"] = _label_anomalies(df["timestamp"], windows)

        return df

    except Exception:
        logger.exception("Failed to load %s", csv_file)
        return None


def _label_anomalies(timestamps: pd.Series, windows: list) -> pd.Series:
    """
    Mark each timestamp True if it falls within any anomaly window.

    Args:
        timestamps: Series of datetime64 timestamps.
        windows:    List of [start_str, end_str] pairs from combined_windows.json.

    Returns:
        Boolean Series aligned with timestamps.index.
    """
    labels = pd.Series(False, index=timestamps.index)
    for window in windows:
        start = pd.to_datetime(window[0])
        end = pd.to_datetime(window[1])
        mask = (timestamps >= start) & (timestamps <= end)
        labels = labels | mask
    return labels


# ---------------------------------------------------------------------------
# Private helpers — Alibaba
# ---------------------------------------------------------------------------


def _normalise_alibaba_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Clean and standardise an Alibaba machine_usage DataFrame.

    Changes applied:
      1. Rename columns to CloudDrift standard names.
      2. Convert time_stamp (seconds from reference epoch) to datetime.
      3. Replace disk_io_percent sentinel values (-1, 101) with NaN.
      4. Clip cpu_util and mem_util to [0, 100] after coercion.
      5. Drop mkpi and mem_gps (sparse, not used in feature engineering).

    Args:
        df: Raw chunk or combined DataFrame with ALIBABA_COLUMNS names.

    Returns:
        Cleaned DataFrame, or None if required columns are absent.
    """
    required_raw = ["cpu_util_percent", "mem_util_percent"]
    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        logger.warning("Alibaba DataFrame missing required columns: %s", missing)
        return None

    # 1 — Convert time_stamp (seconds) to datetime
    if "time_stamp" in df.columns:
        df["timestamp"] = ALIBABA_BASE_TIME + pd.to_timedelta(
            pd.to_numeric(df["time_stamp"], errors="coerce"),
            unit="s",
        )
    else:
        logger.warning("time_stamp column not found — timestamp will be absent")

    # 2 — Replace disk_io_percent sentinel values before rename
    if "disk_io_percent" in df.columns:
        df["disk_io_percent"] = pd.to_numeric(df["disk_io_percent"], errors="coerce")
        sentinel_mask = df["disk_io_percent"].isin(ALIBABA_DISK_IO_SENTINELS)
        if sentinel_mask.any():
            n_sentinels = sentinel_mask.sum()
            logger.info(
                "Replaced %s disk_io_percent sentinel values (-1 or 101) with NaN",
                f"{n_sentinels:,}",
            )
            df.loc[sentinel_mask, "disk_io_percent"] = float("nan")

    # 3 — Rename to CloudDrift standard names
    df = df.rename(columns=ALIBABA_RENAME_MAP)

    # 4 — Clip percentage columns to valid range [0, 100]
    for col in ["cpu_util", "mem_util", "net_io_in", "net_io_out", "disk_io"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 100)

    # 5 — Drop sparse columns not used in CloudDrift feature engineering
    df = df.drop(columns=["mem_gps", "mkpi", "time_stamp"], errors="ignore")

    return df

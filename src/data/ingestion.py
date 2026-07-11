"""
Data ingestion pipeline for CloudDrift.

Loads and unifies:
  - Server Machine Dataset (SMD): 28 server machines with labeled anomalies,
    38-dimensional telemetry (CPU, memory, network, disk I/O), ~4% anomaly rate.
    Primary training dataset for the CloudDrift anomaly detection pipeline.
  - Alibaba Cluster Trace 2018: real production cluster telemetry
    (machine_usage.csv, 8.4 GB, no header row). Supplementary dataset
    for validation and exploration.

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

SMD schema (38 unnamed columns, values pre-normalized to [0, 1]):
    CloudDrift selects 5 columns by index and renames to standard names.
    See SMD_COL_MAP for the index-to-name mapping.

Reference: Guo et al., "Limitations and Improvements of Machine Learning
Workload Scheduling in Cloud Computing", 2019 (Alibaba).
Su et al., "Robust Anomaly Detection for Multivariate Time Series through
Stochastic Recurrent Neural Network", 2019 (SMD / OmniAnomaly).
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ALIBABA_ROOT = Path("data/raw/alibaba")
PROCESSED_DIR = Path("data/processed")

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
# SMD constants
# ---------------------------------------------------------------------------

SMD_ROOT = Path("data/raw/smd")
SMD_DATA_DIR = "ServerMachineDataset"
SMD_BASE_TIME = pd.Timestamp("2024-01-01 00:00:00")

# SMD column indices → CloudDrift standard names (0-indexed, 38 cols total).
# Mapping based on OmniAnomaly paper supplementary and common SMD releases.
# Only these five are selected; the remaining 33 are dropped.
SMD_COL_MAP: dict[int, str] = {
    0: "cpu_util",
    1: "net_io_in",
    2: "net_io_out",
    3: "disk_io",
    5: "mem_util",
}


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
        header=None,
        names=ALIBABA_COLUMNS,
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
# Public interface — SMD
# ---------------------------------------------------------------------------


def load_smd_dataset(
    smd_root: str | Path = SMD_ROOT,
    machines: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load Server Machine Dataset (SMD) with anomaly labels.

    SMD contains telemetry from 28 server machines split into pre-defined
    train and test periods of approximately equal length (~28 days each).
    Train data is guaranteed anomaly-free by the dataset authors; test data
    contains labeled anomalies at roughly 4% overall rate.

    Each machine is loaded as [train rows, test rows] concatenated in
    chronological order. Synthetic timestamps at 1-minute intervals are
    assigned starting from SMD_BASE_TIME, since the dataset contains no
    real timestamps.

    Five columns are selected from the 38-dimensional raw files and renamed
    to CloudDrift standard names (cpu_util, mem_util, net_io_in, net_io_out,
    disk_io), enabling direct reuse of build_alibaba_features() without any
    changes to feature engineering.

    The source_file column is set equal to machine_id. This is required for
    compatibility with SequenceDataset and compute_reconstruction_errors()
    in tcn_autoencoder.py, which group by source_file by default.

    Args:
        smd_root: Path to the directory containing ServerMachineDataset/.
                  Default: data/raw/smd/
        machines: Specific machine names to load, e.g. ["machine-1-1"].
                  Loads all 28 machines if None.

    Returns:
        DataFrame with columns:
            machine_id  — str  (e.g. "machine-1-1")
            source_file — str  (alias of machine_id; required for TCN grouping)
            timestamp   — datetime64[ns] (synthetic, 1-minute intervals)
            cpu_util    — float [0, 1]  (SMD column 0)
            net_io_in   — float [0, 1]  (SMD column 1)
            net_io_out  — float [0, 1]  (SMD column 2)
            disk_io     — float [0, 1]  (SMD column 3)
            mem_util    — float [0, 1]  (SMD column 5)
            is_anomaly  — bool

    Raises:
        FileNotFoundError: If smd_root or any required subdirectory is absent.
        ValueError: If no machine files load successfully.
    """
    smd_root = Path(smd_root)
    data_root = smd_root / SMD_DATA_DIR
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    label_dir = data_root / "test_label"

    for d in (train_dir, test_dir, label_dir):
        if not d.exists():
            raise FileNotFoundError(f"SMD directory not found: {d}")

    all_machines = sorted(p.stem for p in train_dir.glob("machine-*.txt"))
    if not all_machines:
        raise FileNotFoundError(f"No machine-*.txt files found in {train_dir}")

    target = machines if machines is not None else all_machines
    logger.info("Loading SMD: %d of %d machines", len(target), len(all_machines))

    frames = []
    for machine_name in target:
        df = _load_smd_machine(machine_name, train_dir, test_dir, label_dir)
        if df is not None:
            frames.append(df)

    if not frames:
        raise ValueError("No SMD machine files loaded successfully.")

    combined = pd.concat(frames, ignore_index=True)

    n_anomaly = combined["is_anomaly"].sum()
    logger.info(
        "SMD loaded: %s rows | %d machines | %s anomaly rows (%.1f%%)",
        f"{len(combined):,}",
        combined["machine_id"].nunique(),
        f"{n_anomaly:,}",
        combined["is_anomaly"].mean() * 100,
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

    if "time_stamp" in df.columns:
        df["timestamp"] = ALIBABA_BASE_TIME + pd.to_timedelta(
            pd.to_numeric(df["time_stamp"], errors="coerce"),
            unit="s",
        )
    else:
        logger.warning("time_stamp column not found — timestamp will be absent")

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

    df = df.rename(columns=ALIBABA_RENAME_MAP)

    for col in ["cpu_util", "mem_util", "net_io_in", "net_io_out", "disk_io"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 100)

    df = df.drop(columns=["mem_gps", "mkpi", "time_stamp"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Private helpers — SMD
# ---------------------------------------------------------------------------


def _load_smd_machine(
    machine_name: str,
    train_dir: Path,
    test_dir: Path,
    label_dir: Path,
) -> pd.DataFrame | None:
    """
    Load one SMD machine: train rows (normal) followed by test rows (labeled).

    Train rows receive is_anomaly=False — the SMD dataset guarantees that
    train files contain only normal operating periods. Test rows receive
    binary labels from the corresponding test_label file.

    Synthetic timestamps are assigned as sequential 1-minute offsets from
    SMD_BASE_TIME, with row 0 of train receiving timestamp 0 and the first
    row of test continuing immediately after the last train row.

    Args:
        machine_name: Stem of the machine file, e.g. "machine-1-1".
        train_dir:    Path to ServerMachineDataset/train/.
        test_dir:     Path to ServerMachineDataset/test/.
        label_dir:    Path to ServerMachineDataset/test_label/.

    Returns:
        DataFrame for this machine, or None on load failure.
    """
    train_file = train_dir / f"{machine_name}.txt"
    test_file = test_dir / f"{machine_name}.txt"
    label_file = label_dir / f"{machine_name}.txt"

    missing = [str(f) for f in (train_file, test_file, label_file) if not f.exists()]
    if missing:
        logger.warning("Skipping %s — files not found: %s", machine_name, missing)
        return None

    try:
        train_raw = pd.read_csv(train_file, header=None)
        test_raw = pd.read_csv(test_file, header=None)
        labels_raw = pd.read_csv(label_file, header=None, names=["is_anomaly"])

        if len(test_raw) != len(labels_raw):
            logger.warning(
                "%s: test rows (%d) != label rows (%d) — skipping",
                machine_name,
                len(test_raw),
                len(labels_raw),
            )
            return None

        train_df = _select_smd_columns(train_raw)
        test_df = _select_smd_columns(test_raw)

        train_df["is_anomaly"] = False
        test_df["is_anomaly"] = labels_raw["is_anomaly"].astype(bool).values

        combined = pd.concat([train_df, test_df], ignore_index=True)

        combined["timestamp"] = SMD_BASE_TIME + pd.to_timedelta(
            range(len(combined)), unit="min"
        )
        combined["machine_id"] = machine_name
        combined["source_file"] = machine_name  # TCN group_col compatibility

        n_anom = int(test_df["is_anomaly"].sum())
        logger.debug(
            "%s | train=%s test=%s anomalies=%d (%.1f%%)",
            machine_name,
            f"{len(train_df):,}",
            f"{len(test_df):,}",
            n_anom,
            test_df["is_anomaly"].mean() * 100 if len(test_df) > 0 else 0.0,
        )
        return combined

    except Exception:
        logger.exception("Failed to load machine %s", machine_name)
        return None


def _select_smd_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select 5 SMD columns by index and rename to CloudDrift standard names.

    SMD raw files contain 38 unnamed float columns. Five are selected based
    on their known semantic mapping (cpu_util, net_io_in, net_io_out,
    disk_io, mem_util) to match the column names expected by
    build_alibaba_features() in features/engineering.py.

    SMD values are published pre-normalized to approximately [0, 1].
    Clipping enforces exact [0, 1] bounds to prevent Pandera schema
    failures from floating-point boundary values.

    Args:
        raw_df: Raw DataFrame from pd.read_csv() on an SMD .txt file.
                Expected to have at least 6 columns (indices 0–5).

    Returns:
        DataFrame with 5 float columns in [0, 1].
    """
    col_indices = list(SMD_COL_MAP.keys())  # [0, 1, 2, 3, 5]
    df = raw_df[col_indices].copy()
    df.columns = [SMD_COL_MAP[i] for i in col_indices]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0.0, 1.0)
    return df

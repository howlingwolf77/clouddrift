"""
CloudDrift detection service.

Handles artifact loading and anomaly scoring logic for the FastAPI endpoints.

Anomaly scoring strategy:
    Single-snapshot (/detect):
        Z-score deviation from the Alibaba training distribution.
        Each metric's deviation from its training mean is computed in
        units of standard deviation. The anomaly score is derived from
        the top-3 absolute z-scores via the tanh function, which maps
        naturally to the [0, 1] range and the Critical/Warning/Normal
        severity thresholds.

        This is "Track 1" of CloudDrift's two-track explainability design —
        fast, always available, no rolling window context required.

    Batch (/batch_detect):
        Same z-score scoring applied per snapshot, results ranked by score.
        This endpoint is designed to be extended with the full IF+TCN
        ensemble when rolling feature context is available (>= 30 snapshots
        per machine in temporal order).

The IF and TCN artifacts are loaded at startup for /ready validation and
future extension of the batch endpoint to full ensemble scoring.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")

# Severity thresholds — consistent with src/models/ensemble.py constants
SEVERITY_CRITICAL = 0.8
SEVERITY_WARNING = 0.5


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_all_artifacts() -> dict:
    """
    Load all model artifacts and configuration files at API startup.

    Called once in the FastAPI lifespan context manager. The results are
    stored in `app.state.artifacts` and shared across all requests.

    Returns:
        Dict with keys for each artifact and a 'loaded' bool summary.
    """
    artifacts: dict = {}
    status: dict[str, bool] = {}

    def _try_load(key: str, loader_fn, path: Path) -> bool:
        try:
            artifacts[key] = loader_fn(path)
            status[key] = True
            logger.info("Loaded: %s", path)
            return True
        except Exception:
            logger.exception("FAILED to load artifact: %s", path)
            status[key] = False
            return False

    # Model artifacts
    _try_load(
        "isolation_forest",
        joblib.load,
        ARTIFACTS_DIR / "isolation_forest.joblib",
    )
    _try_load(
        "feature_pipeline",
        joblib.load,
        ARTIFACTS_DIR / "feature_pipeline.joblib",
    )
    _try_load(
        "thresholds",
        joblib.load,
        ARTIFACTS_DIR / "thresholds.joblib",
    )

    # TCN uses torch.load — load lazily with map_location="cpu"
    tcn_path = ARTIFACTS_DIR / "tcn_autoencoder.pt"
    try:
        from src.models.tcn_autoencoder import load_tcn_autoencoder

        artifacts["tcn_autoencoder"] = load_tcn_autoencoder(tcn_path)
        status["tcn_autoencoder"] = True
        logger.info("Loaded: %s", tcn_path)
    except Exception:
        logger.exception("FAILED to load TCN artifact: %s", tcn_path)
        status["tcn_autoencoder"] = False

    # JSON configuration files
    def _load_json(path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    for key, filename in [
        ("ensemble_meta", "ensemble_metadata.json"),
        ("feature_meta", "feature_metadata.json"),
        ("reference_stats", "reference_stats.json"),
        ("api_reference_stats", "api_reference_stats.json"),
    ]:
        _try_load(key, _load_json, ARTIFACTS_DIR / filename)

    artifacts["artifact_status"] = status
    artifacts["loaded"] = all(status.values())

    n_ok = sum(status.values())
    n_total = len(status)
    if artifacts["loaded"]:
        logger.info("All %d artifacts loaded successfully", n_total)
    else:
        failed = [k for k, v in status.items() if not v]
        logger.warning("%d/%d artifacts loaded. Failed: %s", n_ok, n_total, failed)

    return artifacts


# ---------------------------------------------------------------------------
# Single-snapshot scoring
# ---------------------------------------------------------------------------

METRIC_COLS = ["cpu_util", "mem_util", "net_io_in", "net_io_out", "disk_io"]


def score_snapshot(
    snapshot: dict,
    api_reference_stats: dict,
    thresholds: dict,
    n_top: int = 5,
) -> dict:
    """
    Score a single telemetry snapshot using z-score attribution (Track 1).

    Computes the absolute z-score for each metric relative to the
    Alibaba training distribution. The composite anomaly score maps
    the mean top-3 z-score through tanh, scaled so a 3σ deviation
    yields score ≈ 0.76 (Warning territory) and 5σ yields ≈ 0.93
    (Critical territory).

    Args:
        snapshot:            Dict of {metric_name: value} from TelemetrySnapshot.
        api_reference_stats: Per-metric mean/std from Alibaba training data.
        thresholds:          Calibrated ensemble thresholds dict.
        n_top:               Number of top contributing metrics to return.

    Returns:
        Dict with anomaly_score, severity_label, top_contributing_features,
        feature_deviation_scores.
    """
    z_scores: dict[str, float] = {}

    for metric in METRIC_COLS:
        value = snapshot.get(metric)
        if value is None:
            continue
        if metric not in api_reference_stats:
            continue

        stats = api_reference_stats[metric]
        mean = stats["mean"]
        std = stats["std"]
        z_scores[metric] = abs(float(value) - mean) / std

    if not z_scores:
        return {
            "anomaly_score": 0.0,
            "severity_label": "Normal",
            "top_contributing_features": [],
            "feature_deviation_scores": {},
        }

    # Composite score: tanh of mean of top-3 z-scores, scaled
    top_z = sorted(z_scores.values(), reverse=True)[:3]
    mean_top_z = float(np.mean(top_z))
    anomaly_score = float(np.tanh(mean_top_z / 3.0))
    anomaly_score = round(max(0.0, min(1.0, anomaly_score)), 4)

    # Severity label
    if anomaly_score >= SEVERITY_CRITICAL:
        severity = "Critical"
    elif anomaly_score >= SEVERITY_WARNING:
        severity = "Warning"
    else:
        severity = "Normal"

    # Ranked attribution
    sorted_metrics = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)
    top_n = sorted_metrics[:n_top]

    return {
        "anomaly_score": anomaly_score,
        "severity_label": severity,
        "top_contributing_features": [m for m, _ in top_n],
        "feature_deviation_scores": {m: round(s, 4) for m, s in top_n},
    }


def score_batch(
    snapshots: list[dict],
    api_reference_stats: dict,
    thresholds: dict,
) -> list[dict]:
    """
    Score a list of telemetry snapshots and return ranked results.

    Args:
        snapshots:           List of snapshot dicts from TelemetrySnapshot.
        api_reference_stats: Per-metric stats from Alibaba training data.
        thresholds:          Calibrated ensemble thresholds.

    Returns:
        List of result dicts sorted by anomaly_score descending, with
        rank, timestamp, and machine_id added to each result.
    """
    scored = []
    for snap in snapshots:
        result = score_snapshot(snap, api_reference_stats, thresholds)
        result["timestamp"] = snap.get("timestamp", "")
        result["machine_id"] = snap.get("machine_id")
        scored.append(result)

    scored.sort(key=lambda x: x["anomaly_score"], reverse=True)

    threshold_val = thresholds.get("isolation_forest", 0.5)
    for rank, item in enumerate(scored, 1):
        item["rank"] = rank

    return scored, threshold_val

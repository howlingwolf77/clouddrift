"""
CloudDrift detection service.

Handles artifact loading and anomaly scoring logic for the FastAPI endpoints.

Anomaly scoring strategy:

    Single-snapshot (/detect):
        Z-score deviation from the SMD training distribution (Track 1).
        Fast, stateless, no rolling window required.

    Batch (/batch_detect):
        Routes per machine_id group:
        - Group with machine_id AND >= 30 sequential snapshots:
            Full IF + TCN ensemble (Track 2). Feature engineering is applied
            to the sequence, IF scores and TCN reconstruction errors are
            combined at IF=0.40 / TCN=0.60 weights.
        - Group with < 30 snapshots OR no machine_id:
            Z-score fallback (same as /detect).

        The detection_mode field in each result indicates which path ran:
            "ensemble_if_tcn"    — full IF+TCN ensemble
            "single_point_zscore" — z-score fallback

        Note on TCN warm-up: with exactly 30 snapshots, only the last row
        gets a full reconstruction error (the first seq_length-1 rows cannot
        complete a window). NaN errors are filled with 0.0, making those rows
        IF-dominant in the ensemble score. With 60+ snapshots, most rows
        receive proper TCN scores.
"""

import logging
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")

# Severity thresholds — 0.75 = 3-sigma convention (tanh(3/3) ≈ 0.762)
SEVERITY_CRITICAL = 0.75
SEVERITY_WARNING = 0.50

# Minimum snapshots required for ensemble scoring
ENSEMBLE_MIN_SNAPSHOTS = 30

# API values are in [0, 100] (percentages); SMD training data is in [0, 1]
API_TO_SMD_SCALE = 100.0

# Raw metric columns expected from TelemetrySnapshot
METRIC_COLS = ["cpu_util", "mem_util", "net_io_in", "net_io_out", "disk_io"]


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_all_artifacts() -> dict:
    """
    Load all model artifacts and configuration files at API startup.

    Called once in the FastAPI lifespan context manager. Results are
    stored in app.state.artifacts and shared across all requests.

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
        "isolation_forest", joblib.load, ARTIFACTS_DIR / "isolation_forest.joblib"
    )
    _try_load(
        "feature_pipeline", joblib.load, ARTIFACTS_DIR / "feature_pipeline.joblib"
    )
    _try_load("thresholds", joblib.load, ARTIFACTS_DIR / "thresholds.joblib")

    # TCN uses torch.load
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
        import json

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
# Z-score attribution (Track 1)
# ---------------------------------------------------------------------------


def score_snapshot(
    snapshot: dict,
    api_reference_stats: dict,
    thresholds: dict,
    n_top: int = 5,
) -> dict:
    """
    Score a single telemetry snapshot using z-score attribution (Track 1).

    Computes |value - training_mean| / training_std per metric.
    Composite anomaly score = tanh(mean_top3_z / 3.0), which maps
    3-sigma deviation to approximately 0.762 (just above Critical threshold).

    Args:
        snapshot:            Dict of {metric_name: value} from TelemetrySnapshot.
        api_reference_stats: Per-metric mean/std from SMD training data
                             (scaled to [0, 100] to match API input range).
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

    top_z = sorted(z_scores.values(), reverse=True)[:3]
    mean_top_z = float(np.mean(top_z))
    anomaly_score = float(np.tanh(mean_top_z / 3.0))
    anomaly_score = round(max(0.0, min(1.0, anomaly_score)), 4)

    if anomaly_score >= SEVERITY_CRITICAL:
        severity = "Critical"
    elif anomaly_score >= SEVERITY_WARNING:
        severity = "Warning"
    else:
        severity = "Normal"

    sorted_metrics = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)
    top_n = sorted_metrics[:n_top]

    return {
        "anomaly_score": anomaly_score,
        "severity_label": severity,
        "top_contributing_features": [m for m, _ in top_n],
        "feature_deviation_scores": {m: round(s, 4) for m, s in top_n},
    }


# ---------------------------------------------------------------------------
# Ensemble inference helpers (Track 2)
# ---------------------------------------------------------------------------


def _patch_pipeline_bounds(pipeline) -> None:
    """
    Patch NaN/Inf normalizer bounds in-place.

    cpu_mem_corr_long produces NaN bounds when rolling Pearson correlation
    is undefined at series boundaries (fewer than 2 data points). Patching
    to (0.0, 1.0) is the same fix applied in the training scripts.
    """
    normalizer = pipeline.named_steps.get("normalizer")
    if normalizer is None or not hasattr(normalizer, "bounds_"):
        return
    for col in list(normalizer.bounds_.keys()):
        lo, hi = normalizer.bounds_[col]
        if math.isnan(lo) or math.isnan(hi) or math.isinf(lo) or math.isinf(hi):
            normalizer.bounds_[col] = (0.0, 1.0)


def _build_smd_df(snapshots: list[dict], machine_id: str) -> pd.DataFrame:
    """
    Convert raw API snapshots to a DataFrame in SMD [0, 1] scale.

    API values arrive in [0, 100] (percentage). SMD training data is in
    [0, 1]. Dividing by 100 puts the values in the same scale the feature
    pipeline and models were fitted on.

    Synthetic timestamps at 1-minute intervals are assigned since the API
    payload carries wall-clock timestamps that may be irregular or missing.
    The feature engineering only needs monotonic ordering within the group.

    Args:
        snapshots:  List of snapshot dicts from TelemetrySnapshot.model_dump().
        machine_id: Machine identifier — used as both machine_id and source_file.

    Returns:
        DataFrame with columns matching the SMD ingestion schema.
    """
    base_time = pd.Timestamp("2024-01-01 00:00:00")
    rows = []
    for i, snap in enumerate(snapshots):
        rows.append(
            {
                "machine_id": machine_id,
                "source_file": machine_id,  # required by TCN group_col
                "timestamp": base_time + pd.to_timedelta(i, unit="min"),
                "cpu_util": float(snap.get("cpu_util") or 0.0) / API_TO_SMD_SCALE,
                "mem_util": float(snap.get("mem_util") or 0.0) / API_TO_SMD_SCALE,
                "net_io_in": float(snap.get("net_io_in") or 0.0) / API_TO_SMD_SCALE,
                "net_io_out": float(snap.get("net_io_out") or 0.0) / API_TO_SMD_SCALE,
                "disk_io": float(snap.get("disk_io") or 0.0) / API_TO_SMD_SCALE,
                "is_anomaly": False,  # required by build_alibaba_features
            }
        )
    return pd.DataFrame(rows)


def _minmax_norm(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip-normalize arr to [0, 1] using pre-fitted bounds."""
    span = hi - lo
    if span <= 0:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / span, 0.0, 1.0)


def _score_group_ensemble(
    snapshots: list[dict],
    machine_id: str,
    artifacts: dict,
    api_reference_stats: dict,
    thresholds: dict,
) -> list[dict]:
    """
    Score a sequential group of snapshots with the full IF + TCN ensemble.

    Pipeline:
        1. Scale API [0,100] values to SMD [0,1]
        2. Build 68 rolling + cross-metric features per row
        3. Normalize using the fitted feature pipeline
        4. Compute IF anomaly scores
        5. Compute TCN reconstruction errors (sliding window seq_length=30)
        6. Normalize both score sets to [0,1] using training normal bounds
        7. Combine: IF_WEIGHT * if_norm + TCN_WEIGHT * tcn_norm
        8. Apply severity thresholds
        9. Attach z-score attribution for explainability

    Args:
        snapshots:           Time-ordered list of raw snapshot dicts.
        machine_id:          Machine identifier for this group.
        artifacts:           Loaded artifact dict from app.state.artifacts.
        api_reference_stats: Per-metric stats for z-score attribution.
        thresholds:          Calibrated threshold dict.

    Returns:
        List of result dicts (one per snapshot), in input order.
    """
    from src.features.engineering import apply_feature_pipeline, build_alibaba_features
    from src.models.isolation_forest import compute_anomaly_scores
    from src.models.tcn_autoencoder import compute_reconstruction_errors

    if_model = artifacts["isolation_forest"]
    tcn_model = artifacts["tcn_autoencoder"]
    pipeline = artifacts["feature_pipeline"]
    feature_meta = artifacts.get("feature_meta", {})
    ens_meta = artifacts.get("ensemble_meta", {})

    feature_cols = feature_meta.get("feature_cols", [])
    if not feature_cols:
        raise ValueError("feature_cols not found in feature_meta artifact")

    # Patch NaN bounds (cpu_mem_corr_long rolling Pearson edge case)
    _patch_pipeline_bounds(pipeline)

    # Ensemble configuration
    if_weight = float(ens_meta.get("if_weight", 0.40))
    tcn_weight = float(ens_meta.get("tcn_weight", 0.60))

    # Score normalization bounds — stored in ensemble_metadata.json
    # Falls back to Day 6 training-run values if not present
    if_bounds = ens_meta.get("if_bounds", {})
    tcn_bounds = ens_meta.get("tcn_bounds", {})
    if_lo = float(if_bounds.get("lower", 0.3776))
    if_hi = float(if_bounds.get("upper", 0.6589))
    tcn_lo = float(tcn_bounds.get("lower", 0.000549))
    tcn_hi = float(tcn_bounds.get("upper", 0.005620))

    # Step 1-2: Build SMD-scale DataFrame and engineer features
    raw_df = _build_smd_df(snapshots, machine_id)
    feat_df = build_alibaba_features(raw_df, group_col="machine_id")
    feat_df[feature_cols] = feat_df[feature_cols].fillna(0.0)

    # Step 3: Normalize
    norm_df = apply_feature_pipeline(pipeline, feat_df, feature_cols)
    norm_df[feature_cols] = norm_df[feature_cols].fillna(0.0)

    # Step 4: IF scores (higher = more anomalous)
    if_raw = compute_anomaly_scores(if_model, norm_df[feature_cols])
    if_norm = _minmax_norm(if_raw, if_lo, if_hi)

    # Step 5: TCN reconstruction errors
    # Rows that cannot complete a seq_length=30 window receive NaN → filled 0.0
    # With exactly 30 snapshots only the last row has a full TCN score.
    # With 60+ snapshots most rows receive proper TCN scores.
    tcn_errors = compute_reconstruction_errors(
        tcn_model, norm_df, feature_cols, group_col="source_file"
    )
    tcn_raw = tcn_errors.fillna(0.0).values
    tcn_norm = _minmax_norm(tcn_raw, tcn_lo, tcn_hi)

    # Step 6-7: Ensemble combination
    ensemble_scores = np.clip(if_weight * if_norm + tcn_weight * tcn_norm, 0.0, 1.0)

    # Step 8-9: Build per-row results with severity and z-score attribution
    results = []
    for snap, ens_score_f in zip(snapshots, ensemble_scores.tolist()):
        ens_score_f = round(float(ens_score_f), 4)

        if ens_score_f >= SEVERITY_CRITICAL:
            severity = "Critical"
        elif ens_score_f >= SEVERITY_WARNING:
            severity = "Warning"
        else:
            severity = "Normal"

        # Z-score attribution — always available regardless of detection mode
        zscore = score_snapshot(snap, api_reference_stats, thresholds, n_top=5)

        results.append(
            {
                "timestamp": snap.get("timestamp", ""),
                "machine_id": machine_id,
                "anomaly_score": ens_score_f,
                "severity_label": severity,
                "top_contributing_features": zscore["top_contributing_features"],
                "feature_deviation_scores": zscore["feature_deviation_scores"],
                "detection_mode": "ensemble_if_tcn",
            }
        )

    n_tcn_scored = int((tcn_errors.notna()).sum())
    logger.info(
        "Ensemble scored %d snapshots for machine %s "
        "(%d with full TCN reconstruction, %d IF-dominant)",
        len(results),
        machine_id,
        n_tcn_scored,
        len(results) - n_tcn_scored,
    )
    return results


# ---------------------------------------------------------------------------
# Batch scoring — routing logic
# ---------------------------------------------------------------------------


def score_batch(
    snapshots: list[dict],
    api_reference_stats: dict,
    thresholds: dict,
    artifacts: dict | None = None,
) -> tuple[list[dict], float, int, int]:
    """
    Score a list of telemetry snapshots, routing per machine_id group.

    Routing rules:
        - machine_id present AND group size >= ENSEMBLE_MIN_SNAPSHOTS (30)
            AND ensemble artifacts available
            → _score_group_ensemble()  (detection_mode: "ensemble_if_tcn")
        - otherwise
            → score_snapshot() loop    (detection_mode: "single_point_zscore")

    Mixed batches (some machines with >= 30 snapshots, others without)
    are fully supported — each group is routed independently.

    Ensemble failures fall back to z-score silently to preserve availability.

    Args:
        snapshots:           List of snapshot dicts from TelemetrySnapshot.
        api_reference_stats: Per-metric stats for z-score scoring.
        thresholds:          Calibrated threshold dict.
        artifacts:           Loaded artifact dict from app.state (optional).
                             Required for ensemble path; z-score only if None.

    Returns:
        Tuple of:
            ranked_results   — list of result dicts sorted by anomaly_score desc
            threshold_val    — ensemble threshold used for n_flagged calculation
            ensemble_count   — number of snapshots scored by IF+TCN ensemble
            zscore_count     — number of snapshots scored by z-score fallback
    """
    # Check whether ensemble artifacts are available
    ensemble_available = (
        artifacts is not None
        and artifacts.get("isolation_forest") is not None
        and artifacts.get("tcn_autoencoder") is not None
        and artifacts.get("feature_pipeline") is not None
        and artifacts.get("feature_meta") is not None
        and artifacts.get("ensemble_meta") is not None
    )

    # Group snapshots by machine_id, preserving original index for re-assembly
    groups: dict[str | None, list[tuple[int, dict]]] = defaultdict(list)
    for idx, snap in enumerate(snapshots):
        groups[snap.get("machine_id")].append((idx, snap))

    all_results: list[dict] = [{}] * len(snapshots)
    ensemble_count = 0
    zscore_count = 0

    for machine_id, indexed_snaps in groups.items():
        indices = [i for i, _ in indexed_snaps]
        group_snaps = [s for _, s in indexed_snaps]

        use_ensemble = (
            ensemble_available
            and machine_id is not None
            and len(group_snaps) >= ENSEMBLE_MIN_SNAPSHOTS
        )

        if use_ensemble:
            try:
                group_results = _score_group_ensemble(
                    group_snaps,
                    machine_id,
                    artifacts,
                    api_reference_stats,
                    thresholds,
                )
                for idx, result in zip(indices, group_results):
                    all_results[idx] = result
                ensemble_count += len(group_results)
                logger.info(
                    "Ensemble path: machine=%s, snapshots=%d",
                    machine_id,
                    len(group_snaps),
                )
            except Exception:
                logger.exception(
                    "Ensemble scoring failed for machine %s — falling back to z-score",
                    machine_id,
                )
                # Fallback: z-score for the whole group
                for idx, snap in zip(indices, group_snaps):
                    r = score_snapshot(snap, api_reference_stats, thresholds)
                    r["timestamp"] = snap.get("timestamp", "")
                    r["machine_id"] = machine_id
                    r["detection_mode"] = "single_point_zscore"
                    all_results[idx] = r
                zscore_count += len(group_snaps)
        else:
            reason = (
                f"< {ENSEMBLE_MIN_SNAPSHOTS} snapshots"
                if machine_id is not None
                else "no machine_id"
            )
            logger.debug(
                "Z-score path: machine=%s, snapshots=%d (%s)",
                machine_id,
                len(group_snaps),
                reason,
            )
            for idx, snap in zip(indices, group_snaps):
                r = score_snapshot(snap, api_reference_stats, thresholds)
                r["timestamp"] = snap.get("timestamp", "")
                r["machine_id"] = snap.get("machine_id")
                r["detection_mode"] = "single_point_zscore"
                all_results[idx] = r
            zscore_count += len(group_snaps)

    # Sort by anomaly_score descending and add ranks
    all_results.sort(key=lambda x: x.get("anomaly_score", 0.0), reverse=True)

    threshold_val = float(thresholds.get("ensemble", 0.566996))
    for rank, item in enumerate(all_results, 1):
        item["rank"] = rank

    logger.info(
        "score_batch complete: %d snapshots | ensemble=%d | zscore=%d",
        len(all_results),
        ensemble_count,
        zscore_count,
    )
    return all_results, threshold_val, ensemble_count, zscore_count

"""
Generate Missing API Artifacts.

Creates the five artifact files that the FastAPI startup requires but
that were not produced by the Day 4-6 training pipeline:

    artifacts/thresholds.joblib        — calibrated thresholds dict
    artifacts/ensemble_metadata.json   — weights, dataset, final metrics
    artifacts/feature_metadata.json    — feature column list and input_dim
    artifacts/reference_stats.json     — per-feature mean/std for Track 1
                                         z-score attribution (68 engineered
                                         features, in normalized [0,1] scale)
    artifacts/api_reference_stats.json — per-metric mean/std for the API's
                                         raw z-score scoring (5 raw metrics,
                                         scaled ×100 for [0, 100] API input)

Run from the project root after Days 4, 5, and 6 are complete:
    python generate_api_artifacts.py
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("gen_artifacts")

ARTIFACTS_DIR = Path("artifacts")
MACHINES = [f"machine-1-{i}" for i in range(1, 8)]

# Calibrated thresholds from Days 4, 5, and 6 logs
# Update these values if models are retrained
CALIBRATED_THRESHOLDS = {
    "isolation_forest": 0.591347,  # Day 4 log: p90 of val IF scores
    "tcn_autoencoder": 0.002988,  # Day 5 log: AE threshold val contamination
    "ensemble": 0.566996,  # Day 6 IF=0.40 log: ensemble threshold
}

# Ensemble configuration
ENSEMBLE_CONFIG = {
    "if_weight": 0.40,
    "tcn_weight": 0.60,
    "dataset": "SMD",
    "machines": MACHINES,
    "n_machines": len(MACHINES),
    "test_auc_roc": 0.899,
    "test_precision": 0.567,
    "test_recall": 0.762,
    "test_f1": 0.650,
    "test_f2": 0.713,
    "val_auc_roc": 0.868,
    "if_bounds": {"lower": 0.3776, "upper": 0.6589},
    "tcn_bounds": {"lower": 0.000549, "upper": 0.005620},
    "val_metrics": {
        "auc_roc": 0.868,
        "precision": 0.284,
        "recall": 0.426,
        "f1": 0.341,
        "f2": 0.388,
    },
}

# Raw API metric columns — match TelemetrySnapshot fields
API_METRIC_COLS = ["cpu_util", "mem_util", "net_io_in", "net_io_out", "disk_io"]

# ---------------------------------------------------------------------------
# Step 1 — Load Day 4 artifacts (feature_cols, pipeline)
# ---------------------------------------------------------------------------

logger.info("=== Step 1: Loading Day 4 artifacts ===")

metrics_path = ARTIFACTS_DIR / "day4_if_metrics.json"
if not metrics_path.exists():
    raise FileNotFoundError(
        f"Day 4 metrics not found at {metrics_path}. Run day4_if_training_smd.py first."
    )

with open(metrics_path) as f:
    day4 = json.load(f)

feature_cols = day4["feature_cols"]
input_dim = day4["input_dim_for_tcn"]
logger.info("feature_cols: %d | input_dim: %d", len(feature_cols), input_dim)

from src.features.engineering import load_feature_pipeline

pipeline = load_feature_pipeline()
normalizer = pipeline.named_steps["normalizer"]

# Patch NaN bounds (cpu_mem_corr_long)
n_patched = 0
for col in list(normalizer.bounds_.keys()):
    lo, hi = normalizer.bounds_[col]
    if np.isnan(lo) or np.isnan(hi) or np.isinf(lo) or np.isinf(hi):
        normalizer.bounds_[col] = (0.0, 1.0)
        n_patched += 1
logger.info("NaN bounds patched: %d columns", n_patched)

# ---------------------------------------------------------------------------
# Step 2 — Recreate normalized training split
# ---------------------------------------------------------------------------

logger.info("=== Step 2: Recreating training split ===")

from src.data.ingestion import load_smd_dataset
from src.data.validation import define_temporal_split_per_series, validate_smd_schema
from src.features.engineering import apply_feature_pipeline, build_alibaba_features

raw_df = load_smd_dataset(machines=MACHINES)
raw_df = validate_smd_schema(raw_df)
feat_df = build_alibaba_features(raw_df, group_col="machine_id")

train_df, _, _ = define_temporal_split_per_series(
    feat_df,
    group_col="machine_id",
    train_pct=0.70,
    val_pct=0.15,
)

train_norm = apply_feature_pipeline(pipeline, train_df, feature_cols)

n_nan = train_norm[feature_cols].isna().sum().sum()
if n_nan > 0:
    train_norm[feature_cols] = train_norm[feature_cols].fillna(0.0)
    logger.warning("train_norm: %d NaN values filled", n_nan)

train_normal = train_norm[~train_norm["is_anomaly"]]
logger.info(
    "Training normal rows: %s | anomaly rows excluded: %s",
    f"{len(train_normal):,}",
    f"{train_norm['is_anomaly'].sum():,}",
)

# ---------------------------------------------------------------------------
# Step 3 — Build reference_stats (68 engineered features, normalized scale)
# ---------------------------------------------------------------------------

logger.info("=== Step 3: Building reference_stats.json (68 features) ===")

from src.utils.explanation import build_reference_stats, save_reference_stats

ref_stats = build_reference_stats(train_normal, feature_cols)
save_reference_stats(ref_stats, ARTIFACTS_DIR / "reference_stats.json")
logger.info("reference_stats.json: %d features saved", len(ref_stats))

# ---------------------------------------------------------------------------
# Step 4 — Build api_reference_stats (5 raw metrics, ×100 for [0,100] API)
#
# TelemetrySnapshot validates raw metric inputs in [0, 100] (percentages).
# SMD training data is in [0, 1]. Multiply by 100 so that z-scores are
# sensible when the API receives e.g. cpu_util=45.0 (45% utilization).
# ---------------------------------------------------------------------------

logger.info("=== Step 4: Building api_reference_stats.json (5 raw metrics) ===")

api_ref_stats: dict = {}
for col in API_METRIC_COLS:
    if col not in train_normal.columns:
        logger.warning("Column '%s' not in train_normal — skipping", col)
        continue
    col_data = train_normal[col].dropna() * 100.0  # scale [0,1] → [0,100]
    std = float(col_data.std())
    api_ref_stats[col] = {
        "mean": round(float(col_data.mean()), 4),
        "std": round(std if std > 1e-4 else 1.0, 4),
        "min": round(float(col_data.min()), 4),
        "max": round(float(col_data.max()), 4),
        "scale": "percent_0_to_100",
    }
    logger.info(
        "  %-12s  mean=%.2f  std=%.2f  range=[%.2f, %.2f]",
        col,
        api_ref_stats[col]["mean"],
        api_ref_stats[col]["std"],
        api_ref_stats[col]["min"],
        api_ref_stats[col]["max"],
    )

api_ref_path = ARTIFACTS_DIR / "api_reference_stats.json"
with open(api_ref_path, "w") as f:
    json.dump(api_ref_stats, f, indent=2)
logger.info("api_reference_stats.json saved: %s", api_ref_path)

# ---------------------------------------------------------------------------
# Step 5 — Save thresholds.joblib
# ---------------------------------------------------------------------------

logger.info("=== Step 5: Saving thresholds.joblib ===")

thresholds_path = ARTIFACTS_DIR / "thresholds.joblib"
joblib.dump(CALIBRATED_THRESHOLDS, thresholds_path)
logger.info(
    "thresholds.joblib saved: IF=%.6f | TCN=%.6f | ensemble=%.6f",
    CALIBRATED_THRESHOLDS["isolation_forest"],
    CALIBRATED_THRESHOLDS["tcn_autoencoder"],
    CALIBRATED_THRESHOLDS["ensemble"],
)

# ---------------------------------------------------------------------------
# Step 6 — Save ensemble_metadata.json
# ---------------------------------------------------------------------------

logger.info("=== Step 6: Saving ensemble_metadata.json ===")

ensemble_meta_path = ARTIFACTS_DIR / "ensemble_metadata.json"
with open(ensemble_meta_path, "w") as f:
    json.dump(ENSEMBLE_CONFIG, f, indent=2)
logger.info("ensemble_metadata.json saved")

# ---------------------------------------------------------------------------
# Step 7 — Save feature_metadata.json
# ---------------------------------------------------------------------------

logger.info("=== Step 7: Saving feature_metadata.json ===")

feature_meta = {
    "feature_cols": feature_cols,
    "n_features": len(feature_cols),
    "input_dim": input_dim,
    "dataset": "SMD",
    "metric_cols": API_METRIC_COLS,
    "seq_length": 30,
    "nan_patched_cols": ["cpu_mem_corr_long"],
}

feature_meta_path = ARTIFACTS_DIR / "feature_metadata.json"
with open(feature_meta_path, "w") as f:
    json.dump(feature_meta, f, indent=2)
logger.info("feature_metadata.json saved: %d feature cols", len(feature_cols))

# ---------------------------------------------------------------------------
# Final confirmation
# ---------------------------------------------------------------------------

expected = [
    "thresholds.joblib",
    "ensemble_metadata.json",
    "feature_metadata.json",
    "reference_stats.json",
    "api_reference_stats.json",
    "isolation_forest.joblib",
    "tcn_autoencoder.pt",
    "feature_pipeline.joblib",
]

print("\n" + "=" * 52)
print("ARTIFACT GENERATION COMPLETE")
print("=" * 52)
all_ok = True
for name in expected:
    path = ARTIFACTS_DIR / name
    exists = path.exists()
    size = f"{path.stat().st_size / 1024:.1f} KB" if exists else "MISSING"
    status = "✓" if exists else "✗"
    print(f"  {status}  {name:<38} {size}")
    if not exists:
        all_ok = False

print("=" * 52)
print(
    f"API /ready status: {'PASS — all artifacts present' if all_ok else 'FAIL — missing artifacts above'}"
)
print("=" * 52 + "\n")

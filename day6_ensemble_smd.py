"""
Day 6 — Ensemble Scoring Pipeline (SMD dataset).

Combines Isolation Forest (5% weight) and TCN Autoencoder (95% weight)
into a weighted ensemble anomaly detector. Both models were trained on
the same 7-machine SMD subset in Days 4 and 5.

Depends on Day 4 and Day 5 artifacts:
    artifacts/isolation_forest.joblib
    artifacts/tcn_autoencoder.pt
    artifacts/feature_pipeline.joblib
    artifacts/day4_if_metrics.json

Pipeline steps:
    1.  Load Day 4/5 artifacts (feature_cols, input_dim, pipeline)
    2.  Recreate normalized feature splits (same as Days 4 and 5)
    3.  Run ensemble pipeline (IF + TCN weighted combination)
    4.  Display results and comparison against individual models
    5.  Save ensemble metrics artifact

Run from the project root:
    python day6_ensemble_smd.py
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("day6_smd")

ARTIFACTS_DIR = Path("artifacts")
MACHINES = [f"machine-1-{i}" for i in range(1, 8)]

# ---------------------------------------------------------------------------
# Step 1 — Load Day 4/5 artifacts
# ---------------------------------------------------------------------------

logger.info("=== Step 1: Loading Day 4/5 artifacts ===")

metrics_path = ARTIFACTS_DIR / "day4_if_metrics.json"
if not metrics_path.exists():
    raise FileNotFoundError(
        f"Day 4 metrics not found at {metrics_path}. Run day4_if_training_smd.py first."
    )

with open(metrics_path) as f:
    day4_metrics = json.load(f)

feature_cols = day4_metrics["feature_cols"]
input_dim = day4_metrics["input_dim_for_tcn"]

logger.info("feature_cols: %d columns | input_dim: %d", len(feature_cols), input_dim)

# Confirm both model artifacts exist before starting
for artifact in [
    "isolation_forest.joblib",
    "tcn_autoencoder.pt",
    "feature_pipeline.joblib",
]:
    path = ARTIFACTS_DIR / artifact
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}. "
            "Ensure Days 4 and 5 completed successfully."
        )
    logger.info(
        "Artifact confirmed: %s (%.1f KB)", artifact, path.stat().st_size / 1024
    )

import numpy as np

from src.features.engineering import load_feature_pipeline

pipeline = load_feature_pipeline()

# Patch NaN bounds (cpu_mem_corr_long)
normalizer = pipeline.named_steps["normalizer"]
n_patched = 0
for col in list(normalizer.bounds_.keys()):
    lo, hi = normalizer.bounds_[col]
    if np.isnan(lo) or np.isnan(hi) or np.isinf(lo) or np.isinf(hi):
        normalizer.bounds_[col] = (0.0, 1.0)
        logger.warning("Patched NaN bounds for '%s' → (0.0, 1.0)", col)
        n_patched += 1
logger.info("Pipeline bounds patched: %d columns", n_patched)

# ---------------------------------------------------------------------------
# Step 2 — Recreate normalized feature splits
# ---------------------------------------------------------------------------

logger.info("=== Step 2: Recreating normalized feature splits ===")

from src.data.ingestion import load_smd_dataset
from src.data.validation import define_temporal_split_per_series, validate_smd_schema
from src.features.engineering import apply_feature_pipeline, build_alibaba_features

raw_df = load_smd_dataset(machines=MACHINES)
raw_df = validate_smd_schema(raw_df)

feat_df = build_alibaba_features(raw_df, group_col="machine_id")

train_df, val_df, test_df = define_temporal_split_per_series(
    feat_df,
    group_col="machine_id",
    train_pct=0.70,
    val_pct=0.15,
)

train_norm = apply_feature_pipeline(pipeline, train_df, feature_cols)
val_norm = apply_feature_pipeline(pipeline, val_df, feature_cols)
test_norm = apply_feature_pipeline(pipeline, test_df, feature_cols)

# NaN guard
for label, df in [
    ("train_norm", train_norm),
    ("val_norm", val_norm),
    ("test_norm", test_norm),
]:
    n_nan = df[feature_cols].isna().sum().sum()
    if n_nan > 0:
        logger.warning("%s: %d NaN values — filling with 0.0", label, n_nan)
        df[feature_cols] = df[feature_cols].fillna(0.0)
    else:
        logger.info("%s: clean", label)

logger.info(
    "Splits — train: %s | val: %s (%.1f%% anom) | test: %s (%.1f%% anom)",
    f"{len(train_norm):,}",
    f"{len(val_norm):,}",
    val_norm["is_anomaly"].mean() * 100,
    f"{len(test_norm):,}",
    test_norm["is_anomaly"].mean() * 100,
)

# ---------------------------------------------------------------------------
# Step 3 — Run ensemble pipeline
# ---------------------------------------------------------------------------

logger.info("=== Step 3: Running ensemble pipeline (IF=5%%, TCN=95%%) ===")

from src.models.ensemble import run_ensemble_pipeline

results = run_ensemble_pipeline(
    train_df=train_norm,
    val_df=val_norm,
    test_df=test_norm,
    feature_cols=feature_cols,
)

# ---------------------------------------------------------------------------
# Step 4 — Display results
# ---------------------------------------------------------------------------

logger.info("=== Step 4: Ensemble results ===")

val_metrics = results["val_metrics"]
test_metrics = results["test_metrics"]

print("\n" + "=" * 60)
print("CLOUDDRIFT ENSEMBLE — SMD DATASET — 7 MACHINES")
print("=" * 60)

print("\n--- Validation Set ---")
print(
    f"  Precision:  {val_metrics['precision']:.3f}  (target ≥0.70: {'✓' if val_metrics['precision'] >= 0.70 else '✗'})"
)
print(
    f"  Recall:     {val_metrics['recall']:.3f}  (target ≥0.65: {'✓' if val_metrics['recall'] >= 0.65 else '✗'})"
)
print(f"  F1:         {val_metrics['f1']:.3f}")
print(f"  F2:         {val_metrics['f2']:.3f}")
print(f"  AUC-ROC:    {val_metrics['auc_roc']:.3f}")

print("\n--- Test Set ---")
print(
    f"  Precision:  {test_metrics['precision']:.3f}  (target ≥0.70: {'✓' if test_metrics['precision'] >= 0.70 else '✗'})"
)
print(
    f"  Recall:     {test_metrics['recall']:.3f}  (target ≥0.65: {'✓' if test_metrics['recall'] >= 0.65 else '✗'})"
)
print(f"  F1:         {test_metrics['f1']:.3f}")
print(f"  F2:         {test_metrics['f2']:.3f}")
print(f"  AUC-ROC:    {test_metrics['auc_roc']:.3f}")

# Load individual model metrics for comparison
day5_path = ARTIFACTS_DIR / "day5_tcn_metrics.json"
if day5_path.exists():
    with open(day5_path) as f:
        day5 = json.load(f)
    print("\n--- Component Comparison (Test AUC-ROC) ---")
    print(
        f"  Isolation Forest (standalone): {day4_metrics['test_metrics']['auc_roc']:.3f}"
    )
    print(f"  TCN Autoencoder (standalone):  {day5['test_metrics']['auc_roc']:.3f}")
    print(f"  Ensemble (IF 5%% + TCN 95%%):   {test_metrics['auc_roc']:.3f}")

print("\n--- Top Anomalies ---")
if "top_anomalies" in results and results["top_anomalies"] is not None:
    top = results["top_anomalies"].head(10)
    print(top.to_string(index=False))
else:
    logger.info("top_anomalies not returned by run_ensemble_pipeline")

print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# Step 5 — Save ensemble metrics
# ---------------------------------------------------------------------------

logger.info("=== Step 5: Saving ensemble metrics ===")

ensemble_metrics_out = {
    "dataset": "SMD",
    "machines": MACHINES,
    "n_machines": len(MACHINES),
    "weights": {"isolation_forest": 0.05, "tcn_autoencoder": 0.95},
    "val_metrics": val_metrics,
    "test_metrics": test_metrics,
    "component_test_auc_roc": {
        "isolation_forest": day4_metrics["test_metrics"]["auc_roc"],
        "tcn_autoencoder": day5["test_metrics"]["auc_roc"]
        if day5_path.exists()
        else None,
        "ensemble": test_metrics["auc_roc"],
    },
}

ensemble_path = ARTIFACTS_DIR / "day6_ensemble_metrics.json"
with open(ensemble_path, "w") as f:
    json.dump(ensemble_metrics_out, f, indent=2, default=str)

logger.info("Ensemble metrics saved to %s", ensemble_path)
logger.info("=== Day 6 complete. CloudDrift ensemble pipeline finished. ===")

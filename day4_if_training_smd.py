"""
Day 4 — Isolation Forest Training Pipeline (SMD dataset).

Replaces the previous NAB-based training with Server Machine Dataset (SMD),
which provides labeled multivariate server telemetry (CPU, memory, network,
disk) aligned to CloudDrift's anomaly detection use case.

Pipeline steps:
    1.  Load SMD (28 machines, train + test per machine)
    2.  Validate schema and data quality
    3.  Build features using build_alibaba_features() — reused without changes
    4.  Temporal split per machine (70% train / 15% val / 15% test)
    5.  Fit feature normalization pipeline on training normal rows
    6.  Apply normalization to all splits
    7.  TimeSeriesSplit cross-validation (5 folds)
    8.  Train final Isolation Forest on all training normal rows
    9.  Calibrate threshold on validation set
    10. Evaluate on validation and test sets
    11. Save model and feature pipeline artifacts

Run from the project root:
    python day4_if_training_smd.py

Or paste cells into notebooks/04_isolation_forest.ipynb in order.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("day4_smd")

# ---------------------------------------------------------------------------
# Step 1 — Load SMD
# ---------------------------------------------------------------------------

from src.data.ingestion import load_smd_dataset

logger.info("=== Step 1: Loading SMD dataset ===")
MACHINES = [f"machine-1-{i}" for i in range(1, 8)]  # 7 machines — ~3.4 GB RAM
raw_df = load_smd_dataset(machines=MACHINES)

logger.info(
    "Loaded: %s rows | %d machines | anomaly rate: %.2f%%",
    f"{len(raw_df):,}",
    raw_df["machine_id"].nunique(),
    raw_df["is_anomaly"].mean() * 100,
)

# ---------------------------------------------------------------------------
# Step 2 — Validate schema and data quality
# ---------------------------------------------------------------------------

from src.data.validation import generate_data_quality_report, validate_smd_schema

logger.info("=== Step 2: Validating schema and data quality ===")
raw_df = validate_smd_schema(raw_df)

quality_report = generate_data_quality_report(raw_df, dataset_name="SMD")
logger.info("Data quality overall pass: %s", quality_report["overall_pass"])

if not quality_report["overall_pass"]:
    raise RuntimeError(
        "SMD data quality check failed — review quality_report before proceeding."
    )

# ---------------------------------------------------------------------------
# Step 3 — Feature engineering
# ---------------------------------------------------------------------------

from src.features.engineering import (
    apply_feature_pipeline,
    build_alibaba_features,
    build_feature_pipeline,
    get_feature_columns,
    save_feature_pipeline,
)

logger.info(
    "=== Step 3: Building features (build_alibaba_features, group_col=machine_id) ==="
)
feat_df = build_alibaba_features(raw_df, group_col="machine_id")
feature_cols = get_feature_columns(feat_df)

logger.info(
    "Feature engineering complete: %s rows, %d feature columns",
    f"{len(feat_df):,}",
    len(feature_cols),
)
logger.info("Feature columns (first 10): %s", feature_cols[:10])
logger.info("input_dim for TCN (Day 5): %d", len(feature_cols))  # ← note for Day 5

# ---------------------------------------------------------------------------
# Step 4 — Temporal split per machine
# ---------------------------------------------------------------------------

from src.data.validation import define_temporal_split_per_series

logger.info("=== Step 4: Temporal split (70 / 15 / 15) per machine ===")
train_df, val_df, test_df = define_temporal_split_per_series(
    feat_df,
    group_col="machine_id",
    train_pct=0.70,
    val_pct=0.15,
)

logger.info(
    "Split sizes — train: %s | val: %s | test: %s",
    f"{len(train_df):,}",
    f"{len(val_df):,}",
    f"{len(test_df):,}",
)
logger.info(
    "Anomaly rates — train: %.2f%% | val: %.2f%% | test: %.2f%%",
    train_df["is_anomaly"].mean() * 100,
    val_df["is_anomaly"].mean() * 100,
    test_df["is_anomaly"].mean() * 100,
)

# ---------------------------------------------------------------------------
# Step 5 — Fit and save feature normalization pipeline
# ---------------------------------------------------------------------------

logger.info(
    "=== Step 5: Fitting feature normalization pipeline on train normal rows ==="
)
train_normal_rows = train_df[~train_df["is_anomaly"]].copy()

# NaN guard: cpu_mem_corr_long produces NaN bounds when rolling Pearson
# correlation is undefined (zero-variance window). Fill before fitting
# so all normalizer bounds are finite.
n_nan = train_normal_rows[feature_cols].isna().sum().sum()
if n_nan > 0:
    logger.warning(
        "train_normal_rows: %d NaN values in features — filling with 0.0 "
        "before fitting normalizer",
        n_nan,
    )
    train_normal_rows[feature_cols] = train_normal_rows[feature_cols].fillna(0.0)

pipeline = build_feature_pipeline(train_normal_rows, feature_cols)
save_feature_pipeline(pipeline)

# ---------------------------------------------------------------------------
# Step 6 — Apply normalization to all splits
# ---------------------------------------------------------------------------

logger.info("=== Step 6: Applying normalization ===")
train_norm = apply_feature_pipeline(pipeline, train_df, feature_cols)
val_norm = apply_feature_pipeline(pipeline, val_df, feature_cols)
test_norm = apply_feature_pipeline(pipeline, test_df, feature_cols)

# ---------------------------------------------------------------------------
# Step 7 — TimeSeriesSplit cross-validation
# ---------------------------------------------------------------------------

from src.models.isolation_forest import run_timeseries_cross_validation

logger.info("=== Step 7: TimeSeriesSplit cross-validation (5 folds) ===")
cv_results = run_timeseries_cross_validation(train_norm, feature_cols, n_splits=5)

summary = cv_results["summary"]
logger.info(
    "CV summary | mean_F1=%.3f std_F1=%.3f | mean_Recall=%.3f | stability=%s",
    summary["mean_f1"],
    summary["std_f1"],
    summary["mean_recall"],
    "PASS" if summary["stability_check_pass"] else "FAIL",
)

# ---------------------------------------------------------------------------
# Step 8 — Train final Isolation Forest on all training normal rows
# ---------------------------------------------------------------------------

from src.models.isolation_forest import save_isolation_forest, train_isolation_forest

logger.info("=== Step 8: Training final Isolation Forest ===")
x_train_normal = train_norm[~train_norm["is_anomaly"]][feature_cols]
if_model = train_isolation_forest(x_train_normal)

# ---------------------------------------------------------------------------
# Step 9 — Calibrate threshold on validation set
# ---------------------------------------------------------------------------

from src.models.isolation_forest import calibrate_threshold

logger.info("=== Step 9: Calibrating threshold on validation set ===")
x_val = val_norm[feature_cols]
threshold = calibrate_threshold(if_model, x_val, percentile=90.0)
logger.info("Calibrated threshold: %.6f", threshold)

# ---------------------------------------------------------------------------
# Step 10 — Evaluate on validation and test sets
# ---------------------------------------------------------------------------

from src.models.isolation_forest import evaluate_model

logger.info("=== Step 10: Evaluating model ===")
val_metrics = evaluate_model(
    if_model,
    threshold,
    val_norm[feature_cols],
    val_norm["is_anomaly"],
    split_name="validation",
)
test_metrics = evaluate_model(
    if_model,
    threshold,
    test_norm[feature_cols],
    test_norm["is_anomaly"],
    split_name="test",
)

logger.info("--- Validation Set ---")
logger.info(
    "  Precision:  %.3f (target ≥0.70: %s)",
    val_metrics["precision"],
    "✓" if val_metrics["meets_precision_target"] else "✗",
)
logger.info(
    "  Recall:     %.3f (target ≥0.65: %s)",
    val_metrics["recall"],
    "✓" if val_metrics["meets_recall_target"] else "✗",
)
logger.info("  F1:         %.3f", val_metrics["f1"])
logger.info("  AUC-ROC:    %.3f", val_metrics["auc_roc"])

logger.info("--- Test Set ---")
logger.info(
    "  Precision:  %.3f (target ≥0.70: %s)",
    test_metrics["precision"],
    "✓" if test_metrics["meets_precision_target"] else "✗",
)
logger.info(
    "  Recall:     %.3f (target ≥0.65: %s)",
    test_metrics["recall"],
    "✓" if test_metrics["meets_recall_target"] else "✗",
)
logger.info("  F1:         %.3f", test_metrics["f1"])
logger.info("  AUC-ROC:    %.3f", test_metrics["auc_roc"])

# ---------------------------------------------------------------------------
# Step 11 — Save artifacts
# ---------------------------------------------------------------------------


logger.info("=== Step 11: Saving artifacts ===")
save_isolation_forest(if_model)

# Save metrics and feature metadata for Day 5 and downstream notebooks
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

metrics_out = {
    "dataset": "SMD",
    "n_machines": int(raw_df["machine_id"].nunique()),
    "n_feature_cols": len(feature_cols),
    "input_dim_for_tcn": len(feature_cols),  # ← Day 5 must use this value
    "feature_cols": feature_cols,
    "cv_summary": summary,
    "threshold": threshold,
    "val_metrics": val_metrics,
    "test_metrics": test_metrics,
}

metrics_path = artifacts_dir / "day4_if_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics_out, f, indent=2, default=str)
logger.info("Metrics saved to %s", metrics_path)

logger.info("=== Day 4 complete. Artifacts written to artifacts/ ===")
logger.info(
    "IMPORTANT: Day 5 TCN must use input_dim=%d (logged in day4_if_metrics.json)",
    len(feature_cols),
)

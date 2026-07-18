"""
Day 5 — TCN Autoencoder Training Pipeline (SMD dataset).

Trains a Temporal Convolutional Network Autoencoder on normal-behavior
sequences from the SMD dataset. The model learns to reconstruct normal
server telemetry; anomalous sequences produce elevated reconstruction
error at inference time.

Depends on Day 4 artifacts:
    artifacts/feature_pipeline.joblib  — fitted normalization pipeline
    artifacts/day4_if_metrics.json     — feature column list and input_dim

Pipeline steps:
    1.  Load Day 4 artifacts (feature_cols, input_dim, fitted pipeline)
    2.  Recreate normalized feature splits (same pipeline as Day 4)
    3.  Build SequenceDatasets (sliding windows of length 30)
    4.  Train TCN Autoencoder with early stopping (patience=5, max 100 epochs)
    5.  Calibrate reconstruction error threshold on validation set
    6.  Evaluate on validation and test sets
    7.  Save model and metrics artifacts

Runtime note:
    Training runs on CPU (accelerator="cpu" in tcn_autoencoder.py).
    With 68 features and ~984k training rows, expect 3-8 minutes per epoch
    and early stopping typically triggering around epoch 10-20.

Run from the project root:
    python day5_tcn_training_smd.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when running from scripts/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("day5_smd")

ARTIFACTS_DIR = Path("artifacts")

# ---------------------------------------------------------------------------
# Step 1 — Load Day 4 artifacts
# ---------------------------------------------------------------------------

logger.info("=== Step 1: Loading Day 4 artifacts ===")

metrics_path = ARTIFACTS_DIR / "day4_if_metrics.json"
if not metrics_path.exists():
    raise FileNotFoundError(
        f"Day 4 metrics not found at {metrics_path}. Run day4_if_training_smd.py first."
    )

with open(metrics_path) as f:
    day4_metrics = json.load(f)

feature_cols = day4_metrics["feature_cols"]
input_dim = day4_metrics["input_dim_for_tcn"]

logger.info("feature_cols loaded: %d columns", len(feature_cols))
logger.info("input_dim for TCN: %d", input_dim)

from src.features.engineering import load_feature_pipeline

pipeline = load_feature_pipeline()

import numpy as np

normalizer = pipeline.named_steps["normalizer"]
n_patched = 0
for col in list(normalizer.bounds_.keys()):
    lo, hi = normalizer.bounds_[col]
    if np.isnan(lo) or np.isnan(hi) or np.isinf(lo) or np.isinf(hi):
        normalizer.bounds_[col] = (0.0, 1.0)
        logger.warning("Patched NaN/Inf bounds for '%s' → (0.0, 1.0)", col)
        n_patched += 1
logger.info("Pipeline bounds check: %d columns patched", n_patched)
logger.info("Feature pipeline loaded from artifacts/feature_pipeline.joblib")

# ---------------------------------------------------------------------------
# Step 2 — Recreate normalized feature splits
# (Day 4 did not save parquet — pipeline reruns in ~40 seconds)
# ---------------------------------------------------------------------------

logger.info("=== Step 2: Recreating normalized feature splits ===")

from src.data.ingestion import load_smd_dataset
from src.data.validation import (
    define_temporal_split_per_series,
    validate_smd_schema,
)
from src.features.engineering import apply_feature_pipeline, build_alibaba_features

logger.info("Loading SMD dataset...")
MACHINES = [f"machine-1-{i}" for i in range(1, 8)]  # 7 machines — ~3.4 GB RAM
raw_df = load_smd_dataset(machines=MACHINES)
raw_df = validate_smd_schema(raw_df)

logger.info("Building features...")
feat_df = build_alibaba_features(raw_df, group_col="machine_id")

logger.info("Splitting 70 / 15 / 15 per machine...")
train_df, val_df, test_df = define_temporal_split_per_series(
    feat_df,
    group_col="machine_id",
    train_pct=0.70,
    val_pct=0.15,
)

logger.info("Applying normalization pipeline...")
train_norm = apply_feature_pipeline(pipeline, train_df, feature_cols)
val_norm = apply_feature_pipeline(pipeline, val_df, feature_cols)
test_norm = apply_feature_pipeline(pipeline, test_df, feature_cols)

# ---------------------------------------------------------------------------
# Step 2b — NaN guard
# Normalization can produce NaN for constant or near-constant features at
# series boundaries. NaN in any feature causes NaN MSE loss, which prevents
# training entirely. Fill with 0.0 (equivalent to median-normal behavior).
# ---------------------------------------------------------------------------
logger.info("=== Step 2b: NaN guard on normalized features ===")
for label, df in [
    ("train_norm", train_norm),
    ("val_norm", val_norm),
    ("test_norm", test_norm),
]:
    n_nan = df[feature_cols].isna().sum().sum()
    if n_nan > 0:
        logger.warning(
            "%s: %d NaN values found in feature columns — filling with 0.0",
            label,
            n_nan,
        )
        df[feature_cols] = df[feature_cols].fillna(0.0)
    else:
        logger.info("%s: no NaN values — features are clean", label)

# Final assertion — training must not start with NaN features
assert train_norm[feature_cols].isna().sum().sum() == 0, (
    "NaN values remain in train_norm after fill — check normalizer bounds"
)

logger.info(
    "Splits ready — train: %s | val: %s | test: %s",
    f"{len(train_norm):,}",
    f"{len(val_norm):,}",
    f"{len(test_norm):,}",
)
logger.info(
    "Anomaly rates — train: %.2f%% | val: %.2f%% | test: %.2f%%",
    train_norm["is_anomaly"].mean() * 100,
    val_norm["is_anomaly"].mean() * 100,
    test_norm["is_anomaly"].mean() * 100,
)

# ---------------------------------------------------------------------------
# Step 3 — Build SequenceDatasets
# ---------------------------------------------------------------------------

logger.info("=== Step 3: Building SequenceDatasets (seq_length=30) ===")

from src.models.tcn_autoencoder import SEQ_LENGTH, create_sequence_dataset

# Training: normal rows only (the autoencoder learns normal patterns)
train_dataset = create_sequence_dataset(
    train_norm,
    feature_cols,
    seq_length=SEQ_LENGTH,
    normal_only=True,
)

# Validation: all rows (normal + anomalous — used to compute val_loss)
val_dataset = create_sequence_dataset(
    val_norm,
    feature_cols,
    seq_length=SEQ_LENGTH,
    normal_only=False,
)

logger.info(
    "Datasets — train sequences: %s | val sequences: %s",
    f"{len(train_dataset):,}",
    f"{len(val_dataset):,}",
)

# ---------------------------------------------------------------------------
# Step 4 — Train TCN Autoencoder
# ---------------------------------------------------------------------------

logger.info("=== Step 4: Training TCN Autoencoder ===")
logger.info(
    "input_dim=%d | seq_length=%d | max_epochs=100 | patience=5",
    input_dim,
    SEQ_LENGTH,
)
logger.info("Training on CPU. Expect 3-8 min/epoch; early stopping ~epoch 10-20.")

from src.models.tcn_autoencoder import train_tcn_autoencoder

model, best_ckpt = train_tcn_autoencoder(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    input_dim=input_dim,
    checkpoint_dir=ARTIFACTS_DIR / "checkpoints",
)

logger.info("Training complete. Best checkpoint: %s", best_ckpt)

# ---------------------------------------------------------------------------
# Step 5 — Calibrate reconstruction error threshold
# ---------------------------------------------------------------------------

logger.info("=== Step 5: Calibrating threshold on validation set ===")

from src.models.tcn_autoencoder import calibrate_autoencoder_threshold

threshold = calibrate_autoencoder_threshold(model, val_norm, feature_cols)
logger.info("Calibrated threshold: %.6f", threshold)

# ---------------------------------------------------------------------------
# Step 6 — Evaluate on validation and test sets
# ---------------------------------------------------------------------------

logger.info("=== Step 6: Evaluating TCN Autoencoder ===")

from src.models.tcn_autoencoder import evaluate_autoencoder

val_metrics = evaluate_autoencoder(
    model, threshold, val_norm, feature_cols, split_name="validation"
)
test_metrics = evaluate_autoencoder(
    model, threshold, test_norm, feature_cols, split_name="test"
)

logger.info("--- Validation Set ---")
logger.info("  Precision:  %.3f", val_metrics["precision"])
logger.info("  Recall:     %.3f", val_metrics["recall"])
logger.info("  F1:         %.3f", val_metrics["f1"])
logger.info("  AUC-ROC:    %.3f", val_metrics["auc_roc"])
logger.info(
    "  Error separation (anomaly > normal): %s",
    "✓"
    if val_metrics["error_mean_anomaly"] > val_metrics["error_mean_normal"]
    else "✗",
)

logger.info("--- Test Set ---")
logger.info("  Precision:  %.3f", test_metrics["precision"])
logger.info("  Recall:     %.3f", test_metrics["recall"])
logger.info("  F1:         %.3f", test_metrics["f1"])
logger.info("  AUC-ROC:    %.3f", test_metrics["auc_roc"])
logger.info(
    "  Error separation (anomaly > normal): %s",
    "✓"
    if test_metrics["error_mean_anomaly"] > test_metrics["error_mean_normal"]
    else "✗",
)

# ---------------------------------------------------------------------------
# Step 7 — Save model and metrics
# ---------------------------------------------------------------------------

logger.info("=== Step 7: Saving artifacts ===")

from src.models.tcn_autoencoder import save_tcn_autoencoder

save_tcn_autoencoder(model)

metrics_out = {
    "dataset": "SMD",
    "input_dim": input_dim,
    "seq_length": SEQ_LENGTH,
    "best_checkpoint": best_ckpt,
    "threshold": threshold,
    "val_metrics": val_metrics,
    "test_metrics": test_metrics,
}

day5_metrics_path = ARTIFACTS_DIR / "day5_tcn_metrics.json"
with open(day5_metrics_path, "w") as f:
    json.dump(metrics_out, f, indent=2, default=str)

logger.info("TCN metrics saved to %s", day5_metrics_path)
logger.info("=== Day 5 complete. Artifacts written to artifacts/ ===")
logger.info("Next: run the ensemble pipeline (Day 6) using both IF and TCN artifacts.")

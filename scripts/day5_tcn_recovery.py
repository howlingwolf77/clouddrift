"""
Day 5 Recovery — Load best TCN checkpoint and complete Steps 5-7.

Run this after interrupting day5_tcn_training_smd.py.
The best checkpoint is already saved; this script loads it and
runs threshold calibration, evaluation, and artifact saving.
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
logger = logging.getLogger("day5_recovery")

ARTIFACTS_DIR = Path("artifacts")
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"

# ---------------------------------------------------------------------------
# Step R1 — Find best checkpoint
# ---------------------------------------------------------------------------

logger.info("=== Step R1: Finding best checkpoint ===")

checkpoints = list(CHECKPOINT_DIR.glob("tcn_best_*.ckpt"))
if not checkpoints:
    raise FileNotFoundError(
        f"No checkpoints found in {CHECKPOINT_DIR}. "
        "Training must have not saved any checkpoint."
    )


# Checkpoint filenames: tcn_best_epoch=XX_val_loss=0.002.ckpt
# Sort by val_loss ascending and take the lowest
def _val_loss_from_path(p: Path) -> float:
    try:
        return float(p.stem.split("val_loss=")[1])
    except (IndexError, ValueError):
        return float("inf")


best_ckpt = min(checkpoints, key=_val_loss_from_path)
logger.info("Best checkpoint: %s", best_ckpt)
logger.info("val_loss: %.6f", _val_loss_from_path(best_ckpt))

# ---------------------------------------------------------------------------
# Step R2 — Reload data (same as Day 5 Steps 1-3, fast ~40 seconds)
# ---------------------------------------------------------------------------

logger.info("=== Step R2: Reloading data and feature splits ===")

with open(ARTIFACTS_DIR / "day4_if_metrics.json") as f:
    day4_metrics = json.load(f)

feature_cols = day4_metrics["feature_cols"]
input_dim = day4_metrics["input_dim_for_tcn"]
MACHINES = [f"machine-1-{i}" for i in range(1, 8)]

import numpy as np

from src.features.engineering import load_feature_pipeline

pipeline = load_feature_pipeline()

# Patch NaN bounds (same as day5 fix)
normalizer = pipeline.named_steps["normalizer"]
n_patched = 0
for col in list(normalizer.bounds_.keys()):
    lo, hi = normalizer.bounds_[col]
    if np.isnan(lo) or np.isnan(hi) or np.isinf(lo) or np.isinf(hi):
        normalizer.bounds_[col] = (0.0, 1.0)
        logger.warning("Patched NaN bounds for '%s'", col)
        n_patched += 1
logger.info("Pipeline bounds patched: %d columns", n_patched)

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

logger.info(
    "Data ready — val: %s rows (%.1f%% anomaly) | test: %s rows (%.1f%% anomaly)",
    f"{len(val_norm):,}",
    val_norm["is_anomaly"].mean() * 100,
    f"{len(test_norm):,}",
    test_norm["is_anomaly"].mean() * 100,
)

# ---------------------------------------------------------------------------
# Step R3 — Load model from best checkpoint
# ---------------------------------------------------------------------------

logger.info("=== Step R3: Loading model from checkpoint ===")

from src.models.tcn_autoencoder import TCNAutoencoder

model = TCNAutoencoder.load_from_checkpoint(
    str(best_ckpt),
    input_dim=input_dim,
)
model.eval()
logger.info(
    "Model loaded: input_dim=%d, params=%d",
    input_dim,
    sum(p.numel() for p in model.parameters()),
)

# ---------------------------------------------------------------------------
# Step R4 — Calibrate threshold
# ---------------------------------------------------------------------------

logger.info("=== Step R4: Calibrating threshold on validation set ===")

from src.models.tcn_autoencoder import SEQ_LENGTH, calibrate_autoencoder_threshold

threshold = calibrate_autoencoder_threshold(model, val_norm, feature_cols)
logger.info("Calibrated threshold: %.6f", threshold)

# ---------------------------------------------------------------------------
# Step R5 — Evaluate
# ---------------------------------------------------------------------------

logger.info("=== Step R5: Evaluating on validation and test sets ===")

from src.models.tcn_autoencoder import evaluate_autoencoder

val_metrics = evaluate_autoencoder(
    model, threshold, val_norm, feature_cols, split_name="validation"
)
test_metrics = evaluate_autoencoder(
    model, threshold, test_norm, feature_cols, split_name="test"
)

print("\n--- Validation Set ---")
print(f"  Precision:            {val_metrics['precision']:.3f}")
print(f"  Recall:               {val_metrics['recall']:.3f}")
print(f"  F1:                   {val_metrics['f1']:.3f}")
print(f"  AUC-ROC:              {val_metrics['auc_roc']:.3f}")
print(f"  Error mean (normal):  {val_metrics['error_mean_normal']:.4f}")
print(f"  Error mean (anomaly): {val_metrics['error_mean_anomaly']:.4f}")
sep_val = val_metrics["error_mean_anomaly"] > val_metrics["error_mean_normal"]
print(f"  Error separation:     {'✓' if sep_val else '✗'}")

print("\n--- Test Set ---")
print(f"  Precision:            {test_metrics['precision']:.3f}")
print(f"  Recall:               {test_metrics['recall']:.3f}")
print(f"  F1:                   {test_metrics['f1']:.3f}")
print(f"  AUC-ROC:              {test_metrics['auc_roc']:.3f}")
print(f"  Error mean (normal):  {test_metrics['error_mean_normal']:.4f}")
print(f"  Error mean (anomaly): {test_metrics['error_mean_anomaly']:.4f}")
sep_test = test_metrics["error_mean_anomaly"] > test_metrics["error_mean_normal"]
print(f"  Error separation:     {'✓' if sep_test else '✗'}")

# ---------------------------------------------------------------------------
# Step R6 — Save artifacts
# ---------------------------------------------------------------------------

logger.info("=== Step R6: Saving artifacts ===")

from src.models.tcn_autoencoder import save_tcn_autoencoder

save_tcn_autoencoder(model)

metrics_out = {
    "dataset": "SMD",
    "machines": MACHINES,
    "input_dim": input_dim,
    "seq_length": SEQ_LENGTH,
    "best_checkpoint": str(best_ckpt),
    "val_loss_at_checkpoint": _val_loss_from_path(best_ckpt),
    "threshold": threshold,
    "val_metrics": val_metrics,
    "test_metrics": test_metrics,
}

day5_path = ARTIFACTS_DIR / "day5_tcn_metrics.json"
with open(day5_path, "w") as f:
    json.dump(metrics_out, f, indent=2, default=str)

logger.info("TCN model saved:    artifacts/tcn_autoencoder.pt")
logger.info("TCN metrics saved:  %s", day5_path)
logger.info("=== Recovery complete. Ready for ensemble (Day 6). ===")

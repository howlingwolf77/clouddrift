"""
Day 6b — Ensemble Weight Sweep.

Finds the optimal IF/TCN weight combination by sweeping IF weight from
0.0 to 1.0 in steps of 0.05 and evaluating AUC-ROC on the validation set.
Scores are computed ONCE; the sweep itself is instant matrix operations.

Critical: the optimal weight is selected on the VALIDATION set only.
Test AUC-ROC at the optimal weight is then reported as the final score.
Tuning on the test set would inflate results and invalidate the benchmark.

Depends on Day 4 and Day 5 artifacts:
    artifacts/isolation_forest.joblib
    artifacts/tcn_autoencoder.pt
    artifacts/feature_pipeline.joblib
    artifacts/day4_if_metrics.json

Run from the project root:
    python day6b_weight_sweep.py
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

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("day6b_sweep")

ARTIFACTS_DIR = Path("artifacts")
MACHINES = [f"machine-1-{i}" for i in range(1, 8)]
WEIGHT_STEP = 0.05  # sweep IF weight 0.00 → 1.00 in 0.05 increments

# ---------------------------------------------------------------------------
# Step 1 — Load artifacts and recreate normalized splits
# (identical to Day 6 Steps 1-2 — reuse for reproducibility)
# ---------------------------------------------------------------------------

logger.info("=== Step 1: Loading artifacts ===")

with open(ARTIFACTS_DIR / "day4_if_metrics.json") as f:
    day4_metrics = json.load(f)

feature_cols = day4_metrics["feature_cols"]

from src.features.engineering import load_feature_pipeline

pipeline = load_feature_pipeline()

# Patch NaN bounds (cpu_mem_corr_long)
normalizer = pipeline.named_steps["normalizer"]
for col in list(normalizer.bounds_.keys()):
    lo, hi = normalizer.bounds_[col]
    if np.isnan(lo) or np.isnan(hi) or np.isinf(lo) or np.isinf(hi):
        normalizer.bounds_[col] = (0.0, 1.0)
        logger.warning("Patched NaN bounds for '%s'", col)

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

for label, df in [
    ("train_norm", train_norm),
    ("val_norm", val_norm),
    ("test_norm", test_norm),
]:
    n_nan = df[feature_cols].isna().sum().sum()
    if n_nan > 0:
        df[feature_cols] = df[feature_cols].fillna(0.0)
        logger.warning("%s: %d NaN values filled", label, n_nan)

val_labels = val_norm["is_anomaly"].values
test_labels = test_norm["is_anomaly"].values

# ---------------------------------------------------------------------------
# Step 3 — Compute raw IF and TCN scores (the slow step — runs once)
# ---------------------------------------------------------------------------

logger.info("=== Step 3: Computing component scores (runs once) ===")

# --- Isolation Forest scores ---
from src.models.isolation_forest import compute_anomaly_scores, load_isolation_forest

if_model = load_isolation_forest()

x_train_normal = train_norm[~train_norm["is_anomaly"]][feature_cols].values
x_val = val_norm[feature_cols].values
x_test = test_norm[feature_cols].values

if_train_scores = compute_anomaly_scores(if_model, x_train_normal)
if_val_scores = compute_anomaly_scores(if_model, x_val)
if_test_scores = compute_anomaly_scores(if_model, x_test)

logger.info(
    "IF scores — train_normal: [%.4f, %.4f] | val: [%.4f, %.4f] | test: [%.4f, %.4f]",
    if_train_scores.min(),
    if_train_scores.max(),
    if_val_scores.min(),
    if_val_scores.max(),
    if_test_scores.min(),
    if_test_scores.max(),
)

# --- TCN reconstruction error scores ---
from src.models.tcn_autoencoder import (
    compute_reconstruction_errors,
    load_tcn_autoencoder,
)

tcn_model = load_tcn_autoencoder()

logger.info("Computing TCN reconstruction errors (val)...")
tcn_val_errors = compute_reconstruction_errors(
    tcn_model, val_norm, feature_cols, group_col="source_file"
)

logger.info("Computing TCN reconstruction errors (test)...")
tcn_test_errors = compute_reconstruction_errors(
    tcn_model, test_norm, feature_cols, group_col="source_file"
)

# TCN training normal errors — needed for normalization bounds
logger.info("Computing TCN reconstruction errors (train normal)...")
tcn_train_errors = compute_reconstruction_errors(
    tcn_model,
    train_norm[~train_norm["is_anomaly"]],
    feature_cols,
    group_col="source_file",
)

tcn_train_scores = tcn_train_errors.fillna(0.0).values
tcn_val_scores = tcn_val_errors.fillna(0.0).values
tcn_test_scores = tcn_test_errors.fillna(0.0).values

logger.info(
    "TCN scores — train_normal: [%.6f, %.6f] | val: [%.6f, %.6f] | test: [%.6f, %.6f]",
    tcn_train_scores.min(),
    tcn_train_scores.max(),
    tcn_val_scores.min(),
    tcn_val_scores.max(),
    tcn_test_scores.min(),
    tcn_test_scores.max(),
)

# ---------------------------------------------------------------------------
# Step 4 — Normalize both score sets to [0, 1] using training normal bounds
# ---------------------------------------------------------------------------

logger.info("=== Step 4: Normalizing scores to [0, 1] ===")


def _minmax_norm(scores, lo, hi):
    """Clip-normalize scores to [0, 1] using pre-fitted bounds."""
    span = hi - lo
    if span == 0:
        return np.zeros_like(scores, dtype=float)
    return np.clip((scores - lo) / span, 0.0, 1.0)


# IF bounds from training normal rows
if_lo, if_hi = float(if_train_scores.min()), float(if_train_scores.max())
if_val_norm_arr = _minmax_norm(if_val_scores, if_lo, if_hi)
if_test_norm_arr = _minmax_norm(if_test_scores, if_lo, if_hi)

# TCN bounds from training normal rows
tcn_lo, tcn_hi = float(tcn_train_scores.min()), float(tcn_train_scores.max())
tcn_val_norm_arr = _minmax_norm(tcn_val_scores, tcn_lo, tcn_hi)
tcn_test_norm_arr = _minmax_norm(tcn_test_scores, tcn_lo, tcn_hi)

logger.info("IF  normalization bounds: [%.4f, %.4f]", if_lo, if_hi)
logger.info("TCN normalization bounds: [%.6f, %.6f]", tcn_lo, tcn_hi)

# ---------------------------------------------------------------------------
# Step 5 — Weight sweep (instant — pure numpy operations)
# ---------------------------------------------------------------------------

logger.info("=== Step 5: Weight sweep (IF 0.00→1.00, step=%.2f) ===", WEIGHT_STEP)

from sklearn.metrics import roc_auc_score

weights = np.round(np.arange(0.0, 1.0 + WEIGHT_STEP, WEIGHT_STEP), 2)
results_rows = []

for if_w in weights:
    tcn_w = round(1.0 - if_w, 2)

    val_ensemble = if_w * if_val_norm_arr + tcn_w * tcn_val_norm_arr
    test_ensemble = if_w * if_test_norm_arr + tcn_w * tcn_test_norm_arr

    val_auc = roc_auc_score(val_labels, val_ensemble)
    test_auc = roc_auc_score(test_labels, test_ensemble)

    results_rows.append(
        {
            "if_weight": if_w,
            "tcn_weight": tcn_w,
            "val_auc_roc": round(val_auc, 4),
            "test_auc_roc": round(test_auc, 4),
        }
    )

# ---------------------------------------------------------------------------
# Step 6 — Find optimal weight and display results
# ---------------------------------------------------------------------------

logger.info("=== Step 6: Results ===")

# Optimal = highest val AUC-ROC (NEVER tune on test set)
optimal = max(results_rows, key=lambda r: r["val_auc_roc"])
original = next(r for r in results_rows if r["if_weight"] == 0.05)

print("\n" + "=" * 62)
print("ENSEMBLE WEIGHT SWEEP — SMD — 7 MACHINES")
print("=" * 62)
print(
    f"{'IF Weight':>10}  {'TCN Weight':>10}  {'Val AUC-ROC':>12}  {'Test AUC-ROC':>13}"
)
print("-" * 62)
for r in results_rows:
    marker = ""
    if r["if_weight"] == optimal["if_weight"]:
        marker = "  ← OPTIMAL (val)"
    elif r["if_weight"] == 0.05:
        marker = "  ← original"
    print(
        f"  {r['if_weight']:>8.2f}  {r['tcn_weight']:>10.2f}"
        f"  {r['val_auc_roc']:>12.4f}  {r['test_auc_roc']:>13.4f}{marker}"
    )
print("=" * 62)

print(
    f"\nOriginal  (IF={original['if_weight']:.2f} / TCN={original['tcn_weight']:.2f}):"
)
print(f"  Val AUC-ROC:  {original['val_auc_roc']:.4f}")
print(f"  Test AUC-ROC: {original['test_auc_roc']:.4f}")

print(f"\nOptimal   (IF={optimal['if_weight']:.2f} / TCN={optimal['tcn_weight']:.2f}):")
print(f"  Val AUC-ROC:  {optimal['val_auc_roc']:.4f}")
print(f"  Test AUC-ROC: {optimal['test_auc_roc']:.4f}")

improvement = optimal["test_auc_roc"] - original["test_auc_roc"]
print(f"\nTest AUC-ROC improvement: {improvement:+.4f}")
print("=" * 62 + "\n")

# ---------------------------------------------------------------------------
# Step 7 — Save sweep results
# ---------------------------------------------------------------------------

sweep_out = {
    "dataset": "SMD",
    "machines": MACHINES,
    "weight_step": WEIGHT_STEP,
    "original_weights": {"if_weight": 0.05, "tcn_weight": 0.95},
    "optimal_weights": {
        "if_weight": optimal["if_weight"],
        "tcn_weight": optimal["tcn_weight"],
    },
    "optimal_val_auc_roc": optimal["val_auc_roc"],
    "optimal_test_auc_roc": optimal["test_auc_roc"],
    "original_test_auc_roc": original["test_auc_roc"],
    "improvement": round(improvement, 4),
    "all_results": results_rows,
}

sweep_path = ARTIFACTS_DIR / "day6b_weight_sweep.json"
with open(sweep_path, "w") as f:
    json.dump(sweep_out, f, indent=2)

logger.info("Sweep results saved to %s", sweep_path)
logger.info(
    "Optimal weight: IF=%.2f / TCN=%.2f → Test AUC-ROC=%.4f (was %.4f, Δ%+.4f)",
    optimal["if_weight"],
    optimal["tcn_weight"],
    optimal["test_auc_roc"],
    original["test_auc_roc"],
    improvement,
)

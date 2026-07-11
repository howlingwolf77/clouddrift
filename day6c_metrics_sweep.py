"""
Day 6c — Detailed Metrics Sweep.

Extends the weight sweep with precision, recall, F1, and F2 at each
weight combination. A contamination-based threshold is calibrated on
the validation set at each weight, then applied to both val and test.

Three weights are highlighted in the output:
    IF=0.05  original ensemble design
    IF=0.10  val AUC-ROC optimal
    IF=0.40  original CloudDrift architecture specification

Run from the project root:
    python day6c_metrics_sweep.py
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("day6c")

ARTIFACTS_DIR = Path("artifacts")
MACHINES = [f"machine-1-{i}" for i in range(1, 8)]
WEIGHT_STEP = 0.05
HIGHLIGHT_WEIGHTS = {0.05: "original", 0.10: "val-optimal", 0.40: "design-intent"}

# ---------------------------------------------------------------------------
# Steps 1-2 — Load artifacts and recreate normalized splits
# (identical to day6b — needed to recompute scores)
# ---------------------------------------------------------------------------

logger.info("=== Steps 1-2: Loading artifacts and recreating splits ===")

with open(ARTIFACTS_DIR / "day4_if_metrics.json") as f:
    day4_metrics = json.load(f)

feature_cols = day4_metrics["feature_cols"]

from src.features.engineering import load_feature_pipeline

pipeline = load_feature_pipeline()
normalizer = pipeline.named_steps["normalizer"]
for col in list(normalizer.bounds_.keys()):
    lo, hi = normalizer.bounds_[col]
    if np.isnan(lo) or np.isnan(hi) or np.isinf(lo) or np.isinf(hi):
        normalizer.bounds_[col] = (0.0, 1.0)
        logger.warning("Patched NaN bounds for '%s'", col)

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

for label, df in [("train", train_norm), ("val", val_norm), ("test", test_norm)]:
    n_nan = df[feature_cols].isna().sum().sum()
    if n_nan > 0:
        df[feature_cols] = df[feature_cols].fillna(0.0)
        logger.warning("%s_norm: %d NaN filled", label, n_nan)

val_labels = val_norm["is_anomaly"].values
test_labels = test_norm["is_anomaly"].values

# Contamination rate: val anomaly rate × 1.5 (bias toward recall)
val_contamination = float(val_labels.mean()) * 1.5
logger.info(
    "Val anomaly rate: %.2f%% | Contamination for threshold: %.2f%%",
    val_labels.mean() * 100,
    val_contamination * 100,
)

# ---------------------------------------------------------------------------
# Step 3 — Compute and normalize component scores (runs once)
# ---------------------------------------------------------------------------

logger.info("=== Step 3: Computing component scores ===")

from src.models.isolation_forest import compute_anomaly_scores, load_isolation_forest

if_model = load_isolation_forest()
x_train_normal = train_norm[~train_norm["is_anomaly"]][feature_cols].values
if_train_scores = compute_anomaly_scores(if_model, x_train_normal)
if_val_scores = compute_anomaly_scores(if_model, val_norm[feature_cols].values)
if_test_scores = compute_anomaly_scores(if_model, test_norm[feature_cols].values)

from src.models.tcn_autoencoder import (
    compute_reconstruction_errors,
    load_tcn_autoencoder,
)

tcn_model = load_tcn_autoencoder()
logger.info("Computing TCN errors (val)...")
tcn_val_errors = compute_reconstruction_errors(
    tcn_model, val_norm, feature_cols, group_col="source_file"
)
logger.info("Computing TCN errors (test)...")
tcn_test_errors = compute_reconstruction_errors(
    tcn_model, test_norm, feature_cols, group_col="source_file"
)
logger.info("Computing TCN errors (train normal)...")
tcn_train_errors = compute_reconstruction_errors(
    tcn_model,
    train_norm[~train_norm["is_anomaly"]],
    feature_cols,
    group_col="source_file",
)

tcn_train_scores = tcn_train_errors.fillna(0.0).values
tcn_val_scores = tcn_val_errors.fillna(0.0).values
tcn_test_scores = tcn_test_errors.fillna(0.0).values


# Normalize both to [0, 1] using training normal bounds
def _minmax_norm(scores, lo, hi):
    span = hi - lo
    if span == 0:
        return np.zeros_like(scores, dtype=float)
    return np.clip((scores - lo) / span, 0.0, 1.0)


if_lo, if_hi = float(if_train_scores.min()), float(if_train_scores.max())
tcn_lo, tcn_hi = float(tcn_train_scores.min()), float(tcn_train_scores.max())

if_val_n = _minmax_norm(if_val_scores, if_lo, if_hi)
if_test_n = _minmax_norm(if_test_scores, if_lo, if_hi)
tcn_val_n = _minmax_norm(tcn_val_scores, tcn_lo, tcn_hi)
tcn_test_n = _minmax_norm(tcn_test_scores, tcn_lo, tcn_hi)

logger.info("Scores computed and normalized. Beginning weight sweep...")

# ---------------------------------------------------------------------------
# Step 4 — Detailed metrics sweep
# ---------------------------------------------------------------------------

logger.info("=== Step 4: Detailed metrics sweep ===")

weights = np.round(np.arange(0.0, 1.0 + WEIGHT_STEP, WEIGHT_STEP), 2)
rows = []

for if_w in weights:
    tcn_w = round(1.0 - if_w, 2)

    val_ens = if_w * if_val_n + tcn_w * tcn_val_n
    test_ens = if_w * if_test_n + tcn_w * tcn_test_n

    # Threshold calibrated on val only
    pct = (1.0 - val_contamination) * 100
    threshold = float(np.percentile(val_ens, pct))

    val_pred = (val_ens >= threshold).astype(int)
    test_pred = (test_ens >= threshold).astype(int)

    rows.append(
        {
            "if_weight": if_w,
            "tcn_weight": tcn_w,
            "threshold": round(threshold, 4),
            "val_auc": round(roc_auc_score(val_labels, val_ens), 4),
            "val_p": round(precision_score(val_labels, val_pred, zero_division=0), 3),
            "val_r": round(recall_score(val_labels, val_pred, zero_division=0), 3),
            "val_f1": round(f1_score(val_labels, val_pred, zero_division=0), 3),
            "val_f2": round(
                fbeta_score(val_labels, val_pred, beta=2, zero_division=0), 3
            ),
            "test_auc": round(roc_auc_score(test_labels, test_ens), 4),
            "test_p": round(
                precision_score(test_labels, test_pred, zero_division=0), 3
            ),
            "test_r": round(recall_score(test_labels, test_pred, zero_division=0), 3),
            "test_f1": round(f1_score(test_labels, test_pred, zero_division=0), 3),
            "test_f2": round(
                fbeta_score(test_labels, test_pred, beta=2, zero_division=0), 3
            ),
        }
    )

# ---------------------------------------------------------------------------
# Step 5 — Display results
# ---------------------------------------------------------------------------

print("\n" + "=" * 100)
print("ENSEMBLE DETAILED METRICS SWEEP  —  SMD  —  7 MACHINES")
print(
    f"Threshold calibrated at {val_contamination * 100:.1f}% contamination on validation set"
)
print("=" * 100)
print(
    f"{'IF W':>5}  {'TCN W':>5}  "
    f"{'ValAUC':>7}  {'Val-P':>6}  {'Val-R':>6}  {'Val-F1':>7}  {'Val-F2':>7}  "
    f"{'TstAUC':>7}  {'Tst-P':>6}  {'Tst-R':>6}  {'Tst-F1':>7}  {'Tst-F2':>7}  "
    f"{'Note':<18}"
)
print("-" * 100)

for r in rows:
    note = HIGHLIGHT_WEIGHTS.get(r["if_weight"], "")
    p_flag = "✓" if r["test_p"] >= 0.70 else " "
    r_flag = "✓" if r["test_r"] >= 0.65 else " "
    print(
        f"  {r['if_weight']:>4.2f}  {r['tcn_weight']:>5.2f}  "
        f"  {r['val_auc']:>6.4f}  {r['val_p']:>6.3f}  {r['val_r']:>6.3f}  "
        f"{r['val_f1']:>7.3f}  {r['val_f2']:>7.3f}  "
        f"  {r['test_auc']:>6.4f}  {r['test_p']:>5.3f}{p_flag}  {r['test_r']:>5.3f}{r_flag}  "
        f"{r['test_f1']:>7.3f}  {r['test_f2']:>7.3f}  "
        f"{'← ' + note if note else ''}"
    )

print("=" * 100)
print(
    "P target ≥0.70 (✓), R target ≥0.65 (✓)  |  Threshold calibrated on val, applied to test"
)

# ---------------------------------------------------------------------------
# Spotlight: three key weights
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SPOTLIGHT: THREE KEY WEIGHTS")
print("=" * 60)
for if_w, label in [
    (0.05, "Original   (IF=0.05 / TCN=0.95)"),
    (0.10, "Val-Optimal(IF=0.10 / TCN=0.90)"),
    (0.40, "Design-Intent(IF=0.40 / TCN=0.60)"),
]:
    r = next(x for x in rows if x["if_weight"] == if_w)
    print(f"\n{label}")
    print(
        f"  Val  — AUC:{r['val_auc']:.4f}  P:{r['val_p']:.3f}  R:{r['val_r']:.3f}  F1:{r['val_f1']:.3f}  F2:{r['val_f2']:.3f}"
    )
    print(
        f"  Test — AUC:{r['test_auc']:.4f}  P:{r['test_p']:.3f}  R:{r['test_r']:.3f}  F1:{r['test_f1']:.3f}  F2:{r['test_f2']:.3f}"
    )
print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# Step 6 — Save
# ---------------------------------------------------------------------------

out = {
    "dataset": "SMD",
    "machines": MACHINES,
    "val_contamination": val_contamination,
    "weight_step": WEIGHT_STEP,
    "highlight_weights": HIGHLIGHT_WEIGHTS,
    "all_results": rows,
}
out_path = ARTIFACTS_DIR / "day6c_detailed_metrics.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
logger.info("Detailed metrics saved to %s", out_path)

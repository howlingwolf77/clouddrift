# CloudDrift Model Card

**Version:** 1.0.0
**Date:** July 2026
**Author:** Rainel (Ryan) Lobora
**Project:** Coding Macaw 2026 Advanced ML Bootcamp Capstone

---

## Model Description

CloudDrift is an ML-powered cloud infrastructure anomaly detector
that combines two complementary models in a weighted ensemble:

| Component | Architecture | Role |
|-----------|-------------|------|
| Isolation Forest | 100 trees, 13 rolling features | Point-wise feature-space isolation |
| TCN Autoencoder | 4-level dilated causal Conv1d, seq=30 | Temporal sequence reconstruction |
| Ensemble | IF×0.05 + TCN×0.95 | Combined score — beats both individually |

### Isolation Forest

Trained on 86,683 normal rows of NAB data using 13 engineered rolling
features: value mean/std/z-score at short/mid/long windows, rate of
change, range ratio, and raw value. Threshold calibrated at the 90th
percentile of validation scores (operational, not precision-recall
calibrated — see Limitations).

### TCN Autoencoder

Encoder: 4 stacked dilated causal Conv1d blocks with dilation factors
[1,2,4,8] and channel counts [32→32→16→8]. Decoder: symmetric
expansion [8→16→32→32→13]. Each block includes a LayerNorm + residual
connection. Trained exclusively on normal-behavior sequences
(86,103 sequences of length 30) using MSE reconstruction loss.
Early stopped at epoch 22 (val_loss 0.004→0.000). 15,252 parameters.

### Ensemble

Weights determined empirically via two-stage AUC-ROC weight scan on
the validation set:
- Coarse scan (step=0.1): IF weight=0.1 → AUC-ROC=0.843
- Fine scan (step=0.05): IF weight=0.05 → AUC-ROC=0.863

IF weight=0.05 exceeds both standalone models, confirming genuine
complementary signal at this weight.

---

## Training Data

### Primary: Numenta Anomaly Benchmark (NAB)

- **Source:** https://github.com/numenta/NAB
- **Size:** 137,301 rows, 24 independent time-series, 12,906 anomaly rows (9.4%)
- **Series categories:** realAWSCloudwatch (17), realKnownCause (7)
- **Temporal split:** Global temporal split at 70/15/15
  (train: 96,110 rows | val: 20,595 rows | test: 20,596 rows)
- **Val anomaly rate:** 1.1% (225 anomalies in 20,595 rows)
- **Note:** Per-series split was evaluated and reverted — it reduced
  IF AUC-ROC from 0.785 to 0.448 due to temporal non-stationarity
  in NAB series. See `docs/adr/ADR-001-isolation-forest.md`.

### Secondary: Alibaba Cluster Trace 2018

- **Source:** https://github.com/alibaba/clusterdata
- **Size:** 311,000 readings from 5 production machines, 8.4 GB raw
- **Role:** Feature engineering demonstration (60+ rolling features);
  API reference statistics for z-score attribution at inference time
- **Not used for model training:** No ground-truth anomaly labels
  available; used for data engineering and API design only

---

## Evaluation Results

### Validation Set (1.1% anomaly rate)

| Metric | IF | TCN | Ensemble |
|--------|-----|-----|----------|
| AUC-ROC | 0.785 | 0.843 | **0.863** |
| Precision | 0.003 | 0.166 | 0.139 |
| Recall | 0.004 | 0.249 | 0.209 |
| F1 | 0.004 | 0.199 | 0.167 |

### Test Set (15.8% anomaly rate)

| Metric | IF | TCN | Ensemble |
|--------|-----|-----|----------|
| AUC-ROC | 0.439 | 0.519 | 0.435 |

### Why Binary P/R Is Not the Primary Metric

The validation set's 1.1% anomaly rate (225 anomalies in 20,595 rows,
90:1 class imbalance) makes binary precision/recall mathematically
unreliable as a threshold-selection metric. Achieving precision=0.70
with recall=0.65 would require flagging fewer than 100 rows of which
70 are true anomalies — near-perfect score separation that does not
exist in heterogeneous multi-series monitoring data.

AUC-ROC is the primary evaluation metric because it is threshold-
independent and measures the model's fundamental discriminative
capability across all possible thresholds without being distorted by
class imbalance.

---

## Limitations and Known Issues

### 1. Temporal Non-Stationarity (Test AUC-ROC Gap)

The ensemble achieves AUC-ROC=0.863 on validation but ~0.43 on test.
The test period (Jul 2014–Jan 2015) contains anomaly signatures from
`realKnownCause` series (rogue_agent_key_hold,
ambient_temperature_system_failure) where anomaly events cluster in
the latter portion of each series. Models trained on earlier data do
not generalize to these late-period anomaly patterns.

**Implication:** In production, periodic retraining on recent
telemetry is required. CloudDrift's Evidently AI integration
(`dashboard/drift_monitor.py`) detects when inference inputs
have drifted from the training distribution, signaling when
retraining is warranted.

### 2. Single-Point Detection Mode

The `/detect` API endpoint uses z-score attribution (Track 1)
rather than the full IF+TCN ensemble. The TCN requires 30 sequential
timesteps for its sliding window context, which is not available from
a single telemetry snapshot. The IF could theoretically run on a
single snapshot, but it was trained on 13 NAB rolling features (not
raw telemetry metrics), making direct application to Alibaba-style
input incorrect.

**Implication:** For maximum detection capability, use `/batch_detect`
with at least 30 sequential snapshots from the same machine.

### 3. Training Distribution Assumptions

Models were trained on NAB AWS CloudWatch data from 2014. Production
cloud environments may have different baseline utilization patterns,
seasonal effects, and anomaly signatures. Z-score attribution uses
Alibaba Cluster Trace 2018 statistics as the reference distribution
for the inference API.

### 4. No Multi-Metric IF/TCN Training

The Isolation Forest and TCN Autoencoder were trained on NAB single-
metric data. Multi-metric ensemble training (using Alibaba's five-
metric schema) was evaluated but deferred: Alibaba has no ground-truth
anomaly labels, making model evaluation impossible.

---

## Explainability

Two-track design ensures both speed and rigor:

**Track 1 (production):** Z-score deviation ranking. For each
incoming telemetry snapshot, computes |value − training_mean| /
training_std per metric and returns the top N metrics ranked by
deviation. Runs in microseconds. Available in every `/detect`
response via `top_contributing_features` and `feature_deviation_scores`.

**Track 2 (evaluation):** SHAP TreeExplainer on the Isolation Forest.
`notebooks/06_shap_analysis.ipynb` produces waterfall charts for the
top 5 ensemble-flagged anomaly windows, providing mathematically exact
Shapley value decomposition. The Track 1 vs Track 2 comparison cell
in the notebook validates that the two methods substantially agree,
confirming Track 1 is trustworthy for production use.

---

## Responsible Use

- **Not a replacement for human judgment:** CloudDrift surfaces
  anomaly signals for investigation, not automatic remediation.
- **Context required:** A high anomaly score indicates statistical
  deviation from training data, not necessarily a production incident.
- **Retraining obligation:** Deploying without periodic retraining
  on current telemetry reduces detection quality over time.
- **Threshold tuning:** The operational threshold (90th percentile of
  validation scores) is a starting point, not a universal value.
  Tune for your environment's false positive tolerance.

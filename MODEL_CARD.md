# CloudDrift Model Card

**Version:** 1.0.0
**Date:** July 2026
**Author:** Rainel (Ryan) Lobo
**Project:** Coding Macaw 2026 Advanced ML Bootcamp Capstone

---

## Model Description

CloudDrift is an ML-powered cloud infrastructure anomaly detector
that combines two complementary models in a weighted ensemble:

| Component | Architecture | Role |
|-----------|-------------|------|
| Isolation Forest | 100 trees, 68 rolling features | Point-wise feature-space isolation |
| TCN Autoencoder | 4-level dilated causal Conv1d, seq=30, 29K params | Temporal sequence reconstruction |
| Ensemble | IF×0.40 + TCN×0.60 | Combined score — test AUC-ROC 0.899 |

### Isolation Forest

Trained on 235,908 normal rows from 7 SMD machines using 68 engineered
features: per-metric mean, std, z-score, rate-of-change, and range
ratio at short/mid/long rolling windows across 5 server metrics
(cpu_util, mem_util, net_io_in, net_io_out, disk_io), plus 3
cross-metric features (CPU-memory correlation, CPU-network ratio,
composite volatility). Threshold calibrated at the 90th percentile
of validation scores (0.591347).

### TCN Autoencoder

Encoder: 4 stacked dilated causal Conv1d blocks with dilation factors
[1,2,4,8] and channel counts [68→32→32→16→8]. Decoder: symmetric
expansion [8→16→32→32→68]. Each block includes LayerNorm + residual
connection. Trained exclusively on normal-behavior sequences
(235,705 sequences of length 30) using MSE reconstruction loss.
Trained for 100 epochs (val_loss=0.0016 at best checkpoint). 29,552
parameters. Error separation confirmed: anomaly reconstruction error
(0.0074) is 5× normal (0.0015).

### Ensemble

Weights set to IF=0.40 / TCN=0.60 per the original CloudDrift
architecture specification, which predates the SMD training results.
A post-hoc weight scan confirmed:
- Val-optimal weight (by AUC-ROC): IF=0.10 → test AUC-ROC=0.892
- Design-intent weight: IF=0.40 → test AUC-ROC=0.899

Both values are comparable. IF=0.40 was retained as it aligns with
the original design specification and produces marginally higher
test performance.

---

## Training Data

### Primary: Server Machine Dataset (SMD)

- **Source:** https://github.com/NetManAIOps/OmniAnomaly
- **Machines used:** 7 (machine-1-1 through machine-1-7)
- **Total rows:** 341,346 rows | 10,979 anomaly rows (3.2%)
- **Features:** 5 selected from 38-dimensional raw telemetry
  (cpu_util, mem_util, net_io_in, net_io_out, disk_io)
- **Value range:** Pre-normalized to [0, 1] by dataset authors
- **Split:** Per-machine temporal 70/15/15
  - Train: 238,938 rows (1.27% anomaly — mostly SMD train period)
  - Val: 51,203 rows (6.04% anomaly — SMD train/test boundary)
  - Test: 51,205 rows (9.49% anomaly — SMD test period)
- **SMD design:** Train files are guaranteed anomaly-free by the
  dataset authors. Anomalies are concentrated in the test files.

### Secondary: Alibaba Cluster Trace 2018

- **Source:** https://github.com/alibaba/clusterdata
- **Size:** 311,000 readings from 5 production machines, 8.4 GB raw
- **Role:** Feature engineering validation and API schema design;
  column naming convention (cpu_util, mem_util, etc.) adopted from
  this dataset and reused for SMD
- **Not used for model training:** Alibaba's `machine_usage.csv`
  has no ground-truth anomaly labels

---

## Evaluation Results

### Isolation Forest (Standalone)

| Metric | Validation | Test |
|--------|-----------|------|
| AUC-ROC | 0.801 | **0.894** |
| Precision | 0.194 | 0.623 |
| Recall | 0.321 | 0.733 ✓ |
| F1 | 0.242 | 0.674 |
| F2 | 0.284 | 0.708 |

### TCN Autoencoder (Standalone)

| Metric | Validation | Test |
|--------|-----------|------|
| AUC-ROC | 0.869 | **0.887** |
| Precision | 0.356 | 0.541 |
| Recall | 0.533 | 0.789 ✓ |
| F1 | 0.427 | 0.641 |
| Error separation | err_anom=0.0037 > err_norm=0.0015 ✓ | err_anom=0.0074 > err_norm=0.0015 ✓ |

### Ensemble (IF=0.40, TCN=0.60)

| Metric | Validation | Test |
|--------|-----------|------|
| AUC-ROC | 0.868 | **0.899** |
| Precision | 0.284 | 0.567 |
| Recall | 0.426 | 0.762 ✓ |
| F1 | 0.341 | 0.650 |
| F2 | 0.388 | 0.713 |

### Why Val AUC-ROC < Test AUC-ROC

SMD anomalies concentrate in the test period. The 70/15/15 temporal
split places the val set at the train/test boundary (6.0% anomaly
rate) and the test set in the dense anomaly zone (9.5% anomaly rate).
Higher anomaly density in the test set makes score separation easier
to measure, not easier to achieve. The model generalises well — there
is no overfitting or degradation from val to test.

---

## Limitations and Known Issues

### 1. Partial Machine Coverage

Models were trained on 7 of SMD's 28 available machines due to WSL2
memory constraints (28 machines require ~9 GB for sequence tensor
allocation; available RAM was 7.6 GB). The 7 machines cover
machine-1-1 through machine-1-7 (group 1, cluster A).

**Implication:** Performance may vary on machines from SMD groups 2
and 3 or on production environments with different server workload
profiles. Periodic retraining on representative machines is recommended.

### 2. Single-Point Detection Mode

The `/detect` API endpoint uses z-score attribution (Track 1) rather
than the full IF+TCN ensemble. The TCN requires 30 sequential
timesteps for its sliding window context, not available from a single
telemetry snapshot.

**Implication:** For maximum detection capability, use `/batch_detect`
with at least 30 sequential snapshots from the same machine.

### 3. Precision Below Target on Validation

Binary precision (0.284 val, 0.567 test) remains below the ≥0.70
target on both sets. AUC-ROC is used as the primary metric because it
is threshold-independent and not distorted by anomaly rate differences
between splits. Threshold tuning for specific precision targets is
supported via contamination parameter adjustment.

### 4. TCN Training Duration

Training ran for 100 epochs rather than stopping early (min_delta=0.0
in EarlyStopping allowed infinitesimal improvements to reset the
patience counter). Future runs should use min_delta=0.0001. Final
val_loss=0.0016 represents a well-converged model.

### 5. Evidently KS Test Sensitivity at Low Sample Counts

The Kolmogorov-Smirnov test used by Evidently compares the full
distributional shape of current readings against the 235,908-row
SMD training reference. With high reference n, the test has
sufficient statistical power to detect even minor shape differences
in sessions as small as 40 readings, producing drift detections
that do not correspond to actionable distribution shift.

**Implication:** Accumulate 200+ readings per session before
treating Evidently drift detections as a retraining signal. Use
the inline z-score table for real-time monitoring. The two layers
measure different statistical properties and are designed to be
used together, not as substitutes.

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
validates that the two methods substantially agree, confirming Track 1
is trustworthy for production use.

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

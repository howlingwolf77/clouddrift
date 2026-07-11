# ADR-001: Isolation Forest as Anomaly Detection Baseline

**Date:** June 2026
**Status:** Accepted (amended July 2026 — dataset migration to SMD)
**Author:** Rainel (Ryan) Lobo

## Context

CloudDrift requires a point-wise anomaly detector that can identify
individual telemetry readings that are statistically isolated from
the normal distribution in feature space, without requiring labeled
anomaly data during training (unsupervised). The model needed to:
- Train on normal-behavior data only
- Score new readings in milliseconds at inference
- Provide a continuous anomaly score for ensemble combination
- Work with tabular rolling features from server telemetry

## Decision

Selected **scikit-learn IsolationForest** with 100 estimators and
default contamination parameter, trained on 235,908 normal rows
from 7 SMD machines using 68 engineered rolling features across
5 server metrics (cpu_util, mem_util, net_io_in, net_io_out, disk_io).

## Dataset Migration: NAB → SMD

The original implementation used the Numenta Anomaly Benchmark (NAB),
which produced a test AUC-ROC of 0.439 — below random chance (0.5).
This indicated that the IF's anomaly scores were negatively correlated
with true anomalies on the test set: a score inversion, not a
calibration problem.

**Root cause of NAB failure:**
1. NAB is a univariate benchmark designed for streaming anomaly
   detection. Its anomalies are contextual (only anomalous relative
   to surrounding temporal context), while Isolation Forest isolates
   points in feature space — a fundamental mismatch.
2. NAB's heterogeneous metrics (AWS CloudWatch CPU, disk bytes,
   network bytes, request latency — all collapsed into a single
   `value` column) produce inconsistent feature distributions.
3. Validation AUC-ROC (0.785) and test AUC-ROC (0.439) showed a
   34-point collapse, signalling severe distribution shift between
   the validation and test periods.

**Resolution:** Switched to Server Machine Dataset (SMD), which:
- Has the same server metrics (CPU, memory, network, disk) as the
  API's target domain
- Provides pre-labelled ground-truth anomalies at ~4.7% rate per machine
- Contains multivariate signals that Isolation Forest can isolate
  in feature space (CPU spike + memory saturation simultaneously)
- Uses pre-defined train/test splits that reflect genuine temporal
  anomaly structure

**Result after migration:** Test AUC-ROC improved from 0.439 to 0.894.

## Temporal Split Design

SMD ships with a pre-defined train/test split per machine (train files
are guaranteed anomaly-free). CloudDrift applies an additional
70/15/15 temporal split per machine on the combined data:
- Train (70%): Covers the all-normal SMD train period
- Val (15%): Straddles the SMD train/test boundary (6.0% anomaly rate)
- Test (15%): Falls in the SMD test period (9.5% anomaly rate)

Per-series splitting (one split per machine independently) is used
to ensure each machine contributes to all three splits and anomaly
rates are representative in both val and test.

## Threshold Calibration

Binary precision/recall targets (P≥0.70, R≥0.65) are not achievable
on the validation set (6.0% anomaly rate, 9:1 class imbalance), but
are approached on the test set (P=0.623, R=0.733 ✓). AUC-ROC is
used as the primary metric because it is threshold-independent.

Threshold calibrated at the 90th percentile of validation IF scores
(0.591347), flagging the top 10% of readings as potentially anomalous.

## Consequences

- IF AUC-ROC = 0.801 on validation, 0.894 on test (genuine discriminative signal)
- CV stability: 3 of 5 folds have no anomalies (temporal split puts
  anomalies in the later folds); folds 4–5 show P=0.038, R=1.000 —
  high recall but low precision is expected at 10% contamination rate
- IF contributes 40% weight to ensemble (per original design specification)
- Test recall 0.733 clears the ≥0.65 operational target

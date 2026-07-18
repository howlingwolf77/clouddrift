# ADR-002: TCN Autoencoder Architecture

**Date:** June 2026
**Status:** Accepted (amended July 2026 — SMD migration, updated channel counts)
**Author:** Rainel (Ryan) Lobo

## Context

CloudDrift required a deep learning component that learns temporal
patterns from normal-behavior sequences and detects anomalies via
reconstruction error. Two candidate architectures were evaluated.

## Decision

Selected **TCN (Temporal Convolutional Network) Autoencoder** with
dilated causal convolutions over LSTM Autoencoder.

## Architecture

- Encoder: 4 stacked `CausalConv1dBlock` layers, dilations [1,2,4,8]
- Channels: 68→32→32→16→8 (encoder), 8→16→32→32→68 (decoder)
- Kernel size: 3, sequence length: 30 timesteps (30-minute look-back
  at 1-minute SMD sampling intervals)
- Each block: left-padded causal Conv1d + LayerNorm + ReLU + residual
- Total parameters: 29,552
- Training: PyTorch Lightning, MSE loss, Adam
- Training data: 235,705 normal-behavior sequences (is_anomaly=False)
- Trained for 100 epochs, best checkpoint at epoch 99 (val_loss=0.0016)

**Note on training duration:** EarlyStopping was configured with
patience=5 but min_delta=0.0. This caused the monitor to reset on
infinitesimal improvements each epoch and training ran to max_epochs.
Future runs should use min_delta=0.0001. The 100-epoch result is a
well-converged model (val_loss=0.0016 on normalized [0,1] features).

## Why TCN Over LSTM

| Property | TCN | LSTM |
|----------|-----|------|
| Parallelism | Full (conv operations) | Sequential (cell state) |
| Gradient flow | Stable (residual connections) | Vanishing gradient risk |
| Receptive field | Explicit (dilation schedule) | Implicit (hidden state) |
| Training speed | ~3× faster on CPU | Slower |
| Determinism | Fully deterministic | Stochastic (dropout) |

The dilated causal structure makes receptive field calculation
explicit: with kernel=3 and dilations [1,2,4,8], the combined
receptive field is 3+5+9+17=34 timesteps — larger than seq_length=30,
ensuring each output timestep attends to the full input sequence.

## Input Dimension Change (NAB → SMD)

The original NAB-based TCN used input_dim=13 (1 raw value column +
12 rolling features from `build_nab_features()`). After migrating
to SMD, `build_alibaba_features()` produces 68 features:
- 5 raw metric columns (cpu_util, mem_util, net_io_in, net_io_out, disk_io)
- 5 metrics × 12 rolling features = 60 rolling features
- 3 cross-metric features (cpu_mem_corr_long, cpu_net_ratio, volatility_score)

Channel counts updated from 13→32→...→13 to 68→32→...→68 accordingly.
The TCN architecture (4 blocks, same dilations, same kernel size) is
unchanged; only input_dim and output_dim changed.

## Sign Convention

`shap.TreeExplainer` uses sklearn's native sign convention (lower =
more anomalous). CloudDrift inverts this (higher = more anomalous) to
match the TCN reconstruction error convention. The SHAP notebook
applies sign-flip before any plotting. See `notebooks/06_shap_analysis.ipynb`.

## Contingency

The plan included LSTM fallback triggers (separation check failure).
The TCN passed the separation check on SMD:
- err_anomaly (test) = 0.0074
- err_normal (test) = 0.0015
- Separation ratio: 5× — strong signal, no fallback needed

## Consequences

- TCN AUC-ROC = 0.869 on validation, 0.887 on test
- Error separation confirmed on both validation and test sets
- Ensemble weight: 0.60 (TCN), 0.40 (IF) per original design specification
- 100-epoch training wall time: approximately 1.5 hours on CPU (WSL2)
- Recommendation for future runs: set min_delta=0.0001 in EarlyStopping
  to enable early stopping when improvement falls below meaningful threshold

## Amendment — Live Serving via /batch_detect (July 2026)

The TCN Autoencoder now runs in live production serving, not only in
training evaluation. When `/batch_detect` receives a `machine_id` group
of ≥ 30 sequential snapshots, `_score_group_ensemble()` in
`api/services/detection.py` calls `compute_reconstruction_errors()` on
the normalized 68-feature matrix built from the incoming batch.

TCN warm-up behavior at the serving layer: with exactly 30 snapshots,
only the last row receives a full reconstruction error (the first 29
rows cannot complete the seq_length=30 sliding window). Those rows
receive `NaN` filled with `0.0`, making them IF-dominant at IF=0.40
rather than the intended IF=0.40 / TCN=0.60 split. Sending 60+
snapshots gives most rows a proper TCN reconstruction score.
The `detection_mode` field in each result indicates which scoring
path ran: `"ensemble_if_tcn"` or `"single_point_zscore"`.

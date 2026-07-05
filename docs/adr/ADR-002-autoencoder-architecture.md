# ADR-002: TCN Autoencoder Architecture

**Date:** June 2026
**Status:** Accepted
**Author:** Rainel (Ryan) Lobora

## Context

CloudDrift required a deep learning component that learns temporal
patterns from normal-behavior sequences and detects anomalies via
reconstruction error. Two candidate architectures were evaluated.

## Decision

Selected **TCN (Temporal Convolutional Network) Autoencoder** with
dilated causal convolutions over LSTM Autoencoder.

## Architecture

- Encoder: 4 stacked `CausalConv1dBlock` layers, dilations [1,2,4,8]
- Channels: 13→32→32→16→8 (encoder), 8→16→32→32→13 (decoder)
- Kernel size: 3, sequence length: 30 timesteps (150-min look-back)
- Each block: left-padded causal Conv1d + LayerNorm + ReLU + residual
- Total parameters: 15,252
- Training: PyTorch Lightning, MSE loss, Adam, early stopping (patience=5)
- Training data: 86,103 normal-behavior sequences (is_anomaly=False)
- Trained to epoch 22, val_loss 0.004→0.000

## Why TCN Over LSTM

| Property | TCN | LSTM |
|----------|-----|------|
| Parallelism | Full (conv operations) | Sequential (cell state) |
| Gradient flow | Stable (residual connections) | Vanishing gradient risk |
| Receptive field | Explicit (dilation schedule) | Implicit (hidden state) |
| Training speed | ~3× faster on CPU | Slower |
| Determinism | Fully deterministic | Stochastic (dropout) |

The dilated causal structure also makes receptive field calculation
explicit: with kernel=3 and dilations [1,2,4,8], the combined
receptive field is 3+5+9+17=34 timesteps — larger than seq_length=30,
ensuring each output timestep attends to the full input sequence.

## Sign Convention

`shap.TreeExplainer` uses sklearn's native sign convention (lower =
more anomalous). CloudDrift inverts this (higher = more anomalous) to
match the TCN reconstruction error convention. The SHAP notebook
applies sign-flip before any plotting. See `notebooks/06_shap_analysis.ipynb`.

## Contingency

The plan included LSTM fallback triggers (separation check failure).
The TCN passed the separation check: err_anomaly (0.000031) >
err_normal (0.000021), confirming the model learned to reconstruct
normal temporal patterns and produces higher error on anomalous
sequences. No fallback was needed.

## Consequences

- TCN AUC-ROC = 0.843 on validation (better than IF's 0.785)
- 72.4% of anomaly sequences score above normal p90 (separation confirmed)
- Ensemble weight: 0.95 (empirically determined via two-stage scan)

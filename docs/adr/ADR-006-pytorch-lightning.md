# ADR-006: PyTorch Lightning for TCN Training

**Date:** June 2026
**Status:** Accepted
**Author:** Rainel (Ryan) Lobora

## Context

The TCN Autoencoder required a training loop with: early stopping,
checkpoint saving (best val_loss), MLflow metric logging, and
reproducible training on CPU. A raw PyTorch training loop was the
original plan; PyTorch Lightning was evaluated as an alternative.

## Decision

Selected **PyTorch Lightning** (`lightning` package) over raw PyTorch
training loop.

## Rationale

| Feature | PyTorch Lightning | Raw PyTorch |
|---------|-------------------|------------|
| Early stopping | `EarlyStopping` callback (2 lines) | Manual implementation |
| Best model saving | `ModelCheckpoint` callback (3 lines) | Manual implementation |
| MLflow logging | `MLFlowLogger` (4 lines) | Manual integration |
| Reproducibility | `deterministic=True` Trainer flag | Manual seed management |
| Progress bar | Built-in | Manual tqdm |
| Boilerplate | Minimal | ~100 extra lines |

Lightning's `LightningModule` separates model architecture
(`__init__`, `forward`) from training logic (`training_step`,
`validation_step`, `configure_optimizers`), making the code easier
to read and the architecture easier to swap (e.g., LSTM fallback
would only require replacing the encoder/decoder blocks).

## MLflow Backend

MLflow was configured to use SQLite backend (`sqlite:///mlflow.db`)
rather than the file-based `mlruns/` store. MLflow 3.14.0 deprecated
the file store and will raise an error without `MLFLOW_ALLOW_FILE_STORE=true`.
SQLite backend provides the full MLflow feature set without a separate
server.

## Consequences

- TCN training loop: 50 lines (including callbacks) vs ~150 estimated
  for raw PyTorch
- Best checkpoint auto-loaded after training completes
- Training reproducible via `torch.manual_seed(42)` + `deterministic=True`
- MLflow experiment `clouddrift-tcn` logged to `mlflow.db`

"""
TCN Autoencoder for CloudDrift anomaly detection.

Architecture:
    Temporal Convolutional Network (TCN) Autoencoder implemented as a
    PyTorch LightningModule.

    Encoder: 4 stacked dilated causal Conv1d blocks (dilation 1, 2, 4, 8)
    that compress 13 feature channels to an 8-channel bottleneck while
    preserving the 30-timestep temporal dimension.

    Decoder: 4 symmetric blocks that expand from 8 back to 13 channels.

    Each block includes a residual connection (shortcut addition) to
    prevent vanishing gradients through the depth of the network.

Training strategy:
    Trains exclusively on normal-behavior sequences (is_anomaly=False).
    The model learns to reconstruct normal temporal patterns efficiently.
    At inference time, anomalous sequences produce higher reconstruction
    error (MSE) because the model cannot apply the patterns it learned
    from normal data.

Score convention:
    Higher reconstruction error = more anomalous.
    Consistent with the IF score convention from Day 4.

Reference:
    Bai et al., "An Empirical Evaluation of Generic Convolutional and
    Recurrent Networks for Sequence Modeling" (2018).
    https://github.com/locuslab/TCN
"""

import logging
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQ_LENGTH = 30  # timesteps per sequence (150-min at 5-min intervals)
KERNEL_SIZE = 3  # Conv1d kernel size
NUM_LEVELS = 4  # stacked dilated blocks in encoder/decoder
ENCODER_CHANNELS = [32, 32, 16, 8]  # channel counts per encoder level
DILATIONS = [1, 2, 4, 8]  # dilation factor per level
LEARNING_RATE = 1e-3  # Adam optimizer initial learning rate
BATCH_SIZE = 256  # sequences per training batch
MAX_EPOCHS = 100  # training epoch ceiling
EARLY_STOPPING_PATIENCE = 5  # stop after N epochs without val_loss improvement
RANDOM_SEED = 42  # reproducibility
MLFLOW_EXPERIMENT = "clouddrift-tcn"  # MLflow experiment name


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """
    Sliding window sequence dataset for time-series autoencoder training.

    Creates all valid sliding windows of length seq_length from the input
    DataFrame, grouped by series (source_file) to prevent windows from
    crossing independent series boundaries.

    Args:
        df:           Feature DataFrame (from nab_train/val/test_features.parquet).
        feature_cols: Engineered feature column names.
        seq_length:   Number of timesteps per sequence.
        normal_only:  If True, exclude is_anomaly=True rows before windowing.
                      Use True for training, False for evaluation.
        group_col:    Column identifying independent time-series.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_length: int = SEQ_LENGTH,
        normal_only: bool = False,
        group_col: str = "source_file",
    ):
        self.seq_length = seq_length
        self.feature_cols = feature_cols

        if normal_only:
            df = df[~df["is_anomaly"]].copy()

        sequences: list[np.ndarray] = []
        labels: list[bool] = []

        for _, group in df.groupby(group_col):
            group = group.sort_values("timestamp")
            x = group[feature_cols].values.astype(np.float32)
            y = group["is_anomaly"].values.astype(bool)
            n = len(x)

            if n < seq_length:
                # Series too short for even one window — skip
                logger.debug(
                    "Series with %d rows < seq_length=%d — skipping", n, seq_length
                )
                continue

            for i in range(n - seq_length + 1):
                sequences.append(x[i : i + seq_length])
                labels.append(bool(y[i + seq_length - 1]))

        if not sequences:
            raise ValueError(
                f"No sequences created — all series shorter than seq_length={seq_length}. "
                "Check that the feature matrix contains data."
            )

        self.sequences = torch.FloatTensor(np.array(sequences))
        self.labels = torch.BoolTensor(np.array(labels))

        logger.info(
            "SequenceDataset: %s sequences, seq_length=%d, features=%d, "
            "anomaly_rate=%.1f%%",
            f"{len(self.sequences):,}",
            seq_length,
            len(feature_cols),
            float(self.labels.float().mean()) * 100,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# TCN building block
# ---------------------------------------------------------------------------


class CausalConv1dBlock(nn.Module):
    """
    Dilated causal Conv1d block with residual connection.

    Causality is enforced by left-padding the input with (kernel_size-1)*dilation
    zeros so each output timestep only attends to current and past inputs.

    The residual connection adds the block input to its output. A 1×1
    conv is applied to the residual when the channel counts differ.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        kernel_size:  Convolution kernel size.
        dilation:     Dilation factor for the convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            [batch, out_channels, seq_len]
        """
        # Left-pad for causal convolution
        padded = F.pad(x, (self.causal_pad, 0))
        out = self.conv(padded)
        # LayerNorm expects [batch, seq_len, channels] — transpose, norm, transpose
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        return out + self.residual_proj(x)


# ---------------------------------------------------------------------------
# TCN Autoencoder — LightningModule
# ---------------------------------------------------------------------------


class TCNAutoencoder(L.LightningModule):
    """
    TCN Autoencoder implemented as a PyTorch LightningModule.

    Encoder compresses input_dim feature channels to an 8-channel bottleneck
    via 4 stacked dilated causal Conv1d blocks. The decoder mirrors the
    encoder symmetrically to reconstruct the original feature channels.

    The temporal dimension (seq_length) is preserved throughout — the
    "compression" is in the channel (feature) dimension, not the time dimension.

    Loss: MSE between input and reconstruction.
    Anomaly score: MSE over the sequence (higher = more anomalous).

    Args:
        input_dim:     Number of input features (default: 13 for NAB).
        seq_length:    Sequence length (default: 30 timesteps).
        kernel_size:   Conv1d kernel size (default: 3).
        learning_rate: Adam optimizer learning rate (default: 1e-3).
    """

    def __init__(
        self,
        input_dim: int = 13,
        seq_length: int = SEQ_LENGTH,
        kernel_size: int = KERNEL_SIZE,
        learning_rate: float = LEARNING_RATE,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder: input_dim → bottleneck (8 channels)
        encoder_in = [input_dim] + ENCODER_CHANNELS[:-1]
        self.encoder = nn.Sequential(
            *[
                CausalConv1dBlock(in_ch, out_ch, kernel_size, dilation)
                for in_ch, out_ch, dilation in zip(
                    encoder_in, ENCODER_CHANNELS, DILATIONS
                )
            ]
        )

        # Decoder: bottleneck → input_dim (symmetric, reversed dilations)
        decoder_channels = list(reversed(ENCODER_CHANNELS[:-1])) + [input_dim]
        decoder_in = [ENCODER_CHANNELS[-1]] + decoder_channels[:-1]
        self.decoder = nn.Sequential(
            *[
                CausalConv1dBlock(in_ch, out_ch, kernel_size, dilation)
                for in_ch, out_ch, dilation in zip(
                    decoder_in, decoder_channels, reversed(DILATIONS)
                )
            ]
        )

        # Track best val loss for logging
        self._best_val_loss: float = float("inf")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            x_recon: [batch, seq_len, input_dim]
        """
        # Conv1d expects [batch, channels, seq_len]
        x = x.transpose(1, 2)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.transpose(1, 2)

    def _reconstruction_loss(self, batch: tuple) -> torch.Tensor:
        x, _ = batch
        x_recon = self(x)
        return F.mse_loss(x_recon, x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self._reconstruction_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self._reconstruction_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# ---------------------------------------------------------------------------
# Public interface — training
# ---------------------------------------------------------------------------


def create_sequence_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_length: int = SEQ_LENGTH,
    normal_only: bool = False,
) -> SequenceDataset:
    """
    Create a SequenceDataset from a feature DataFrame.

    Args:
        df:           Feature DataFrame (nab_train/val/test_features.parquet).
        feature_cols: Engineered feature column names.
        seq_length:   Number of timesteps per sequence.
        normal_only:  True for training (exclude anomaly rows);
                      False for evaluation (include all rows).

    Returns:
        SequenceDataset ready for DataLoader.
    """
    return SequenceDataset(
        df=df,
        feature_cols=feature_cols,
        seq_length=seq_length,
        normal_only=normal_only,
    )


def train_tcn_autoencoder(
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset,
    input_dim: int,
    checkpoint_dir: str | Path = ARTIFACTS_DIR / "checkpoints",
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> tuple["TCNAutoencoder", str]:
    """
    Train the TCN Autoencoder using PyTorch Lightning.

    Uses three callbacks:
        ModelCheckpoint: saves the best model by val_loss automatically.
        EarlyStopping:   stops training when val_loss stops improving.
        MLFlowLogger:    logs training/validation loss to MLflow experiment.

    Args:
        train_dataset: SequenceDataset with normal-only training sequences.
        val_dataset:   SequenceDataset with all validation sequences.
        input_dim:     Number of input features (e.g. 13 for NAB).
        checkpoint_dir: Directory to save Lightning checkpoints.
        max_epochs:    Maximum training epochs.
        batch_size:    Sequences per training batch.
        learning_rate: Adam learning rate.
        patience:      EarlyStopping patience (epochs without improvement).

    Returns:
        Tuple of (best_model, best_checkpoint_path).
    """
    torch.manual_seed(RANDOM_SEED)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # num_workers>0 can cause issues on WSL2
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = TCNAutoencoder(
        input_dim=input_dim,
        seq_length=SEQ_LENGTH,
        kernel_size=KERNEL_SIZE,
        learning_rate=learning_rate,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "TCN Autoencoder: %s parameters | encoder_channels=%s | dilations=%s",
        f"{n_params:,}",
        ENCODER_CHANNELS,
        DILATIONS,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="tcn_best_{epoch:02d}_{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=False,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            verbose=True,
        ),
    ]

    mlf_logger = MLFlowLogger(
        experiment_name=MLFLOW_EXPERIMENT,
        tracking_uri="sqlite:///mlflow.db",
        log_model=False,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=mlf_logger,
        accelerator="cpu",
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=True,
    )

    logger.info(
        "Starting training: max_epochs=%d, batch_size=%d, "
        "patience=%d, train_sequences=%s, val_sequences=%s",
        max_epochs,
        batch_size,
        patience,
        f"{len(train_dataset):,}",
        f"{len(val_dataset):,}",
    )

    trainer.fit(model, train_loader, val_loader)

    best_ckpt = callbacks[0].best_model_path
    logger.info("Training complete. Best checkpoint: %s", best_ckpt)

    # Load the best checkpoint weights into a fresh model
    best_model = TCNAutoencoder.load_from_checkpoint(
        best_ckpt,
        input_dim=input_dim,
    )
    best_model.eval()

    return best_model, best_ckpt


# ---------------------------------------------------------------------------
# Public interface — inference and evaluation
# ---------------------------------------------------------------------------


def compute_reconstruction_errors(
    model: "TCNAutoencoder",
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_length: int = SEQ_LENGTH,
    batch_size: int = BATCH_SIZE,
    group_col: str = "source_file",
) -> pd.Series:
    """
    Compute per-row reconstruction error for all rows in a feature DataFrame.

    For each series in df:
        - Creates all sliding windows of length seq_length
        - Runs the model on each window to compute MSE reconstruction error
        - Assigns the window error to the LAST timestep in that window
        - The first (seq_length-1) rows in each series receive the error
          of the first complete window (no look-ahead contamination)

    Args:
        model:        Fitted TCNAutoencoder (in eval mode).
        df:           Feature DataFrame to score (all rows, not just normal).
        feature_cols: Feature column names.
        seq_length:   Window size.
        batch_size:   Inference batch size.
        group_col:    Column identifying independent time-series.

    Returns:
        pd.Series with the same index as df, containing reconstruction
        error per row. Higher error = more anomalous.
    """
    model.eval()
    errors = pd.Series(np.nan, index=df.index, dtype=np.float64)

    for _, group in df.groupby(group_col):
        group = group.sort_values("timestamp")
        x_np = group[feature_cols].values.astype(np.float32)
        n = len(x_np)

        if n < seq_length:
            continue

        # Stack all windows: [n_windows, seq_len, n_features]
        windows = np.stack(
            [x_np[i : i + seq_length] for i in range(n - seq_length + 1)]
        )
        windows_tensor = torch.FloatTensor(windows)

        # Compute MSE per window in batches
        window_errors: list[float] = []
        with torch.no_grad():
            for start in range(0, len(windows_tensor), batch_size):
                batch = windows_tensor[start : start + batch_size]
                recon = model(batch)
                mse = ((batch - recon) ** 2).mean(dim=(1, 2))
                window_errors.extend(mse.cpu().numpy().tolist())

        window_errors_arr = np.array(window_errors)

        # Assign errors to rows: window i → row i + seq_length - 1
        # Rows 0 .. seq_length-2 get the first window's error (warm-up)
        row_errors = np.empty(n)
        row_errors[: seq_length - 1] = window_errors_arr[0]
        row_errors[seq_length - 1 :] = window_errors_arr

        errors.loc[group.index] = row_errors

    remaining_nans = errors.isna().sum()
    if remaining_nans > 0:
        logger.warning(
            "%d rows have NaN reconstruction error (series shorter than seq_length=%d)",
            remaining_nans,
            seq_length,
        )
    return errors


def calibrate_autoencoder_threshold(
    model: "TCNAutoencoder",
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> float:
    """
    Set the reconstruction error threshold using a contamination-based
    approach on the validation score distribution.

    Uses 1.5 × the observed validation anomaly rate as the contamination
    fraction. This flags slightly more rows than the true anomaly rate,
    biasing toward recall over precision — appropriate for anomaly detection.

    Why not training-distribution percentile:
        The TCN's reconstruction errors on training data can be an order of
        magnitude different from validation errors, because the model's
        internal scale adapts during training. Using training-derived
        thresholds on validation data produces a threshold that sits above
        all validation scores, flagging nothing.

    Args:
        model:        Fitted TCNAutoencoder.
        val_df:       Validation feature DataFrame (all rows, normal + anomaly).
        feature_cols: Feature column names.

    Returns:
        Float threshold value.
    """
    val_errors = compute_reconstruction_errors(model, val_df, feature_cols)
    val_anomaly_rate = float(val_df["is_anomaly"].mean())
    contamination = min(max(val_anomaly_rate * 1.5, 0.01), 0.10)
    threshold = float(np.nanpercentile(val_errors.values, (1 - contamination) * 100))

    flagged = int((val_errors.fillna(0) >= threshold).sum())
    logger.info(
        "AE threshold (val contamination=%.1f%%): %.6f "
        "→ flags %d val rows (%.1f%%) "
        "(val error range: [%.6f, %.6f])",
        contamination * 100,
        threshold,
        flagged,
        flagged / len(val_df) * 100,
        float(np.nanmin(val_errors.values)),
        float(np.nanmax(val_errors.values)),
    )
    return threshold


def evaluate_autoencoder(
    model: "TCNAutoencoder",
    threshold: float,
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str = "split",
) -> dict:
    """
    Evaluate the TCN Autoencoder on a labelled split.

    Computes per-row reconstruction errors, applies the threshold to produce
    binary predictions, and reports precision, recall, F1, F2, and AUC-ROC.

    Args:
        model:        Fitted TCNAutoencoder.
        threshold:    Calibrated reconstruction error threshold.
        df:           Feature DataFrame (all rows, normal + anomaly).
        feature_cols: Feature column names.
        split_name:   Label for logging.

    Returns:
        Metrics dict.
    """
    from sklearn.metrics import (
        f1_score,
        fbeta_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    errors = compute_reconstruction_errors(model, df, feature_cols)
    y_true = df["is_anomaly"].astype(int).values
    scores = errors.fillna(0.0).values
    y_pred = (scores >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

    try:
        auc_roc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc_roc = float("nan")

    metrics = {
        "split": split_name,
        "n_rows": int(len(y_true)),
        "n_anomaly_true": int(y_true.sum()),
        "n_anomaly_predicted": int(y_pred.sum()),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "auc_roc": auc_roc,
        "error_mean_normal": float(np.nanmean(scores[y_true == 0])),
        "error_mean_anomaly": float(np.nanmean(scores[y_true == 1]))
        if y_true.sum() > 0
        else float("nan"),
    }

    separation = (
        metrics["error_mean_anomaly"] > metrics["error_mean_normal"]
        if not np.isnan(metrics["error_mean_anomaly"])
        else False
    )

    logger.info(
        "%s | P=%.3f R=%.3f F1=%.3f F2=%.3f AUC-ROC=%.3f | "
        "err_normal=%.4f err_anomaly=%.4f separation=%s",
        split_name.upper(),
        precision,
        recall,
        f1,
        f2,
        auc_roc if not np.isnan(auc_roc) else 0,
        metrics["error_mean_normal"],
        metrics["error_mean_anomaly"]
        if not np.isnan(metrics["error_mean_anomaly"])
        else 0,
        "✓" if separation else "✗",
    )

    if not separation:
        logger.warning(
            "Reconstruction error NOT separated: anomaly mean (%.4f) ≤ "
            "normal mean (%.4f). Consider dropping to LSTM fallback.",
            metrics["error_mean_anomaly"]
            if not np.isnan(metrics["error_mean_anomaly"])
            else 0,
            metrics["error_mean_normal"],
        )

    return metrics


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def save_tcn_autoencoder(
    model: "TCNAutoencoder",
    path: str | Path = ARTIFACTS_DIR / "tcn_autoencoder.pt",
) -> None:
    """
    Save TCN Autoencoder weights to disk in PyTorch format (.pt).

    Saves the state dict and hyperparameters together so the model
    can be reconstructed without the original class definition.

    Args:
        model: Fitted TCNAutoencoder.
        path:  Destination path. Default: artifacts/tcn_autoencoder.pt.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hparams": dict(model.hparams),
        },
        path,
    )
    size_kb = path.stat().st_size / 1024
    logger.info("TCN Autoencoder saved: %s (%.1f KB)", path, size_kb)


def load_tcn_autoencoder(
    path: str | Path = ARTIFACTS_DIR / "tcn_autoencoder.pt",
) -> "TCNAutoencoder":
    """
    Load a previously saved TCN Autoencoder from disk.

    Args:
        path: Path to the saved .pt artifact.

    Returns:
        Fitted TCNAutoencoder in eval mode.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"TCN Autoencoder artifact not found: {path}. "
            "Run the Day 5 training pipeline first."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = TCNAutoencoder(**checkpoint["hparams"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logger.info("TCN Autoencoder loaded from: %s", path)
    return model

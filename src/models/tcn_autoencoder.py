"""
TCN Autoencoder implemented as a PyTorch LightningModule.
Primary path: TCN Autoencoder.
Fallback A: LSTM Autoencoder as LightningModule.
Fallback B: Raw PyTorch LSTM Autoencoder.
Contingency decision gate: start of Day 5.
Implemented: Day 5
"""

import lightning as L


class TCNAutoencoder(L.LightningModule):
    """
    Temporal Convolutional Network Autoencoder for anomaly detection.
    Encoder: stacked dilated causal Conv1d blocks (dilation 1, 2, 4, 8).
    Decoder: mirrored transposed dilated convolutions.
    """

    def __init__(
        self,
        input_dim: int = 5,
        sequence_length: int = 30,
        kernel_size: int = 3,
        num_levels: int = 4,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        raise NotImplementedError("Implemented Day 5")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Implemented Day 5")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Implemented Day 5")

    def configure_optimizers(self):
        raise NotImplementedError("Implemented Day 5")

"""
Day 5 tests: TCN Autoencoder architecture, sequence dataset,
reconstruction error computation, threshold calibration, artifact I/O.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.tcn_autoencoder import (
    ENCODER_CHANNELS,
    CausalConv1dBlock,
    SequenceDataset,
    TCNAutoencoder,
    compute_reconstruction_errors,
    load_tcn_autoencoder,
    save_tcn_autoencoder,
)

ARTIFACT_EXISTS = Path("artifacts/tcn_autoencoder.pt").exists()
FEATURES_EXIST = Path("data/processed/nab_val_features.parquet").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_df(
    n: int = 200,
    n_features: int = 5,
    n_series: int = 3,
    anomaly_rate: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Synthetic feature DataFrame for testing."""
    rng = np.random.default_rng(seed)
    feature_cols = [f"feat_{i}" for i in range(n_features)]
    dfs = []
    for s in range(n_series):
        timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
        is_anomaly = rng.random(n) < anomaly_rate
        data = {
            "timestamp": timestamps,
            "source_file": f"test/series_{s}.csv",
            "is_anomaly": is_anomaly,
        }
        for col in feature_cols:
            base = rng.uniform(0, 1, n)
            data[col] = (base + 0.5 * is_anomaly).astype(float)
        dfs.append(pd.DataFrame(data))
    return pd.concat(dfs, ignore_index=True), feature_cols


def _make_tiny_model(input_dim: int = 5) -> TCNAutoencoder:
    """Tiny model for fast unit tests."""
    return TCNAutoencoder(input_dim=input_dim, seq_length=10)


# ---------------------------------------------------------------------------
# CausalConv1dBlock tests
# ---------------------------------------------------------------------------


class TestCausalConv1dBlock:
    """Tests for the dilated causal convolution block."""

    def test_output_shape_unchanged(self):
        block = CausalConv1dBlock(8, 16, kernel_size=3, dilation=2)
        x = torch.randn(4, 8, 30)  # [batch, channels, seq_len]
        out = block(x)
        assert out.shape == (4, 16, 30)

    def test_channel_expansion(self):
        block = CausalConv1dBlock(3, 32, kernel_size=3, dilation=1)
        x = torch.randn(2, 3, 20)
        out = block(x)
        assert out.shape == (2, 32, 20)

    def test_same_channel_count(self):
        block = CausalConv1dBlock(16, 16, kernel_size=3, dilation=4)
        x = torch.randn(2, 16, 30)
        out = block(x)
        assert out.shape == (2, 16, 30)

    def test_causality_no_future_leakage(self):
        """Zeroing out future timesteps should not change past outputs."""
        block = CausalConv1dBlock(4, 4, kernel_size=3, dilation=1)
        block.eval()
        x = torch.randn(1, 4, 20)
        x_modified = x.clone()
        x_modified[:, :, 10:] = 0.0  # zero out future
        with torch.no_grad():
            out_orig = block(x)
            out_modified = block(x_modified)
        # Past outputs (timesteps 0..9) should be identical
        assert torch.allclose(out_orig[:, :, :10], out_modified[:, :, :10], atol=1e-5)


# ---------------------------------------------------------------------------
# SequenceDataset tests
# ---------------------------------------------------------------------------


class TestSequenceDataset:
    """Tests for the sliding window sequence dataset."""

    def test_sequence_shape(self):
        df, fc = _make_feature_df(n=50, n_features=5, n_series=2)
        ds = SequenceDataset(df, fc, seq_length=10)
        x, y = ds[0]
        assert x.shape == (10, 5)
        assert y.shape == ()

    def test_length_correct(self):
        n, seq_len, n_series = 50, 10, 2
        df, fc = _make_feature_df(n=n, n_features=5, n_series=n_series)
        ds = SequenceDataset(df, fc, seq_length=seq_len)
        expected = n_series * (n - seq_len + 1)
        assert len(ds) == expected

    def test_normal_only_excludes_anomalies(self):
        df, fc = _make_feature_df(n=100, n_features=5, n_series=2, anomaly_rate=0.5)
        ds = SequenceDataset(df, fc, seq_length=10, normal_only=True)
        # All sequences must end on a non-anomaly row
        assert not ds.labels.any()

    def test_skips_short_series(self):
        """A short series is silently skipped; valid series still contribute."""
        # One long series (50 rows → 41 sequences) + one short series (5 rows → skipped)
        long_df, fc = _make_feature_df(n=50, n_features=5, n_series=1)
        short_df, _ = _make_feature_df(n=5, n_features=5, n_series=1, seed=99)
        short_df["source_file"] = "test/short_series.csv"
        df = pd.concat([long_df, short_df], ignore_index=True)
        ds = SequenceDataset(df, fc, seq_length=10)
        # Only the 50-row series contributes: 50 - 10 + 1 = 41 sequences
        assert len(ds) == 41

    def test_raises_when_all_series_too_short(self):
        df, fc = _make_feature_df(n=3, n_features=5, n_series=2)
        with pytest.raises(ValueError, match="No sequences"):
            SequenceDataset(df, fc, seq_length=10)

    def test_sequences_are_float_tensors(self):
        df, fc = _make_feature_df(n=50, n_features=5, n_series=1)
        ds = SequenceDataset(df, fc, seq_length=10)
        x, _ = ds[0]
        assert x.dtype == torch.float32

    def test_no_series_boundary_crossing(self):
        """Each sequence must come from a single series."""
        n, seq_len, n_series = 30, 10, 3
        df, fc = _make_feature_df(n=n, n_features=5, n_series=n_series)
        ds = SequenceDataset(df, fc, seq_length=seq_len)
        # Total sequences = n_series * (n - seq_len + 1) = 3 * 21 = 63
        assert len(ds) == n_series * (n - seq_len + 1)


# ---------------------------------------------------------------------------
# TCNAutoencoder architecture tests
# ---------------------------------------------------------------------------


class TestTCNAutoencoderArchitecture:
    """Tests for the LightningModule architecture."""

    def test_forward_output_shape_matches_input(self):
        model = _make_tiny_model(input_dim=5)
        x = torch.randn(4, 10, 5)  # [batch, seq_len, features]
        out = model(x)
        assert out.shape == x.shape

    def test_reconstruction_is_different_from_input(self):
        """Untrained model should produce different output from input."""
        model = _make_tiny_model(input_dim=5)
        x = torch.randn(2, 10, 5)
        out = model(x)
        assert not torch.allclose(x, out)

    def test_encoder_has_correct_number_of_levels(self):
        model = TCNAutoencoder(input_dim=13)
        assert len(list(model.encoder.children())) == 4

    def test_decoder_has_correct_number_of_levels(self):
        model = TCNAutoencoder(input_dim=13)
        assert len(list(model.decoder.children())) == 4

    def test_bottleneck_channels(self):
        """Output of encoder should have ENCODER_CHANNELS[-1] channels."""
        model = TCNAutoencoder(input_dim=13)
        x = torch.randn(2, 13, 30)  # [batch, channels, seq_len]
        encoded = model.encoder(x)
        assert encoded.shape[1] == ENCODER_CHANNELS[-1]  # 8

    def test_hyperparameters_saved(self):
        model = TCNAutoencoder(input_dim=7, seq_length=20)
        assert model.hparams.input_dim == 7
        assert model.hparams.seq_length == 20

    def test_mse_loss_non_negative(self):
        model = _make_tiny_model(input_dim=5)
        x = torch.randn(4, 10, 5)
        recon = model(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        assert loss.detach().item() >= 0


# ---------------------------------------------------------------------------
# Reconstruction error tests
# ---------------------------------------------------------------------------


class TestComputeReconstructionErrors:
    """Tests for compute_reconstruction_errors()."""

    def test_returns_series_same_length_as_input(self):
        df, fc = _make_feature_df(n=50, n_features=5, n_series=2)
        model = _make_tiny_model(input_dim=5)
        errors = compute_reconstruction_errors(model, df, fc, seq_length=10)
        assert len(errors) == len(df)

    def test_errors_are_non_negative(self):
        df, fc = _make_feature_df(n=50, n_features=5, n_series=2)
        model = _make_tiny_model(input_dim=5)
        errors = compute_reconstruction_errors(model, df, fc, seq_length=10)
        assert (errors.dropna() >= 0).all()

    def test_no_series_boundary_contamination(self):
        """First rows of each series should not use errors from another series."""
        df, fc = _make_feature_df(n=30, n_features=5, n_series=2)
        model = _make_tiny_model(input_dim=5)
        errors = compute_reconstruction_errors(model, df, fc, seq_length=10)
        # All rows should have an assigned error (warm-up uses first window)
        per_series_nan = df.groupby("source_file").apply(
            lambda g: errors.loc[g.index].isna().sum(),
            include_groups=False,
        )
        # Series with enough rows should have no NaN
        for series_id, n_nan in per_series_nan.items():
            n_rows = df[df["source_file"] == series_id].shape[0]
            if n_rows >= 10:
                assert n_nan == 0, f"Series {series_id} has {n_nan} NaN errors"

    def test_errors_are_finite_where_not_nan(self):
        df, fc = _make_feature_df(n=50, n_features=5, n_series=2)
        model = _make_tiny_model(input_dim=5)
        errors = compute_reconstruction_errors(model, df, fc, seq_length=10)
        assert np.isfinite(errors.dropna()).all()


# ---------------------------------------------------------------------------
# Artifact save/load tests
# ---------------------------------------------------------------------------


class TestArtifactIO:
    """Tests for save_tcn_autoencoder() and load_tcn_autoencoder()."""

    def test_save_and_load_round_trip(self, tmp_path):
        model = _make_tiny_model(input_dim=5)
        path = tmp_path / "test_tcn.pt"
        save_tcn_autoencoder(model, path=path)
        assert path.exists()
        loaded = load_tcn_autoencoder(path=path)
        assert isinstance(loaded, TCNAutoencoder)
        assert loaded.hparams.input_dim == model.hparams.input_dim

    def test_loaded_model_produces_same_output(self, tmp_path):
        model = _make_tiny_model(input_dim=5)
        path = tmp_path / "test_tcn.pt"
        save_tcn_autoencoder(model, path=path)
        loaded = load_tcn_autoencoder(path=path)
        x = torch.randn(2, 10, 5)
        model.eval()
        with torch.no_grad():
            out_original = model(x)
            out_loaded = loaded(x)
        assert torch.allclose(out_original, out_loaded, atol=1e-6)

    def test_load_raises_on_missing_artifact(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tcn_autoencoder(path=tmp_path / "nonexistent.pt")

    def test_save_creates_parent_directory(self, tmp_path):
        model = _make_tiny_model(input_dim=5)
        nested = tmp_path / "nested" / "dir" / "model.pt"
        save_tcn_autoencoder(model, path=nested)
        assert nested.exists()

    def test_reconstruction_error_separation(self):
        """Anomaly sequences should have higher error than normal sequences."""
        import json

        model = load_tcn_autoencoder()
        with open("artifacts/feature_metadata.json") as f:
            fc = json.load(f)["nab_feature_cols"]
        val = pd.read_parquet("data/processed/nab_val_features.parquet")
        errors = compute_reconstruction_errors(model, val, fc)
        normal_mean = errors[~val["is_anomaly"]].dropna().mean()
        anomaly_mean = errors[val["is_anomaly"]].dropna().mean()
        assert anomaly_mean > normal_mean, (
            f"Separation check FAILED: anomaly mean ({anomaly_mean:.4f}) "
            f"≤ normal mean ({normal_mean:.4f}). Consider LSTM fallback."
        )


# ---------------------------------------------------------------------------
# Integration tests — require trained model artifact
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ARTIFACT_EXISTS, reason="TCN artifact not generated")
class TestTCNIntegration:
    """Integration tests against the real trained model."""

    def test_model_loadable(self):
        model = load_tcn_autoencoder()
        assert isinstance(model, TCNAutoencoder)

    def test_model_in_eval_mode(self):
        model = load_tcn_autoencoder()
        assert not model.training

    def test_thresholds_has_tcn_key(self):
        import joblib

        thresholds = joblib.load("artifacts/thresholds.joblib")
        assert "tcn_autoencoder" in thresholds
        assert isinstance(thresholds["tcn_autoencoder"], float)

    def test_metrics_has_tcn_results(self):
        import json

        with open("artifacts/metrics.json") as f:
            metrics = json.load(f)
        assert "tcn_autoencoder" in metrics
        assert "validation" in metrics["tcn_autoencoder"]
        auc = metrics["tcn_autoencoder"]["validation"]["auc_roc"]
        assert float(auc) > 0.0

    @pytest.mark.skipif(not FEATURES_EXIST, reason="Feature matrices missing")
    def test_inference_smoke_test(self):
        """Model produces finite errors on real feature data."""
        import json

        model = load_tcn_autoencoder()
        with open("artifacts/feature_metadata.json") as f:
            fc = json.load(f)["nab_feature_cols"]
        val = pd.read_parquet("data/processed/nab_val_features.parquet")
        sample = val.groupby("source_file").head(35)
        errors = compute_reconstruction_errors(model, sample, fc)
        assert errors.notna().sum() > 0
        assert np.isfinite(errors.dropna()).all()

    @pytest.mark.skipif(not FEATURES_EXIST, reason="Feature matrices missing")
    def test_reconstruction_error_separation(self):
        """Anomaly sequences should have higher error than normal sequences."""
        import json

        model = load_tcn_autoencoder()
        with open("artifacts/feature_metadata.json") as f:
            fc = json.load(f)["nab_feature_cols"]
        val = pd.read_parquet("data/processed/nab_val_features.parquet")
        errors = compute_reconstruction_errors(model, val, fc)
        normal_mean = errors[~val["is_anomaly"]].dropna().mean()
        anomaly_mean = errors[val["is_anomaly"]].dropna().mean()
        assert anomaly_mean > normal_mean, (
            f"Separation check FAILED: anomaly mean ({anomaly_mean:.4f}) "
            f"≤ normal mean ({normal_mean:.4f}). Consider LSTM fallback."
        )

"""
Feature engineering pipeline tests.

Unit tests (no data required):
    TestAlibabaFeatures             — build_alibaba_features() on synthetic data
    TestRobustPercentileNormalizer  — normalizer fit/transform
    TestFeaturePipeline             — pipeline build/save/load/apply

Integration tests (skip when artifacts absent):
    TestSMDArtifactsIntegration     — feature_metadata.json and pipeline consistency
    TestFeaturePipelineIntegration  — pipeline bounds against live artifact
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    RobustPercentileNormalizer,
    apply_feature_pipeline,
    build_alibaba_features,
    build_feature_pipeline,
    get_feature_columns,
    load_feature_pipeline,
    save_feature_pipeline,
)

PIPELINE_EXISTS = Path("artifacts/feature_pipeline.joblib").exists()
FEATURE_META_EXISTS = Path("artifacts/feature_metadata.json").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alibaba_df(n: int = 30, n_machines: int = 2) -> pd.DataFrame:
    """Create a minimal SMD/Alibaba-format DataFrame for testing."""
    dfs = []
    for m in range(n_machines):
        machine_name = f"machine-1-{m + 1}"
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
        dfs.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "machine_id": machine_name,
                    "source_file": machine_name,
                    "cpu_util": [float(0.20 + j * 0.005 + m * 0.05) for j in range(n)],
                    "mem_util": [float(0.40 + j * 0.003) for j in range(n)],
                    "net_io_in": [float(0.10 + j * 0.001) for j in range(n)],
                    "net_io_out": [float(0.05 + j * 0.0005) for j in range(n)],
                    "disk_io": [0.08] * n,
                    "is_anomaly": [j > n - 5 for j in range(n)],
                }
            )
        )
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Alibaba / SMD rolling feature tests
# ---------------------------------------------------------------------------


class TestAlibabaFeatures:
    """Tests for build_alibaba_features() — covers SMD and Alibaba formats."""

    def test_returns_rolling_features_per_metric(self):
        df = _make_alibaba_df()
        result = build_alibaba_features(df)
        for metric in ["cpu_util", "mem_util"]:
            assert f"{metric}_mean_long" in result.columns
            assert f"{metric}_zscore_mid" in result.columns

    def test_cross_metric_columns_present(self):
        df = _make_alibaba_df(n=30)
        result = build_alibaba_features(df)
        assert "cpu_mem_corr_long" in result.columns
        assert "cpu_net_ratio" in result.columns
        assert "volatility_score" in result.columns

    def test_per_machine_isolation(self):
        """First row roc of each machine must be 0."""
        df = _make_alibaba_df(n=10, n_machines=2)
        result = build_alibaba_features(df)
        for machine_id in df["machine_id"].unique():
            first_row = (
                result[result["machine_id"] == machine_id]
                .sort_values("timestamp")
                .iloc[0]
            )
            assert first_row["cpu_util_roc"] == 0.0

    def test_row_count_preserved(self):
        df = _make_alibaba_df(n=30, n_machines=3)
        result = build_alibaba_features(df)
        assert len(result) == len(df)

    def test_feature_count_is_68(self):
        """
        5 raw + 5×12 rolling + 3 cross-metric = 68 total.
        Must match input_dim=68 in the TCN Autoencoder.
        If this fails, update input_dim and regenerate artifacts.
        """
        df = _make_alibaba_df(n=50, n_machines=1)
        result = build_alibaba_features(df)
        feat_cols = get_feature_columns(result)
        assert len(feat_cols) == 68, (
            f"Expected 68 feature columns, got {len(feat_cols)}. "
            "Update TCN input_dim and day4_if_metrics.json if feature "
            "engineering has changed."
        )

    def test_raises_on_missing_group_column(self):
        df = pd.DataFrame({"timestamp": [], "cpu_util": []})
        with pytest.raises(ValueError, match="machine_id"):
            build_alibaba_features(df, group_col="machine_id")

    def test_raises_on_missing_metric_columns(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="1min"),
                "machine_id": ["m1"] * 3,
                "unrelated_col": [1.0, 2.0, 3.0],
            }
        )
        with pytest.raises(ValueError, match="No metric columns"):
            build_alibaba_features(df)


# ---------------------------------------------------------------------------
# RobustPercentileNormalizer tests
# ---------------------------------------------------------------------------


class TestRobustPercentileNormalizer:
    """Tests for the custom sklearn transformer."""

    def _make_feature_df(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "feat_a": rng.uniform(0, 100, n),
                "feat_b": rng.uniform(-10, 200, n),
            }
        )

    def test_fit_stores_bounds(self):
        norm = RobustPercentileNormalizer()
        df = self._make_feature_df()
        norm.fit(df)
        assert "feat_a" in norm.bounds_
        lo, hi = norm.bounds_["feat_a"]
        assert lo < hi

    def test_transform_output_mostly_in_0_1(self):
        norm = RobustPercentileNormalizer(lower_pct=1, upper_pct=99)
        df = self._make_feature_df(1000)
        norm.fit(df)
        result = norm.transform(df)
        assert result["feat_a"].between(0.0, 1.0).mean() > 0.98
        assert result["feat_b"].between(0.0, 1.0).mean() > 0.98

    def test_constant_column_gets_degenerate_bounds(self):
        norm = RobustPercentileNormalizer()
        df = pd.DataFrame({"constant": [5.0] * 50})
        norm.fit(df)
        lo, hi = norm.bounds_["constant"]
        assert hi > lo

    def test_missing_column_at_transform_gets_zeros(self):
        norm = RobustPercentileNormalizer()
        train = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "feat_b": [4.0, 5.0, 6.0]})
        norm.fit(train)
        new_data = pd.DataFrame({"feat_a": [1.5, 2.5]})
        result = norm.transform(new_data)
        assert "feat_b" in result.columns
        assert (result["feat_b"] == 0.0).all()

    def test_sklearn_pipeline_compatible(self):
        from sklearn.pipeline import Pipeline

        norm = RobustPercentileNormalizer()
        pipe = Pipeline([("normalizer", norm)])
        df = self._make_feature_df()
        pipe.fit(df)
        result = pipe.transform(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape


# ---------------------------------------------------------------------------
# Feature pipeline save/load tests
# ---------------------------------------------------------------------------


class TestFeaturePipeline:
    """
    Tests for pipeline build/save/load/apply round-trip.
    Uses SMD-format synthetic data throughout.
    """

    def test_build_pipeline_returns_fitted_pipeline(self):
        df = _make_alibaba_df(n=50)
        feat_df = build_alibaba_features(df)
        feat_cols = get_feature_columns(feat_df)
        pipeline = build_feature_pipeline(feat_df, feat_cols)
        assert hasattr(pipeline.named_steps["normalizer"], "bounds_")

    def test_save_load_round_trip(self, tmp_path):
        df = _make_alibaba_df(n=50)
        feat_df = build_alibaba_features(df)
        feat_cols = get_feature_columns(feat_df)
        pipeline = build_feature_pipeline(feat_df, feat_cols)

        save_path = tmp_path / "test_pipeline.joblib"
        save_feature_pipeline(pipeline, path=save_path)
        assert save_path.exists()

        loaded = load_feature_pipeline(path=save_path)
        assert pipeline.named_steps["normalizer"].bounds_ == (
            loaded.named_steps["normalizer"].bounds_
        )

    def test_load_raises_on_missing_artifact(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_feature_pipeline(path=tmp_path / "nonexistent.joblib")

    def test_apply_pipeline_preserves_metadata(self):
        df = _make_alibaba_df(n=50)
        feat_df = build_alibaba_features(df)
        feat_cols = get_feature_columns(feat_df)
        pipeline = build_feature_pipeline(feat_df, feat_cols)
        result = apply_feature_pipeline(pipeline, feat_df, feat_cols)

        assert "timestamp" in result.columns
        assert "is_anomaly" in result.columns
        assert "machine_id" in result.columns
        pd.testing.assert_series_equal(feat_df["timestamp"], result["timestamp"])

    def test_apply_pipeline_features_in_0_1(self):
        df = _make_alibaba_df(n=100)
        feat_df = build_alibaba_features(df)
        feat_cols = get_feature_columns(feat_df)
        pipeline = build_feature_pipeline(feat_df, feat_cols)
        result = apply_feature_pipeline(pipeline, feat_df, feat_cols)
        feat_values = result[feat_cols]
        assert feat_values.min().min() >= 0.0
        assert feat_values.max().max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Integration tests — require artifacts from Days 4-6
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not FEATURE_META_EXISTS,
    reason="feature_metadata.json not generated — run generate_api_artifacts.py",
)
class TestSMDArtifactsIntegration:
    """Validates feature_metadata.json is consistent with live feature engineering."""

    def test_feature_metadata_has_correct_keys(self):
        import json

        with open("artifacts/feature_metadata.json") as f:
            meta = json.load(f)

        required_keys = {
            "feature_cols",
            "n_features",
            "input_dim",
            "dataset",
            "metric_cols",
            "seq_length",
        }
        assert required_keys.issubset(set(meta.keys()))

    def test_feature_count_matches_metadata(self):
        import json

        with open("artifacts/feature_metadata.json") as f:
            meta = json.load(f)

        df = _make_alibaba_df(n=50)
        feat_df = build_alibaba_features(df)
        actual_cols = get_feature_columns(feat_df)

        assert len(actual_cols) == meta["n_features"], (
            f"Metadata says {meta['n_features']} features but "
            f"build_alibaba_features() produces {len(actual_cols)}. "
            "Regenerate via generate_api_artifacts.py."
        )

    def test_input_dim_matches_n_features(self):
        import json

        with open("artifacts/feature_metadata.json") as f:
            meta = json.load(f)
        assert meta["input_dim"] == meta["n_features"]

    def test_dataset_is_smd(self):
        import json

        with open("artifacts/feature_metadata.json") as f:
            meta = json.load(f)
        assert meta["dataset"] == "SMD"


@pytest.mark.skipif(
    not PIPELINE_EXISTS,
    reason="feature_pipeline.joblib not generated — run day4_if_training_smd.py",
)
class TestFeaturePipelineIntegration:
    """Integration tests for the live fitted pipeline artifact."""

    def test_pipeline_artifact_loadable(self):
        pipeline = load_feature_pipeline()
        norm = pipeline.named_steps["normalizer"]
        assert hasattr(norm, "bounds_")
        assert len(norm.bounds_) > 0

    def test_pipeline_bounds_are_finite(self):
        """
        All normalizer bounds must be finite.
        NaN bounds cause NaN loss in the TCN on the first epoch.
        """
        pipeline = load_feature_pipeline()
        norm = pipeline.named_steps["normalizer"]
        nan_cols = [
            col
            for col, (lo, hi) in norm.bounds_.items()
            if np.isnan(lo) or np.isnan(hi) or np.isinf(lo) or np.isinf(hi)
        ]
        assert nan_cols == [], (
            f"NaN/Inf bounds found: {nan_cols}. "
            "Apply bounds patch before using the pipeline."
        )

    def test_bounds_unchanged_after_apply(self):
        """Applying the pipeline must not refit bounds on new data."""
        pipeline = load_feature_pipeline()
        norm = pipeline.named_steps["normalizer"]
        all_fitted_cols = list(norm.bounds_.keys())

        df = _make_alibaba_df(n=100)
        feat_df = build_alibaba_features(df)

        # Use only columns present in both the pipeline and the synthetic df
        common_cols = sorted([c for c in all_fitted_cols if c in feat_df.columns])
        if not common_cols:
            pytest.skip("No overlapping columns between pipeline and synthetic data")

        bounds_before = {c: norm.bounds_[c] for c in common_cols}
        apply_feature_pipeline(pipeline, feat_df, common_cols)

        for col in common_cols:
            assert norm.bounds_[col] == bounds_before[col], (
                f"Bounds for '{col}' changed after apply — pipeline was refitted"
            )

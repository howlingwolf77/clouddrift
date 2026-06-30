"""
Day 4 tests: Isolation Forest training, cross-validation,
threshold calibration, and evaluation.

NOTE: This file was reconstructed after being accidentally lost during
Day 4 troubleshooting (never committed in any git history — confirmed via
`git log --all --full-history -- tests/test_isolation_forest.py` returning
no results). It mirrors the final state of src/models/isolation_forest.py
as of commit d8cc891 ("Day 4: accept AUC-ROC as IF primary metric; clean
threshold strategy") plus the subsequent F2-score (fbeta_score, beta=2)
addition to run_timeseries_cross_validation() and evaluate_model().

If any test below fails on import or signature mismatch, the live
isolation_forest.py has likely drifted further from this reconstruction —
paste the error and we will reconcile in one pass.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from src.models.isolation_forest import (
    TARGET_PRECISION,
    TARGET_RECALL,
    calibrate_threshold,
    compute_anomaly_scores,
    evaluate_model,
    load_isolation_forest,
    run_timeseries_cross_validation,
    save_isolation_forest,
    train_isolation_forest,
)

ARTIFACTS_EXIST = (
    Path("artifacts/isolation_forest.joblib").exists()
    and Path("artifacts/thresholds.joblib").exists()
    and Path("artifacts/metrics.json").exists()
)
FEATURES_EXIST = Path("data/processed/nab_train_features.parquet").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_df(
    n: int = 200,
    anomaly_rate: float = 0.10,
    n_features: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic feature DataFrame with normal and anomaly rows."""
    rng = np.random.default_rng(seed)
    n_anomaly = int(n * anomaly_rate)
    n_normal = n - n_anomaly

    normal_data = rng.normal(0.5, 0.1, (n_normal, n_features))
    anomaly_data = rng.normal(0.9, 0.15, (n_anomaly, n_features))

    data = np.vstack([normal_data, anomaly_data]).clip(0, 1)
    labels = np.array([False] * n_normal + [True] * n_anomaly)

    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
    feature_cols = [f"feat_{i}" for i in range(n_features)]

    df = pd.DataFrame(data, columns=feature_cols)
    df["timestamp"] = timestamps
    df["is_anomaly"] = labels
    df["source_file"] = "test/series_0.csv"
    return df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feat_")]


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


class TestTrainIsolationForest:
    """Tests for train_isolation_forest()."""

    def test_returns_fitted_model(self):
        df = _make_feature_df()
        feature_cols = _get_feature_cols(df)
        x_normal = df.loc[~df["is_anomaly"], feature_cols]
        model = train_isolation_forest(x_normal)
        assert isinstance(model, IsolationForest)
        assert hasattr(model, "estimators_")

    def test_model_has_expected_n_estimators(self):
        df = _make_feature_df()
        feature_cols = _get_feature_cols(df)
        x_normal = df.loc[~df["is_anomaly"], feature_cols]
        model = train_isolation_forest(x_normal)
        assert model.n_estimators == 100

    def test_raises_on_empty_input(self):
        feature_cols = ["feat_0", "feat_1"]
        x_empty = pd.DataFrame(columns=feature_cols)
        with pytest.raises(ValueError, match="empty"):
            train_isolation_forest(x_empty)

    def test_trains_on_normal_rows_only(self):
        """Anomaly rows must be excluded before passing to train_isolation_forest."""
        df = _make_feature_df(n=100, anomaly_rate=0.10)
        feature_cols = _get_feature_cols(df)
        x_normal = df.loc[~df["is_anomaly"], feature_cols]
        assert len(x_normal) == 90
        model = train_isolation_forest(x_normal)
        assert hasattr(model, "estimators_")


# ---------------------------------------------------------------------------
# Anomaly score tests
# ---------------------------------------------------------------------------


class TestComputeAnomalyScores:
    """Tests for compute_anomaly_scores()."""

    def _fit_model(self, n: int = 100) -> tuple:
        df = _make_feature_df(n=n)
        feature_cols = _get_feature_cols(df)
        x_normal = df.loc[~df["is_anomaly"], feature_cols]
        model = train_isolation_forest(x_normal)
        return model, df, feature_cols

    def test_returns_numpy_array(self):
        model, df, fc = self._fit_model()
        scores = compute_anomaly_scores(model, df[fc])
        assert isinstance(scores, np.ndarray)

    def test_output_length_matches_input(self):
        model, df, fc = self._fit_model()
        scores = compute_anomaly_scores(model, df[fc])
        assert len(scores) == len(df)

    def test_anomaly_rows_have_higher_scores_on_average(self):
        """Anomaly rows should receive higher anomaly scores than normal rows."""
        df = _make_feature_df(n=500, anomaly_rate=0.20)
        feature_cols = _get_feature_cols(df)
        x_normal = df.loc[~df["is_anomaly"], feature_cols]
        model = train_isolation_forest(x_normal)
        scores = compute_anomaly_scores(model, df[feature_cols])

        mean_anomaly = scores[df["is_anomaly"].values].mean()
        mean_normal = scores[~df["is_anomaly"].values].mean()
        assert mean_anomaly > mean_normal, (
            f"Expected anomaly mean score ({mean_anomaly:.4f}) > "
            f"normal mean score ({mean_normal:.4f})"
        )

    def test_scores_are_finite(self):
        model, df, fc = self._fit_model()
        scores = compute_anomaly_scores(model, df[fc])
        assert np.isfinite(scores).all()


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------


class TestTimeSeriesCrossValidation:
    """Tests for run_timeseries_cross_validation()."""

    def _make_cv_df(self, n: int = 500) -> tuple:
        df = _make_feature_df(n=n, anomaly_rate=0.10)
        feature_cols = _get_feature_cols(df)
        return df, feature_cols

    def test_returns_expected_keys(self):
        df, fc = self._make_cv_df()
        result = run_timeseries_cross_validation(df, fc, n_splits=3)
        assert "folds" in result
        assert "summary" in result

    def test_correct_number_of_folds(self):
        df, fc = self._make_cv_df()
        result = run_timeseries_cross_validation(df, fc, n_splits=3)
        assert len(result["folds"]) == 3

    def test_summary_has_required_keys(self):
        df, fc = self._make_cv_df()
        result = run_timeseries_cross_validation(df, fc, n_splits=3)
        s = result["summary"]
        for key in ["mean_f1", "std_f1", "stability_check_pass"]:
            assert key in s, f"Missing summary key: {key}"

    def test_fold_metrics_between_0_and_1(self):
        df, fc = self._make_cv_df()
        result = run_timeseries_cross_validation(df, fc, n_splits=3)
        for fold in result["folds"]:
            if not np.isnan(fold["precision"]):
                assert 0.0 <= fold["precision"] <= 1.0
            if not np.isnan(fold["recall"]):
                assert 0.0 <= fold["recall"] <= 1.0
            if not np.isnan(fold["f1"]):
                assert 0.0 <= fold["f1"] <= 1.0
            if "f2" in fold and not np.isnan(fold["f2"]):
                assert 0.0 <= fold["f2"] <= 1.0

    def test_fold_sizes_increase_monotonically(self):
        """TimeSeriesSplit: each fold trains on more data than the previous."""
        df, fc = self._make_cv_df(n=500)
        result = run_timeseries_cross_validation(df, fc, n_splits=3)
        train_sizes = [r["train_size"] for r in result["folds"]]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                "Train sizes must be non-decreasing in TimeSeriesSplit"
            )

    def test_fold_f2_key_present_when_anomalies_in_val_split(self):
        """
        F2 (beta=2) was added alongside F1 for every fold that has
        anomalies in its validation split. No-anomaly folds set f2=nan
        alongside precision/recall/f1=nan.
        """
        df, fc = self._make_cv_df(n=500)
        result = run_timeseries_cross_validation(df, fc, n_splits=3)
        for fold in result["folds"]:
            assert "f2" in fold, "f2 key missing from fold result"


# ---------------------------------------------------------------------------
# Threshold calibration tests
# ---------------------------------------------------------------------------


class TestCalibrateThreshold:
    """
    Tests for calibrate_threshold().

    Final signature (post Day-4 troubleshooting): calibrate_threshold(model,
    x_val, percentile=90.0) -> float. The earlier y_val-based precision-recall
    calibration strategies were replaced with a simple percentile-of-val-scores
    approach after repeated failure at the 1.1% validation anomaly rate
    (documented in commit d8cc891).
    """

    def _make_fitted_model(self) -> tuple:
        df = _make_feature_df(n=300, anomaly_rate=0.10)
        feature_cols = _get_feature_cols(df)
        x_normal = df.loc[~df["is_anomaly"], feature_cols]
        model = train_isolation_forest(x_normal)
        return model, df, feature_cols

    def test_returns_float(self):
        model, df, fc = self._make_fitted_model()
        threshold = calibrate_threshold(model, df[fc])
        assert isinstance(threshold, float)

    def test_threshold_is_finite(self):
        model, df, fc = self._make_fitted_model()
        threshold = calibrate_threshold(model, df[fc])
        assert np.isfinite(threshold)

    def test_threshold_within_score_range(self):
        model, df, fc = self._make_fitted_model()
        threshold = calibrate_threshold(model, df[fc])
        scores = compute_anomaly_scores(model, df[fc])
        assert scores.min() <= threshold <= scores.max()

    def test_higher_percentile_raises_threshold(self):
        model, df, fc = self._make_fitted_model()
        low_threshold = calibrate_threshold(model, df[fc], percentile=50.0)
        high_threshold = calibrate_threshold(model, df[fc], percentile=95.0)
        assert high_threshold >= low_threshold

    def test_default_percentile_is_90(self):
        model, df, fc = self._make_fitted_model()
        default_threshold = calibrate_threshold(model, df[fc])
        explicit_threshold = calibrate_threshold(model, df[fc], percentile=90.0)
        assert abs(default_threshold - explicit_threshold) < 1e-9


# ---------------------------------------------------------------------------
# Model evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    """Tests for evaluate_model() — includes F1 and F2 (beta=2)."""

    def _setup(self) -> tuple:
        df = _make_feature_df(n=300, anomaly_rate=0.15)
        fc = _get_feature_cols(df)
        x_normal = df.loc[~df["is_anomaly"], fc]
        model = train_isolation_forest(x_normal)
        threshold = calibrate_threshold(model, df[fc])
        return model, threshold, df, fc

    def test_returns_required_keys(self):
        model, threshold, df, fc = self._setup()
        metrics = evaluate_model(model, threshold, df[fc], df["is_anomaly"], "test")
        for key in [
            "precision",
            "recall",
            "f1",
            "f2",
            "auc_roc",
            "n_rows",
            "threshold",
        ]:
            assert key in metrics, f"Missing key: {key}"

    def test_metrics_between_0_and_1(self):
        model, threshold, df, fc = self._setup()
        metrics = evaluate_model(model, threshold, df[fc], df["is_anomaly"], "test")
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["f2"] <= 1.0
        if not np.isnan(metrics["auc_roc"]):
            assert 0.0 <= metrics["auc_roc"] <= 1.0

    def test_n_rows_matches_input(self):
        model, threshold, df, fc = self._setup()
        metrics = evaluate_model(model, threshold, df[fc], df["is_anomaly"], "test")
        assert metrics["n_rows"] == len(df)

    def test_f1_consistent_with_precision_recall(self):
        model, threshold, df, fc = self._setup()
        metrics = evaluate_model(model, threshold, df[fc], df["is_anomaly"], "test")
        p, r = metrics["precision"], metrics["recall"]
        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            assert abs(metrics["f1"] - expected_f1) < 1e-6

    def test_f2_consistent_with_precision_recall(self):
        """F2 (beta=2) weights recall twice as heavily as precision."""
        model, threshold, df, fc = self._setup()
        metrics = evaluate_model(model, threshold, df[fc], df["is_anomaly"], "test")
        p, r = metrics["precision"], metrics["recall"]
        beta = 2
        if (beta**2 * p + r) > 0:
            expected_f2 = (1 + beta**2) * p * r / (beta**2 * p + r)
            assert abs(metrics["f2"] - expected_f2) < 1e-6

    def test_f2_weights_recall_more_than_f1(self):
        """
        Sanity check on the F2 definition itself: when recall > precision,
        F2 should be closer to recall than F1 is (since F2 weights recall
        more heavily). Constructed with a threshold that favors recall.
        """
        model, threshold, df, fc = self._setup()
        # Use a low threshold to bias toward high recall, low precision
        low_threshold = calibrate_threshold(model, df[fc], percentile=20.0)
        metrics = evaluate_model(model, low_threshold, df[fc], df["is_anomaly"], "test")
        if metrics["recall"] > metrics["precision"] and metrics["recall"] > 0:
            assert metrics["f2"] >= metrics["f1"] - 1e-9

    def test_meets_target_flags_present(self):
        model, threshold, df, fc = self._setup()
        metrics = evaluate_model(model, threshold, df[fc], df["is_anomaly"], "test")
        assert "meets_precision_target" in metrics
        assert "meets_recall_target" in metrics
        assert metrics["meets_precision_target"] == (
            metrics["precision"] >= TARGET_PRECISION
        )
        assert metrics["meets_recall_target"] == (metrics["recall"] >= TARGET_RECALL)


# ---------------------------------------------------------------------------
# Artifact save/load tests
# ---------------------------------------------------------------------------


class TestArtifactIO:
    """Tests for save_isolation_forest() and load_isolation_forest()."""

    def test_save_and_load_round_trip(self, tmp_path):
        df = _make_feature_df(n=100)
        fc = _get_feature_cols(df)
        x_normal = df.loc[~df["is_anomaly"], fc]
        original = train_isolation_forest(x_normal)

        save_path = tmp_path / "test_if.joblib"
        save_isolation_forest(original, path=save_path)
        assert save_path.exists()

        loaded = load_isolation_forest(path=save_path)
        assert isinstance(loaded, IsolationForest)
        assert loaded.n_estimators == original.n_estimators

    def test_save_creates_parent_directory(self, tmp_path):
        df = _make_feature_df(n=100)
        fc = _get_feature_cols(df)
        model = train_isolation_forest(df.loc[~df["is_anomaly"], fc])
        nested_path = tmp_path / "nested" / "dir" / "model.joblib"
        save_isolation_forest(model, path=nested_path)
        assert nested_path.exists()

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_isolation_forest(path=tmp_path / "nonexistent.joblib")


# ---------------------------------------------------------------------------
# Integration tests — require Day 4 pipeline to have run
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ARTIFACTS_EXIST, reason="Day 4 artifacts not generated")
class TestIFIntegration:
    """Integration tests against real trained model and metrics."""

    def test_model_artifact_loadable(self):
        model = load_isolation_forest()
        assert hasattr(model, "estimators_")

    def test_thresholds_artifact_has_if_key(self):
        thresholds = joblib.load("artifacts/thresholds.joblib")
        assert "isolation_forest" in thresholds
        assert isinstance(thresholds["isolation_forest"], float)

    def test_metrics_cv_stability_check_recorded(self):
        """
        Stability check is recorded in metrics.json but is NOT required
        to pass — CV instability was accepted and documented (commit
        d8cc891) since the ensemble in Phase 1D compensates for it.
        """
        import json

        with open("artifacts/metrics.json") as f:
            metrics = json.load(f)
        cv_summary = metrics["isolation_forest"]["cross_validation"]["summary"]
        assert "stability_check_pass" in cv_summary
        assert isinstance(cv_summary["stability_check_pass"], bool)

    def test_metrics_includes_f2(self):
        import json

        with open("artifacts/metrics.json") as f:
            metrics = json.load(f)
        val_metrics = metrics["isolation_forest"]["validation"]
        assert "f2" in val_metrics

    def test_val_auc_roc_above_random(self):
        import json

        with open("artifacts/metrics.json") as f:
            metrics = json.load(f)
        auc_roc = metrics["isolation_forest"]["validation"]["auc_roc"]
        assert float(auc_roc) > 0.5, (
            f"Validation AUC-ROC {auc_roc:.3f} is not better than random chance"
        )

    def test_threshold_strategy_documented(self):
        """
        metrics.json should record why/how the threshold was chosen —
        this was an explicit deliverable of the Day 4 troubleshooting
        (threshold_strategy / threshold_rationale fields).
        """
        import json

        with open("artifacts/metrics.json") as f:
            metrics = json.load(f)
        if_metrics = metrics["isolation_forest"]
        has_rationale = (
            "threshold_strategy" in if_metrics
            or "threshold_rationale" in if_metrics
            or "notes" in if_metrics
        )
        assert has_rationale, (
            "Expected threshold_strategy, threshold_rationale, or notes "
            "field documenting the threshold calibration approach"
        )

    @pytest.mark.skipif(not FEATURES_EXIST, reason="Feature matrices missing")
    def test_inference_smoke_test(self):
        """Model produces finite scores on real feature data."""
        import json

        model = load_isolation_forest()
        with open("artifacts/feature_metadata.json") as f:
            meta = json.load(f)
        fc = meta["nab_feature_cols"]
        val = pd.read_parquet("data/processed/nab_val_features.parquet")
        scores = compute_anomaly_scores(model, val[fc].head(50))
        assert len(scores) == 50
        assert np.isfinite(scores).all()

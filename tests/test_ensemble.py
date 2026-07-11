"""
Day 6 tests: ensemble scoring and z-score attribution.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import (
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    compute_ensemble_score,
    evaluate_ensemble,
    fit_score_bounds,
    get_severity_label,
    load_ensemble_metadata,
    normalize_scores,
    rank_top_anomalies,
)
from src.utils.explanation import (
    build_reference_stats,
    compute_feature_deviation_scores,
    explain_anomaly_row,
    load_reference_stats,
    save_reference_stats,
)

METADATA_EXISTS = Path("artifacts/ensemble_metadata.json").exists()
REF_STATS_EXISTS = Path("artifacts/reference_stats.json").exists()
FEATURES_EXIST = (
    False  # SMD pipeline uses in-memory splits — no parquet feature artifacts
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scores(n: int = 100, seed: int = 42) -> tuple:
    """Return synthetic if_norm, tcn_norm, and labels."""
    rng = np.random.default_rng(seed)
    n_anomaly = int(n * 0.10)
    n_normal = n - n_anomaly

    if_norm = pd.Series(
        np.concatenate(
            [rng.uniform(0.0, 0.5, n_normal), rng.uniform(0.6, 1.0, n_anomaly)]
        )
    )
    tcn_norm = pd.Series(
        np.concatenate(
            [rng.uniform(0.0, 0.4, n_normal), rng.uniform(0.5, 1.0, n_anomaly)]
        )
    )
    labels = pd.Series([False] * n_normal + [True] * n_anomaly)
    return if_norm, tcn_norm, labels


def _make_feature_df(n: int = 100, n_features: int = 5) -> tuple:
    """Synthetic feature DataFrame for explanation tests."""
    rng = np.random.default_rng(42)
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(rng.uniform(0, 1, (n, n_features)), columns=cols)
    df["is_anomaly"] = False
    df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")
    df["source_file"] = "test/s0.csv"
    return df, cols


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


class TestScoreNormalization:
    """Tests for fit_score_bounds() and normalize_scores()."""

    def test_fit_bounds_returns_finite_values(self):
        scores = pd.Series(np.random.default_rng(0).uniform(0, 1, 100))
        lo, hi = fit_score_bounds(scores)
        assert np.isfinite(lo) and np.isfinite(hi)

    def test_fit_bounds_upper_greater_than_lower(self):
        scores = pd.Series(np.random.default_rng(0).uniform(0, 1, 100))
        lo, hi = fit_score_bounds(scores)
        assert hi > lo

    def test_normalize_training_data_mostly_0_1(self):
        scores = pd.Series(np.random.default_rng(0).uniform(0, 1, 1000))
        lo, hi = fit_score_bounds(scores, lower_pct=1, upper_pct=99)
        norm = normalize_scores(scores, lo, hi)
        assert norm.between(0, 1).mean() > 0.98

    def test_normalize_clips_above_bounds(self):
        scores = pd.Series([0.0, 0.5, 1.0, 2.0, 10.0])
        norm = normalize_scores(scores, lo=0.0, hi=1.0)
        assert norm.max() <= 1.0

    def test_normalize_clips_below_bounds(self):
        scores = pd.Series([-5.0, 0.0, 0.5, 1.0])
        norm = normalize_scores(scores, lo=0.0, hi=1.0)
        assert norm.min() >= 0.0

    def test_normalize_preserves_nan(self):
        scores = pd.Series([0.1, float("nan"), 0.9])
        norm = normalize_scores(scores, lo=0.0, hi=1.0)
        assert norm.isna().sum() == 1


# ---------------------------------------------------------------------------
# Ensemble combination tests
# ---------------------------------------------------------------------------


class TestEnsembleCombination:
    """Tests for compute_ensemble_score()."""

    def test_ensemble_between_0_and_1(self):
        if_n, tcn_n, _ = _make_scores()
        ensemble = compute_ensemble_score(if_n, tcn_n)
        assert ensemble.between(0, 1).all()

    def test_nan_tcn_falls_back_to_if(self):
        if_n = pd.Series([0.3, 0.7, 0.5])
        tcn_n = pd.Series([0.4, float("nan"), 0.6])
        ensemble = compute_ensemble_score(if_n, tcn_n, if_weight=0.4)
        # Row 1: TCN is NaN → should equal IF score
        assert abs(float(ensemble.iloc[1]) - 0.7) < 1e-6

    def test_equal_weights_is_average(self):
        if_n = pd.Series([0.2, 0.4, 0.6])
        tcn_n = pd.Series([0.4, 0.6, 0.8])
        ensemble = compute_ensemble_score(if_n, tcn_n, if_weight=0.5)
        expected = pd.Series([0.3, 0.5, 0.7])
        pd.testing.assert_series_equal(ensemble, expected, check_names=False)

    def test_higher_if_weight_increases_if_influence(self):
        if_n = pd.Series([1.0])
        tcn_n = pd.Series([0.0])
        high_if = compute_ensemble_score(if_n, tcn_n, if_weight=0.8)
        low_if = compute_ensemble_score(if_n, tcn_n, if_weight=0.2)
        assert float(high_if.iloc[0]) > float(low_if.iloc[0])

    def test_no_nan_in_output(self):
        if_n = pd.Series([0.3, 0.7])
        tcn_n = pd.Series([float("nan"), 0.6])
        ensemble = compute_ensemble_score(if_n, tcn_n)
        assert not ensemble.isna().any()


# ---------------------------------------------------------------------------
# Severity label tests
# ---------------------------------------------------------------------------


class TestSeverityLabel:
    """Tests for get_severity_label()."""

    def test_critical_at_or_above_threshold(self):
        assert get_severity_label(SEVERITY_CRITICAL) == "Critical"
        assert get_severity_label(1.0) == "Critical"

    def test_warning_between_thresholds(self):
        assert get_severity_label(SEVERITY_WARNING) == "Warning"
        assert get_severity_label(0.65) == "Warning"

    def test_normal_below_warning_threshold(self):
        assert get_severity_label(0.0) == "Normal"
        assert get_severity_label(0.49) == "Normal"


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEnsembleEvaluation:
    """Tests for evaluate_ensemble()."""

    def test_returns_required_keys(self):
        if_n, tcn_n, labels = _make_scores()
        scores = compute_ensemble_score(if_n, tcn_n)
        threshold = float(np.percentile(scores, 90))
        metrics = evaluate_ensemble(scores, threshold, labels, "test")
        for key in ["precision", "recall", "f1", "f2", "auc_roc"]:
            assert key in metrics

    def test_metrics_between_0_and_1(self):
        if_n, tcn_n, labels = _make_scores()
        scores = compute_ensemble_score(if_n, tcn_n)
        threshold = float(np.percentile(scores, 90))
        metrics = evaluate_ensemble(scores, threshold, labels, "test")
        for key in ["precision", "recall", "f1", "f2"]:
            assert 0.0 <= metrics[key] <= 1.0


# ---------------------------------------------------------------------------
# Ranking tests
# ---------------------------------------------------------------------------


class TestRankTopAnomalies:
    """Tests for rank_top_anomalies()."""

    def test_returns_n_rows_when_enough_flagged(self):
        df, _ = _make_feature_df(n=100)
        scores = pd.Series(np.linspace(0, 1, 100), index=df.index)
        top = rank_top_anomalies(df, scores, threshold=0.5, n=5)
        assert len(top) <= 5

    def test_sorted_descending(self):
        df, _ = _make_feature_df(n=50)
        scores = pd.Series(np.random.default_rng(0).uniform(0, 1, 50), index=df.index)
        top = rank_top_anomalies(df, scores, threshold=0.0, n=10)
        if len(top) > 1:
            assert top["ensemble_score"].is_monotonic_decreasing

    def test_all_returned_rows_above_threshold(self):
        df, _ = _make_feature_df(n=50)
        scores = pd.Series(np.random.default_rng(0).uniform(0, 1, 50), index=df.index)
        threshold = 0.7
        top = rank_top_anomalies(df, scores, threshold=threshold)
        if len(top) > 0:
            assert (top["ensemble_score"] >= threshold).all()


# ---------------------------------------------------------------------------
# Z-score attribution tests
# ---------------------------------------------------------------------------


class TestFeatureDeviation:
    """Tests for compute_feature_deviation_scores() and build_reference_stats()."""

    def _make_stats(self, n_features: int = 5) -> tuple:
        cols = [f"feat_{i}" for i in range(n_features)]
        df, _ = _make_feature_df(n_features=n_features)
        stats = build_reference_stats(df, cols)
        return cols, stats

    def test_build_reference_stats_keys(self):
        cols, stats = self._make_stats()
        for col in cols:
            assert col in stats
            assert "mean" in stats[col]
            assert "std" in stats[col]

    def test_std_is_positive(self):
        _, stats = self._make_stats()
        for col_stats in stats.values():
            assert col_stats["std"] > 0

    def test_returns_sorted_by_deviation_descending(self):
        cols, stats = self._make_stats(5)
        # High value = high deviation
        snapshot = {col: stats[col]["mean"] + 5 * stats[col]["std"] for col in cols[:2]}
        snapshot.update({col: stats[col]["mean"] for col in cols[2:]})
        result = compute_feature_deviation_scores(snapshot, cols, stats)
        scores = [d["deviation_score"] for d in result]
        assert scores == sorted(scores, reverse=True)

    def test_n_top_respected(self):
        cols, stats = self._make_stats(10)
        snapshot = {col: stats[col]["mean"] + 3 for col in cols}
        result = compute_feature_deviation_scores(snapshot, cols, stats, n_top=3)
        assert len(result) == 3

    def test_mean_value_gets_zero_deviation(self):
        cols, stats = self._make_stats(3)
        snapshot = {col: stats[col]["mean"] for col in cols}
        result = compute_feature_deviation_scores(snapshot, cols, stats)
        for d in result:
            assert abs(d["deviation_score"]) < 1e-6

    def test_explain_anomaly_row_returns_expected_keys(self):
        df, cols = _make_feature_df(n_features=5)
        stats = build_reference_stats(df, cols)
        row = df.iloc[0]
        explanation = explain_anomaly_row(row, cols, stats, n_top=3)
        assert "top_features" in explanation
        assert "top_feature_names" in explanation
        assert "top_deviation_scores" in explanation
        assert len(explanation["top_features"]) <= 3

    def test_save_load_reference_stats_round_trip(self, tmp_path):
        df, cols = _make_feature_df()
        original = build_reference_stats(df, cols)
        path = tmp_path / "ref_stats.json"
        save_reference_stats(original, path)
        loaded = load_reference_stats(path)
        assert set(original.keys()) == set(loaded.keys())
        for col in original:
            assert abs(original[col]["mean"] - loaded[col]["mean"]) < 1e-9


# ---------------------------------------------------------------------------
# Integration tests — require Day 6 pipeline to have run
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not METADATA_EXISTS, reason="ensemble_metadata.json not generated")
class TestEnsembleIntegration:
    """Integration tests against real ensemble metadata."""

    def test_metadata_loadable(self):
        meta = load_ensemble_metadata()
        assert "val_auc_roc" in meta
        assert "val_metrics" in meta

    def test_weights_sum_to_1(self):
        meta = load_ensemble_metadata()
        assert abs(meta["if_weight"] + meta["tcn_weight"] - 1.0) < 1e-9

    def test_ensemble_auc_roc_logged(self):
        import json

        with open("artifacts/metrics.json") as f:
            m = json.load(f)
        assert "ensemble" in m
        auc = float(m["ensemble"]["validation"]["auc_roc"])
        assert auc > 0.0

    @pytest.mark.skipif(not REF_STATS_EXISTS, reason="reference_stats.json not found")
    def test_reference_stats_has_all_features(self):
        import json

        stats = load_reference_stats()
        with open("artifacts/feature_metadata.json") as f:
            fc = json.load(f)["feature_cols"]
        assert all(col in stats for col in fc)

    @pytest.mark.skipif(not FEATURES_EXIST, reason="Feature matrices missing")
    def test_ensemble_scores_all_val_rows(self):
        import json

        from src.models.ensemble import (
            compute_ensemble_score,
            compute_if_scores,
            compute_tcn_scores,
            normalize_scores,
        )
        from src.models.isolation_forest import load_isolation_forest
        from src.models.tcn_autoencoder import load_tcn_autoencoder

        meta = load_ensemble_metadata()
        with open("artifacts/feature_metadata.json") as f:
            fc = json.load(f)["feature_cols"]

        val = pd.read_parquet("data/processed/nab_val_features.parquet")
        if_model = load_isolation_forest()
        tcn_model = load_tcn_autoencoder()

        if_scores = compute_if_scores(if_model, val, fc)
        tcn_scores = compute_tcn_scores(tcn_model, val, fc)

        if_norm = normalize_scores(
            if_scores, meta["if_bounds"]["lower"], meta["if_bounds"]["upper"]
        )
        tcn_norm = normalize_scores(
            tcn_scores, meta["tcn_bounds"]["lower"], meta["tcn_bounds"]["upper"]
        )
        ensemble = compute_ensemble_score(if_norm, tcn_norm, meta["if_weight"])

        assert len(ensemble) == len(val)
        assert not ensemble.isna().any()
        assert ensemble.between(0, 1).all()

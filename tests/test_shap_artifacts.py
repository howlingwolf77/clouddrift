"""
Day 7 tests: SHAP notebook and artifact verification.
"""

import json
from pathlib import Path

import pytest

NOTEBOOK_PATH = Path("notebooks/06_shap_analysis.ipynb")
PLOTS_DIR = Path("artifacts/shap_plots")

NOTEBOOK_EXISTS = NOTEBOOK_PATH.exists()
PLOTS_EXIST = PLOTS_DIR.exists()


def test_shap_importable_in_main_env():
    """Confirms shap lives in the main project environment — no isolation needed."""
    import shap  # noqa: F401


def test_shap_treeexplainer_works_on_real_if_model():
    import joblib
    import shap

    model = joblib.load("artifacts/isolation_forest.joblib")
    explainer = shap.TreeExplainer(model)
    assert explainer is not None


@pytest.mark.skipif(not NOTEBOOK_EXISTS, reason="Notebook not generated")
class TestNotebookStructure:
    """Tests for notebooks/06_shap_analysis.ipynb structure."""

    def _load(self) -> dict:
        with open(NOTEBOOK_PATH) as f:
            return json.load(f)

    def test_valid_nbformat_json(self):
        nb = self._load()
        assert nb["nbformat"] == 4

    def test_has_minimum_cell_count(self):
        nb = self._load()
        assert len(nb["cells"]) >= 10

    def test_kernel_is_main_environment(self):
        nb = self._load()
        assert nb["metadata"]["kernelspec"]["name"] == "clouddrift"

    def test_contains_markdown_and_code_cells(self):
        nb = self._load()
        cell_types = {c["cell_type"] for c in nb["cells"]}
        assert "markdown" in cell_types
        assert "code" in cell_types

    def test_code_cells_were_executed(self):
        nb = self._load()
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        executed = [c for c in code_cells if c.get("execution_count") is not None]
        assert len(executed) == len(code_cells), (
            "Not all code cells have been executed — "
            "run Day 7 Step 4 (nbconvert --execute)"
        )

    def test_at_least_one_cell_has_output(self):
        nb = self._load()
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        with_output = [c for c in code_cells if len(c.get("outputs", [])) > 0]
        assert len(with_output) > 0

    def test_mentions_tree_explainer(self):
        nb = self._load()

        def _source(cell: dict) -> str:
            s = cell.get("source", "")
            return "".join(s) if isinstance(s, list) else s

        all_source = " ".join(
            _source(c) for c in nb["cells"] if c["cell_type"] == "code"
        )
        assert "TreeExplainer" in all_source

    def test_mentions_sign_flip_convention(self):
        nb = self._load()

        def _source(cell: dict) -> str:
            s = cell.get("source", "")
            return "".join(s) if isinstance(s, list) else s

        all_source = " ".join(_source(c) for c in nb["cells"])
        assert "flip" in all_source.lower() or "sign" in all_source.lower()


@pytest.mark.skipif(not PLOTS_EXIST, reason="SHAP plots not generated")
class TestShapPlots:
    """Tests for generated SHAP plot images."""

    def test_summary_plot_exists(self):
        assert (PLOTS_DIR / "summary_plot.png").exists()

    def test_at_least_one_waterfall_plot_exists(self):
        waterfalls = list(PLOTS_DIR.glob("waterfall_*.png"))
        assert len(waterfalls) > 0

    def test_plot_files_are_non_empty(self):
        for png in PLOTS_DIR.glob("*.png"):
            assert png.stat().st_size > 1000, f"{png} is suspiciously small"

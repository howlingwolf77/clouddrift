"""
Placeholder test to confirm pytest is wired correctly.
Real tests implemented Days 8-11.
"""


def test_imports():
    """Verify core packages are importable."""
    import fastapi  # noqa: F401
    import lightning  # noqa: F401
    import mlflow  # noqa: F401
    import pandas  # noqa: F401
    import pandera  # noqa: F401
    import sklearn  # noqa: F401
    import streamlit  # noqa: F401
    import torch  # noqa: F401


def test_artifact_utils():
    """Verify artifact utilities import cleanly."""
    from src.utils.artifacts import save_metadata, save_metrics  # noqa: F401


def test_schema_imports():
    """Verify Pydantic schemas import cleanly."""
    from api.schemas.telemetry import AnomalyResponse, TelemetrySnapshot  # noqa: F401

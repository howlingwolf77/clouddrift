"""
Artifact versioning and I/O utilities.
Handles loading and saving model artifacts from the /artifacts directory.
"""

import json
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")


def save_metadata(metadata: dict) -> None:
    """Write metadata.json to the artifacts directory."""
    path = ARTIFACTS_DIR / "metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to {path}")


def save_metrics(metrics: dict) -> None:
    """Write metrics.json to the artifacts directory."""
    path = ARTIFACTS_DIR / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")


def load_metadata() -> dict:
    """Load metadata.json from the artifacts directory."""
    path = ARTIFACTS_DIR / "metadata.json"
    with open(path) as f:
        return json.load(f)

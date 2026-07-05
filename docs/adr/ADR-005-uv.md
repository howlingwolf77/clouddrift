# ADR-005: uv as Package Manager

**Date:** June 2026
**Status:** Accepted
**Author:** Rainel (Ryan) Lobora

## Context

The project required a Python package manager for dependency
resolution, virtual environment management, and reproducible
installs across local development, Docker containers, and GitHub
Actions CI.

## Decision

Selected **uv** over pip + requirements.txt, pip + pyproject.toml,
Poetry, and conda.

## Rationale

| Criterion | uv | pip + req.txt | Poetry |
|-----------|-----|--------------|--------|
| Speed | 10–100× faster | Baseline | ~2× |
| Lockfile | uv.lock (cross-platform) | Partial (pip freeze) | poetry.lock |
| Python management | Built-in (uv python install) | Requires pyenv | Requires pyenv |
| PEP 517/518 | ✓ | ✓ | ✓ |
| Workspace support | ✓ | ✗ | Partial |
| Custom index support | ✓ (pytorch-cpu) | ✓ | Limited |

The custom pytorch-cpu index support was critical for this project:
```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

This ensures the CPU-only PyTorch wheel is installed without pulling
the CUDA version, reducing the Docker image size significantly.

## Shap/llvmlite Resolution Challenge

When `shap` was added to the project, `uv add shap` consistently
resolved to `llvmlite==0.36.0` (Python 3.9 max) when resolving against
the full project graph, even though isolated installs resolved correctly
to `llvmlite==0.47.0`. The fix: pin `numba>=0.61.0` and `llvmlite>=0.44.0`
as direct dependencies, removing old incompatible versions from the
resolver's consideration.

## CI Python Version Challenge

GitHub Actions runners provision Python 3.13.14 while uv.lock was
generated with Python 3.13.12. The `--frozen` flag enforced the exact
version. Resolution: set `UV_PYTHON: "3.13"` in the workflow's global
env block, telling all uv commands to accept any 3.13.x interpreter.

## Consequences

- `uv.lock` commits ensure 100% reproducible installs
- `uv run pytest` and `uv run uvicorn` wrap the venv activation
- `uv sync --frozen` in Dockerfile guarantees the image uses identical
  dependency versions as local development

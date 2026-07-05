# ADR-004: Ruff for Linting and Formatting

**Date:** June 2026
**Status:** Accepted
**Author:** Rainel (Ryan) Lobora

## Context

The project needed a Python linting and formatting solution that
integrates cleanly with uv and GitHub Actions CI. Alternatives
considered: flake8 + black + isort (the traditional stack),
pylint + autopep8, and ruff.

## Decision

Selected **Ruff** as the sole linting and formatting tool, replacing
the traditional flake8 + black + isort stack.

## Rationale

| Criterion | Ruff | flake8 + black + isort |
|-----------|------|----------------------|
| Speed | 10–100× faster | Baseline |
| Configuration | Single `pyproject.toml` | Three separate config files |
| Rule coverage | 800+ rules (includes flake8, isort, pyupgrade) | Requires multiple plugins |
| Auto-fix | ✓ (ruff check --fix) | Partial |
| Format consistency | ✓ (ruff format) | Requires separate black invocation |
| uv integration | Native | Manual |

Ruff's speed advantage compounds in CI — lint check runs in under
10 seconds versus 30–60 seconds for the traditional stack. The single
`pyproject.toml` configuration reduces cognitive overhead and eliminates
the common "flake8 passes but black reformats" mismatch.

## Configuration

Ruff rules are configured in `ruff.toml` (separate from `pyproject.toml`
for clarity). Notable additions to the ignore list: `N803` (argument
names lowercase — conflicts with ML convention `X` for feature matrix),
`N812` (lowercase alias imports — conflicts with `import numpy as np`).

## Consequences

- `uv run ruff check .` and `uv run ruff format --check .` in CI
- All 11 test files and all source files pass ruff checks
- Single config to maintain for the project lifetime

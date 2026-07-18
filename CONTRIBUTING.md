# Contributing to CloudDrift

## Development Setup

```bash
git clone https://github.com/howlingwolf77/clouddrift.git
cd clouddrift
uv sync
uv pip install -e .    # editable install — required for notebook src imports
```

## Branch Strategy

- `main`: stable, CI-verified releases only
- `dev`: active development; open PRs against this branch

## Code Quality

CloudDrift uses Ruff for both linting and formatting (replaces
the traditional flake8 + black + isort stack):

```bash
# Check and auto-fix lint issues
uv run ruff check --fix .

# Format all files
uv run ruff format .

# CI checks (must pass before merging)
uv run ruff check . && uv run ruff format --check .
```

Training scripts (`day4_`, `day5_`, `day6_`, `generate_api_artifacts.py`)
are exempt from E402 (module-level import not at top) via `ruff.toml`
`[lint.per-file-ignores]`, because their imports are intentionally
structured to match pipeline step boundaries.

## Running Tests

```bash
uv run pytest tests/ -q
```

295 tests across 11 files (290 passed, 5 skipped in the standard
configuration). Integration tests requiring model artifacts are guarded
with `@pytest.mark.skipif(not ARTIFACTS_EXIST, ...)` and skip cleanly
without the artifact files. All unit and schema tests run without
any model artifacts.

The 5 skipped tests are intentional — they require NAB parquet feature
matrices from the original pipeline, which the SMD pipeline does not
generate. They are retained as documentation of the original design.

## Adding Dependencies

```bash
uv add <package>           # production dependency
uv add --dev <package>     # development dependency
```

Always commit the updated `uv.lock` alongside `pyproject.toml`.

## Retraining Models

The training pipeline lives in `scripts/` at the project root:

```bash
python scripts/day4_if_training_smd.py    # Isolation Forest (~2 min)
python scripts/day5_tcn_training_smd.py   # TCN Autoencoder (~1.5h on CPU)
python scripts/day6_ensemble_smd.py       # Ensemble scoring and weights
python scripts/generate_api_artifacts.py  # API artifacts and reference stats
```

All scripts must be run from the **project root**, not from inside `scripts/`.
The `src` package imports depend on the working directory being the project root.

After retraining, regenerate `artifacts/metrics.json` with updated
values before committing artifacts. See `docs/DEPLOYMENT_GUIDE.md`
for the generation command.

## Pull Request Checklist

- [ ] All tests pass: `uv run pytest tests/ -q`
- [ ] Ruff clean: `uv run ruff check . && uv run ruff format --check .`
- [ ] New features have corresponding tests
- [ ] Significant architectural decisions documented as ADRs in `docs/adr/`
- [ ] `uv.lock` updated and committed
- [ ] `artifacts/metrics.json` updated if models were retrained

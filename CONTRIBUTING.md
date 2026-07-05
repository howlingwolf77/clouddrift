# Contributing to CloudDrift
 
## Development Setup
 
```bash
git clone https://github.com/howlingwolf77/clouddrift.git
cd clouddrift
uv sync          # installs all dependencies including dev group
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
 
## Running Tests
 
```bash
uv run pytest tests/ -q
```
 
296 tests across 11 files. Integration tests requiring model artifacts
are guarded with `@pytest.mark.skipif(not ARTIFACTS_EXIST, ...)` and
skip cleanly without the artifact files. All unit and schema tests
run without any model artifacts.
 
## Adding Dependencies
 
```bash
uv add <package>           # production dependency
uv add --dev <package>     # development dependency
```
 
Always commit the updated `uv.lock` alongside `pyproject.toml`.
 
## Pull Request Checklist
 
- [ ] All tests pass: `uv run pytest tests/ -q`
- [ ] Ruff clean: `uv run ruff check . && uv run ruff format --check .`
- [ ] New features have corresponding tests
- [ ] Significant architectural decisions documented as ADRs in `docs/adr/`
- [ ] `uv.lock` updated and committed
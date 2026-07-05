# CloudDrift Deployment Guide
 
Two deployment modes: local development and Docker Compose.
For EC2 deployment, see the project README EC2 section.
 
---
 
## Prerequisites
 
- Python 3.13 (managed by uv)
- uv 0.5.26+: https://docs.astral.sh/uv/getting-started/installation/
- Docker + Docker Compose v2 (for containerised deployment)
- Model artifacts in `artifacts/` (run training pipelines Days 4–6
  or contact the project author)
 
---
 
## Local Development
 
```bash
git clone https://github.com/howlingwolf77/clouddrift.git
cd clouddrift
uv sync
```
 
**Start the API:**
```bash
uv run uvicorn api.main:app --reload --port 8000
```
 
**Start the dashboard (separate terminal):**
```bash
uv run streamlit run dashboard/app.py
```
 
**Run tests:**
```bash
uv run pytest tests/ -q
```
 
**Lint check:**
```bash
uv run ruff check . && uv run ruff format --check .
```
 
---
 
## Docker Compose
 
```bash
docker compose up --build
```
 
**Services started:**
- `api`: FastAPI on port 8000
- `dashboard`: Streamlit on port 8501
 
**With Prometheus monitoring:**
```bash
docker compose --profile monitoring up --build
```
Adds `prometheus` on port 9090.
 
**Stop everything:**
```bash
docker compose --profile monitoring down
```
 
---
 
## Artifact Requirements
 
The following files must be present in `artifacts/` before starting:
 
````
isolation_forest.joblib     ~737 KB   IF model
tcn_autoencoder.pt          ~75 KB    TCN model
feature_pipeline.joblib     ~1 KB     NAB normalizer
alibaba_feature_pipeline.joblib  ~3 KB   Alibaba normalizer
thresholds.joblib           ~1 KB     Calibrated thresholds
ensemble_metadata.json      ~2 KB     Weights and bounds
reference_stats.json        ~2 KB     NAB feature statistics
api_reference_stats.json    ~1 KB     Alibaba metric statistics
feature_metadata.json       ~1 KB     Feature column names
metrics.json                ~5 KB     Evaluation metrics
````
 
The `/ready` endpoint confirms all artifacts loaded:
```bash
curl http://localhost:8000/ready | python3 -m json.tool
```
 
---
 
## Environment Variables
 
| Variable | Default | Description |
|----------|---------|-------------|
| `CLOUDDRIFT_API_URL` | `http://localhost:8000` | API URL for Streamlit dashboard |
| `UV_PYTHON` | — | Python version for uv (set to "3.13" in CI) |
| `UV_SYSTEM_PYTHON` | — | Use system Python (set to "1" in CI) |
 
Set in Docker Compose via `environment:` in `compose.yml`.
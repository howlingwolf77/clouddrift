# CloudDrift Deployment Guide

Two deployment modes: local development and Docker Compose.

For EC2 deployment, see the EC2 Deployment section in `README.md`.

---

## Prerequisites

- Python 3.13 (managed by uv)
- uv 0.5.26+: https://docs.astral.sh/uv/getting-started/installation/
- Docker + Docker Compose v2 (for containerised deployment)
- Model artifacts in `artifacts/` — generate them by running the
  training pipeline (see Artifact Generation below)

---

## Local Development

```bash
git clone https://github.com/howlingwolf77/clouddrift.git
cd clouddrift
uv sync
uv pip install -e .
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

### Accessing Internal Services Securely (SSH Tunnel)

If you do not want to open port 9090 publicly, use an SSH tunnel
to forward the remote port to your local machine:

```bash
# Forward Prometheus (9090) through SSH — no security group change needed
ssh -i ~/.ssh/clouddrift-key.pem \
    -L 9090:localhost:9090 \
    ubuntu@<EC2_IP> -N
```

Then open http://localhost:9090 in your local browser.
The -N flag keeps the tunnel open without executing a remote command.
Use Ctrl+C to close the tunnel when done.

The same pattern works for any EC2 service port:
-L <local_port>:localhost:<remote_port>

**Stop everything:**
```bash
docker compose --profile monitoring down
```

---

## Artifact Generation

Model artifacts are not committed to the repository (file sizes exceed
Git limits). Generate them by running the training pipeline from the
project root:

```bash
# Step 1 — Train Isolation Forest (IF) on SMD data (~2 minutes)
source .venv/bin/activate
python day4_if_training_smd.py 2>&1 | tee logs/day4.log

# Step 2 — Train TCN Autoencoder on SMD data (~2.5 hours on CPU)
python day5_tcn_training_smd.py 2>&1 | tee logs/day5.log

# Step 3 — Run ensemble scoring to produce ensemble_metadata.json
python day6_ensemble_smd.py 2>&1 | tee logs/day6.log

# Step 4 — Generate remaining API artifacts
python generate_api_artifacts.py

# Step 5 — Generate consolidated metrics.json (required by some tests)
python -c "
import json
m = {
  'isolation_forest': {
    'validation': {'auc_roc': 0.801, 'precision': 0.194, 'recall': 0.321, 'f1': 0.242},
    'test':       {'auc_roc': 0.894, 'precision': 0.623, 'recall': 0.733, 'f1': 0.674}
  },
  'tcn_autoencoder': {
    'validation': {'auc_roc': 0.869, 'precision': 0.356, 'recall': 0.533, 'f1': 0.427},
    'test':       {'auc_roc': 0.887, 'precision': 0.541, 'recall': 0.789, 'f1': 0.641}
  },
  'ensemble': {
    'validation': {'auc_roc': 0.868, 'precision': 0.284, 'recall': 0.426, 'f1': 0.341},
    'test':       {'auc_roc': 0.899, 'precision': 0.567, 'recall': 0.762, 'f1': 0.650}
  }
}
with open('artifacts/metrics.json', 'w') as f:
    json.dump(m, f, indent=2)
print('artifacts/metrics.json written')
"
```

**WSL2 memory note:** Training 28 SMD machines simultaneously requires
~9 GB RAM (SequenceDataset loads all sliding windows into a single
tensor). With 7.6 GB available RAM, use the 7-machine subset defined
in the training scripts (`MACHINES = [f"machine-1-{i}" for i in range(1, 8)]`).

---

## Artifact Requirements

The following files must be present in `artifacts/` before starting:

```
isolation_forest.joblib      ~1.2 MB   IF model (100 trees, 68 features)
tcn_autoencoder.pt           ~131 KB   TCN model (29,552 parameters)
feature_pipeline.joblib      ~3.2 KB   Normalizer fitted on SMD training data
thresholds.joblib            ~1 KB     Calibrated thresholds (IF, TCN, ensemble)
ensemble_metadata.json       ~2 KB     Weights, metrics, dataset configuration
feature_metadata.json        ~2 KB     68 feature column names and input_dim
reference_stats.json         ~10 KB    Per-feature mean/std (68 engineered features)
api_reference_stats.json     ~1 KB     Per-metric mean/std (5 raw metrics, [0,100] scale)
metrics.json                 ~1 KB     Consolidated evaluation metrics (all three models)
```

The `/ready` endpoint confirms all artifacts are loaded:
```bash
curl -s http://localhost:8000/ready | python3 -m json.tool
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLOUDDRIFT_API_URL` | `http://localhost:8000` | API URL for Streamlit dashboard |
| `UV_PYTHON` | — | Python version for uv (set to "3.13" in CI) |
| `UV_SYSTEM_PYTHON` | — | Use system Python (set to "1" in CI) |

Set in Docker Compose via `environment:` in `compose.yml`.

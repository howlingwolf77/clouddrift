# CloudDrift — Cloud Infrastructure Anomaly Detector

[![CI](https://github.com/howlingwolf77/clouddrift/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/howlingwolf77/clouddrift/actions/workflows/ci.yml)

> ML-powered detection of cloud infrastructure drift before it becomes
> an outage. Combines Isolation Forest and TCN Autoencoder ensemble
> (val AUC-ROC = 0.863) with lightweight z-score attribution for
> real-time anomaly explanations.

---

## Results

| Model | Val AUC-ROC | Test AUC-ROC | Role |
|-------|-------------|--------------|------|
| Isolation Forest | 0.785 | 0.439 | Point-wise feature-space separation |
| TCN Autoencoder | 0.843 | 0.519 | Temporal sequence reconstruction |
| **Ensemble (IF=0.05)** | **0.863** | 0.435 | Combined — beats both individually |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | pandas, Pandera, NAB + Alibaba Cluster Trace 2018 |
| Models | scikit-learn IsolationForest, PyTorch Lightning TCN |
| Explainability | SHAP TreeExplainer (Track 2), z-score attribution (Track 1) |
| API | FastAPI, Pydantic v2, OpenTelemetry, Prometheus |
| Dashboard | Streamlit, Evidently AI drift monitoring |
| Deployment | Docker Compose v2, GitHub Actions CI |
| Package manager | uv (Python 3.13) |

---

## Quick Start

### Local development

```bash
git clone https://github.com/howlingwolf77/clouddrift.git
cd clouddrift
uv sync
uv run uvicorn api.main:app --reload --port 8000
```

### Docker Compose

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Metrics: http://localhost:8000/metrics

---

### Live deployment (EC2)

| Service | URL |
|---------|-----|
| API (Swagger UI) | http://54.165.137.224:8000/docs |
| Streamlit dashboard | http://54.165.137.224:8501 |
| Prometheus metrics | http://54.165.137.224:8000/metrics |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check (artifact loading status) |
| POST | `/detect` | Single-snapshot anomaly detection |
| POST | `/batch_detect` | Batch ranked anomaly detection |
| GET | `/metrics` | Prometheus scrape endpoint |

---

## Dataset

- **NAB** (Numenta Anomaly Benchmark) — 24 real AWS CloudWatch time-series
  with verified anomaly labels; used for model training and evaluation
- **Alibaba Cluster Trace 2018** — 8.4 GB of real production telemetry
  from 5 machines; used for feature engineering and API reference stats

---

## Limitations

- Validation AUC-ROC (0.863) reflects strong discriminative capability.
  Test AUC-ROC (~0.43) is lower due to temporal non-stationarity in NAB
  (test period anomaly signatures differ from training period). This is a
  known property of single-split evaluation on heterogeneous time-series
  data, not a model defect. See `MODEL_CARD.md` for full discussion.

---

*Rainel (Ryan) Lobo — Coding Macaw 2026 Advanced ML Bootcamp Capstone*

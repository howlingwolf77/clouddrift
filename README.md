# CloudDrift — Cloud Infrastructure Anomaly Detector

> ML-powered detection of cloud infrastructure drift before it becomes an outage.

[![CI](https://github.com/howlingwolf77/clouddrift/actions/workflows/ci.yml/badge.svg)](https://github.com/howlingwolf77/clouddrift/actions)

---

## What It Does

CloudDrift monitors cloud infrastructure telemetry (CPU, memory, network I/O, disk)
and detects anomalies before KPI thresholds are breached. It combines two models:

- **Isolation Forest** — fast statistical anomaly scoring
- **TCN Autoencoder** (PyTorch Lightning) — temporal pattern reconstruction

Their scores are combined in a weighted ensemble and served through a FastAPI REST API
with Prometheus metrics, OpenTelemetry tracing, and a Streamlit ops dashboard.

**Dataset:** Real production telemetry from Numenta Anomaly Benchmark and
Alibaba Cluster Trace — not simulated data.

---

## Quick Start (Local)

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Start API (after training — Day 8+)
uv run uvicorn api.main:app --reload

# Start dashboard (Day 10+)
uv run streamlit run dashboard/app.py
```

---

## Quick Start (Docker Compose)

```bash
docker compose up --build
```

API: http://localhost:8000/docs  
Dashboard: http://localhost:8501  
Metrics: http://localhost:8000/metrics

---

## Project Status

🚧 **Sprint Day 1/14** — Repository setup and project scaffolding complete.

---

## Architecture

*Architecture diagram added Day 12.*

---

## Results

*Model evaluation results added Day 7.*

---

## Documentation

- [Technical Specification](TECHNICAL_SPEC.md)
- [API Documentation](API_DOCUMENTATION.md) *(Day 8)*
- [Deployment Guide](DEPLOYMENT_GUIDE.md) *(Day 11)*
- [Model Card](MODEL_CARD.md) *(Day 13)*
- [Monitoring Guide](MONITORING_GUIDE.md) *(Day 10)*
- [Architecture Decisions](docs/adr/) *(Day 13)*

---

## Dataset Sources

- Numenta Anomaly Benchmark (NAB): https://github.com/numenta/NAB
- Alibaba Cluster Trace: https://github.com/alibaba/clusterdata
# CloudDrift Monitoring Guide
 
This guide covers the three monitoring layers in CloudDrift and
explains how to use them together to maintain detection quality
over time.
 
---
 
## Layer 1: Prometheus Metrics (Operational)
 
**Purpose:** Real-time operational visibility — request rate, error
rate, latency, and anomaly detection rate.
 
**Start monitoring:**
```bash
docker compose --profile monitoring up --build
```
 
Open http://localhost:9090 (Prometheus UI).
 
### Useful Queries
 
**Request rate (last 5 minutes):**
````
rate(clouddrift_requests_total[5m])
````
 
**Anomaly detection rate by severity:**
````
rate(clouddrift_anomalies_total[5m])
````
 
**p95 inference latency:**
````
histogram_quantile(0.95, rate(clouddrift_prediction_latency_seconds_bucket[5m]))
````
 
**Schema violation rate (data quality signal):**
````
rate(clouddrift_schema_violations_total[5m])
````
 
### Alerting Thresholds
 
| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| p95 latency | > 200ms | > 500ms | Profile detection service |
| Schema violations | > 5/min | > 20/min | Investigate input source |
| Critical anomalies | > 10/min | > 50/min | Check infrastructure |
 
---
 
## Layer 2: Evidently AI Drift Reports (Data Quality)
 
**Purpose:** Detect when the distribution of incoming telemetry has
drifted from the training reference distribution. A sustained drift
indicates the models may need retraining.
 
**Generate a report:**
 
1. Open the Streamlit dashboard (http://localhost:8501)
2. Send at least 30 telemetry readings using any simulation mode
3. Click **Generate Evidently Drift Report**
4. The report compares your current readings against the Alibaba
   Cluster Trace 2018 training distribution
 
**Interpreting results:**
 
| Column | Status | Meaning |
|--------|--------|---------|
| drift_score | < 0.1 | No significant drift |
| drift_score | 0.1–0.3 | Moderate drift — monitor closely |
| drift_score | > 0.3 | Significant drift — consider retraining |
| drifted | Yes | Wasserstein distance exceeded threshold |
 
**Inline z-score drift table:**
 
The dashboard also shows a lightweight z-score drift table that
updates with every reading (no minimum count required):
- Values < 1.0: stable
- Values 1.0–2.0: elevated — monitor
- Values > 2.0: significant drift — investigate
 
**Reports are saved to** `logs/drift/drift_report_YYYYMMDD_HHMMSS.html`.
Open any saved report in a browser for the full Evidently interactive view.
 
---
 
## Layer 3: SHAP Explainability (Model Behavior)
 
**Purpose:** Understand *why* specific anomalies were flagged and
validate that the model's reasoning matches operational expectations.
 
**Run the SHAP analysis notebook:**
 
```bash
# Register the kernel if not already done
uv run python -m ipykernel install --user \
  --name clouddrift --display-name "CloudDrift (Python 3.13)"
 
# Execute the notebook
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/06_shap_analysis.ipynb
```
 
Or open it interactively in VS Code and run cells manually.
 
**What the notebook produces:**
 
1. **Summary plot** (`artifacts/shap_plots/summary_plot.png`): Global
   feature importance across 1,000 sampled validation rows. Shows which
   rolling features the Isolation Forest relies on most.
 
2. **Waterfall charts** (`artifacts/shap_plots/waterfall_*.png`): One
   per top ensemble-flagged anomaly window. Shows exactly which features
   pushed the IF score toward anomalous.
 
3. **Track 1 vs Track 2 comparison**: Verifies z-score attribution
   (production API) substantially agrees with SHAP (mathematically exact)
   on the same anomaly row — validating Track 1 is trustworthy.
 
---
 
## When to Retrain
 
Retrain the models when any of these signals appear:
 
| Signal | Source | Threshold |
|--------|--------|-----------|
| Evidently drift detected | Drift reports | Any column drifted = True |
| Z-score drift sustained | Streamlit table | Any metric > 2.0 for 24h+ |
| AUC-ROC degradation | Re-evaluation on new data | Drop > 0.05 from baseline |
| Schema violation spike | Prometheus | > 10% of requests over 1h |
| SHAP feature importance shift | Notebook | Top features change significantly |
 
**Retraining procedure (high level):**
 
1. Collect labeled anomaly data from the current period if possible
2. Re-run Days 2–6 pipeline scripts with updated data
3. Compare new ensemble AUC-ROC against baseline (0.863)
4. If improved or equivalent: deploy new artifacts
5. Update `artifacts/api_reference_stats.json` from new Alibaba data
6. Restart containers: `docker compose restart api dashboard`
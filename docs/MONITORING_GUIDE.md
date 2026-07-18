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
```
rate(clouddrift_requests_total[5m])
```

**Anomaly detection rate by severity:**
```
rate(clouddrift_anomalies_total[5m])
```

**p95 inference latency:**
```
histogram_quantile(0.95, rate(clouddrift_prediction_latency_seconds_bucket[5m]))
```

**Schema violation rate (data quality signal):**
```
rate(clouddrift_schema_violations_total[5m])
```

### Alerting Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| p95 latency (`/detect` endpoint) | > 200ms | > 500ms | Profile detection service |
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
4. The report compares your current readings against the SMD training
   distribution (machine-1-1 through machine-1-7, normal-behavior period)

### Two monitoring layers — what they measure differently

| Layer | Method | Measures | Works from |
|-------|--------|----------|------------|
| Z-score table (inline) | Mean deviation | First-order statistics — session mean vs training mean | 1 reading |
| Evidently report | Kolmogorov-Smirnov test | Full distributional shape — every quantile | 30+ readings |

The two layers will sometimes disagree. This is expected and informative:
- All z-scores Stable + Evidently shows drift → session means are normal but
  the *shape* of your readings differs from training. Common with small sessions.
- Z-score shows Drifted + Evidently shows Not Detected → one metric's mean has
  shifted but the overall distribution shape is still similar.

**Interpreting Evidently Drift Score:**

The "Drift Score" column is the **KS p-value**, not a drift magnitude.
A score near 0 means the probability that the two distributions are
identical is essentially zero — the strongest evidence of drift.
A score of 0.14 means no significant difference detected.

| Drift Score (p-value) | Meaning |
|---|---|
| < 0.05 | Drift Detected — distributions are statistically different |
| ≥ 0.05 | Not Detected — distributions are consistent |

**Sample size asymmetry:**

CloudDrift's reference distribution contains ~235,908 training rows.
The KS test becomes increasingly sensitive as the reference grows —
even genuinely normal readings will show drift when your session has
40 points and your reference has 235,000. This is a known statistical
property of the KS test, not a model defect.

Practical guidance:
- Use z-score table for real-time per-reading monitoring
- Accumulate 200+ readings before generating an Evidently report
- Use Evidently for daily/weekly drift assessment, not per-session
- If 3+ columns show drift at p < 0.001 consistently over hundreds of
  readings, that is a genuine retraining signal

**Important:** SMD metrics are in [0, 1] range (pre-normalized).
The API accepts values in [0, 100] (percentage scale). Drift reports
compare against the scaled reference distribution stored in
`artifacts/api_reference_stats.json`. A reading of `cpu_util=45.0`
(45%) is compared against a reference mean of ~30.0% — consistent
with the SMD training data scaled to percentage space.

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
   feature importance across sampled validation rows. Shows which of
   the 68 rolling features the Isolation Forest relies on most.

2. **Waterfall charts** (`artifacts/shap_plots/waterfall_*.png`): One
   per top ensemble-flagged anomaly window. Shows exactly which features
   pushed the IF score toward anomalous.

3. **Track 1 vs Track 3 comparison**: Verifies z-score attribution
   (production API) identifies the same root metric family as SHAP (mathematically exact) —
   metric-level consistency, not feature-level exact agreement
   on the same anomaly row — validating Track 1 is trustworthy for production use.

---

## When to Retrain

Retrain the models when any of these signals appear:

| Signal | Source | Threshold |
|--------|--------|-----------|
| Evidently drift detected | Drift reports | Any column drifted = True |
| Z-score drift sustained | Streamlit table | Any metric > 2.0 for 24h+ |
| AUC-ROC degradation | Re-evaluation on new data | Drop > 0.05 from baseline (0.899) |
| Schema violation spike | Prometheus | > 10% of requests over 1h |
| SHAP feature importance shift | Notebook | Top features change significantly |

**Retraining procedure (high level):**

1. Collect labeled anomaly data from the current period if possible
2. Re-run Days 2–6 pipeline scripts with updated SMD or equivalent data
3. Compare new ensemble AUC-ROC against baseline (0.899 test)
4. If improved or equivalent: deploy new artifacts
5. Regenerate `artifacts/api_reference_stats.json` from new training data:
   ```bash
   python generate_api_artifacts.py
   ```
6. Restart containers: `docker compose restart api dashboard`

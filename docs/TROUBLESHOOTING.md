# CloudDrift Troubleshooting Guide

---

## API Issues

### /ready returns 503

**Symptom:** `{"all_ready": false}` with one or more artifacts marked `false`.

**Cause:** A required artifact file is missing from `artifacts/`.

**Fix:**
```bash
# Check which artifacts are missing
ls artifacts/

# Regenerate all API-required artifacts
source .venv/bin/activate
python generate_api_artifacts.py

# Regenerate metrics.json if missing
python -c "
import json
m = {
  'isolation_forest': {'validation': {'auc_roc': 0.801}, 'test': {'auc_roc': 0.894}},
  'tcn_autoencoder':  {'validation': {'auc_roc': 0.869}, 'test': {'auc_roc': 0.887}},
  'ensemble':         {'validation': {'auc_roc': 0.868}, 'test': {'auc_roc': 0.899}}
}
with open('artifacts/metrics.json', 'w') as f:
    json.dump(m, f, indent=2)
"

# Restart API
docker compose restart api
```

### "No module named 'src'" in notebooks or scripts

**Cause:** The project root is not on the Python path. VS Code Jupyter
kernels do not add the working directory automatically.

**Fix — permanent (recommended):**
```bash
cd ~/projects/clouddrift
uv pip install -e .
```
Then restart the Jupyter kernel (Ctrl+Shift+P → "Jupyter: Restart Kernel").

**Fix — per-notebook (fallback):**
Add this as the first cell in any notebook:
```python
import os, sys
from pathlib import Path
_root = Path.cwd()
while not (_root / "src").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
os.chdir(_root)
```

---

## Training Issues

### WSL2 crashes or OOM during TCN training

**Cause:** `SequenceDataset` loads all 28 SMD machines into a single
tensor simultaneously, requiring ~9 GB RAM. WSL2 default allocation
is 7.6 GB.

**Fix:** Reduce machine count in the training scripts. 7 machines
(the default in day5_tcn_training_smd.py) stays within 3.5 GB:
```python
MACHINES = [f"machine-1-{i}" for i in range(1, 8)]  # 7 machines
```

Confirm available RAM before expanding machine count:
```bash
free -h
```

### TCN trains for 100 epochs without early stopping

**Cause:** `EarlyStopping(min_delta=0.0)` allows infinitesimal
improvements to reset the patience counter. Training runs to `max_epochs`.

**Fix:** Set `min_delta=0.0001` in the `train_tcn_autoencoder()` call:
```python
model, best_ckpt = train_tcn_autoencoder(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    input_dim=input_dim,
    checkpoint_dir=ARTIFACTS_DIR / "checkpoints",
    min_delta=0.0001,
)
```
At 100 epochs with 7 machines, wall time is approximately 1.5 hours on CPU.
The result is a well-converged model (val_loss=0.0016); the only cost
is unnecessary training time.

### NaN loss on TCN epoch 0 (`val_loss = nan`)

**Cause:** `cpu_mem_corr_long` produces NaN normalization bounds in
`RobustPercentileNormalizer` (rolling Pearson correlation is undefined
at series boundaries with fewer than 2 data points). NaN bounds cause
NaN output from the normalizer, which propagates through MSE loss.

**Fix:** The permanent fix is in `src/features/engineering.py`
`RobustPercentileNormalizer.fit()` — NaN/Inf bounds are replaced with
(0.0, 1.0). If the issue recurs, verify the fix is present:
```python
# In fit():
if math.isnan(lo) or math.isnan(hi) or np.isinf(lo) or np.isinf(hi):
    lo, hi = 0.0, 1.0
```

The training scripts also apply a runtime patch after loading the
feature pipeline:
```python
for col in list(normalizer.bounds_.keys()):
    lo, hi = normalizer.bounds_[col]
    if np.isnan(lo) or np.isnan(hi):
        normalizer.bounds_[col] = (0.0, 1.0)
```

---

## Docker Issues

### Docker build fails with OOM

**Cause:** PyTorch download + extraction exceeds available RAM
(common on t2.micro/t3.micro with 1 GB RAM).

**Fix:** Add swap space:
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Dashboard cannot reach API

**Symptom:** Red "API not reachable" banner in Streamlit.

**Cause:** Dashboard container started before API was healthy, or
`CLOUDDRIFT_API_URL` is set incorrectly.

**Fix:**
```bash
# Check API is healthy
docker compose ps | grep api

# Restart dashboard after API is healthy
docker compose restart dashboard
```

---

## CI Issues

### "No interpreter found for Python 3.13.12"

**Cause:** The GitHub runner has a different Python 3.13.x patch
than what is recorded in uv.lock. uv's `--frozen` flag enforces
strict version matching.

**Fix:** Ensure `UV_PYTHON: "3.13"` and `UV_SYSTEM_PYTHON: "1"` are
set in the workflow's global `env:` block. See `docs/adr/ADR-005-uv.md`.

---

## Evidently Issues

### "Report object has no attribute save_html"

**Cause:** Evidently 0.7+ changed the API — `Report.run()` returns
a `Snapshot` object that owns the export methods, not the `Report` itself.

**Fix:** This is handled in `dashboard/drift_monitor.py`:
```python
snapshot = report.run(reference_data=ref, current_data=curr)
snapshot.save_html(str(html_path))
```

### "Need at least 30 readings for a meaningful drift report"

**Cause:** Insufficient session data to compute drift statistics.

**Fix:** Send more readings using the **▶▶ Send 20 readings** button
until you reach 30+ total, then retry.

---

## Pandera Issues

### /detect returns 422 with "schema_validation_failed"

**Cause:** A telemetry value passed Pydantic bounds (0–100) but failed
Pandera's DataFrame schema check.

**Fix:** Check for sentinel values (-1, 101) in the input source.
Check `clouddrift_schema_violations_total` in Prometheus to assess
violation rate. The API accepts values in [0, 100]; the SMD training
reference data is in [0, 1] — `api_reference_stats.json` handles
this scale difference internally.

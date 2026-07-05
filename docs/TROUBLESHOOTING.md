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
# Compare against the required list in DEPLOYMENT_GUIDE.md
# Transfer missing files then restart
docker compose restart api
```
 
### "No module named 'src'" on startup
 
**Cause:** The project root is not on the Python path.
 
**Fix:** Always run commands with `uv run` from the project root, or
set `PYTHONPATH=.` explicitly.
 
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
violation rate.
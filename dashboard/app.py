"""
CloudDrift — Live Anomaly Dashboard

A Streamlit operations dashboard that:
  1. Generates synthetic telemetry in three modes (Normal, Anomaly, Mixed)
  2. Submits each reading to the CloudDrift API /detect endpoint
  3. Displays real-time anomaly scores, severity, contributing metrics
  4. Maintains a rolling session history for trend visualization
  5. Generates Evidently drift reports on demand from session data

Usage:
    uv run streamlit run dashboard/app.py

Prerequisites:
    CloudDrift API running on localhost:8000 (start with:
    uv run uvicorn api.main:app --port 8000)
"""

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Add project root to sys.path so 'dashboard' is importable as a package
# when Streamlit runs this file directly (it adds dashboard/ to sys.path,
# not the project root)
sys.path.insert(0, str(_Path(__file__).parent.parent))

from dashboard.drift_monitor import (
    MIN_CURRENT_ROWS,
    compute_zscore_drift,
    generate_drift_report,
    load_reference_data,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CloudDrift — Anomaly Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("CloudDrift")
st.sidebar.caption("Cloud Infrastructure Anomaly Detector")
st.sidebar.markdown("---")

# CLOUDDRIFT_API_URL env var is set in compose.yml for Docker deployments.
# Falls back to localhost for local development.
_DEFAULT_API_URL = os.environ.get("CLOUDDRIFT_API_URL", "http://localhost:8000")
API_URL = st.sidebar.text_input(
    "API URL",
    value=_DEFAULT_API_URL,
    help="CloudDrift FastAPI service URL. Set CLOUDDRIFT_API_URL env var to override.",
)

st.sidebar.markdown("### Simulation Mode")
MODE = st.sidebar.radio(
    "Telemetry source",
    ["Normal", "Anomaly", "Mixed (50/50)", "Manual input"],
    index=2,
    help=(
        "Normal: typical baseline readings\n"
        "Anomaly: high utilization readings\n"
        "Mixed: 50/50 random\n"
        "Manual: enter values below"
    ),
)

if MODE == "Manual input":
    st.sidebar.markdown("#### Enter telemetry values")
    cpu_manual = st.sidebar.slider("cpu_util (%)", 0.0, 100.0, 41.0)
    mem_manual = st.sidebar.slider("mem_util (%)", 0.0, 100.0, 72.0)
    netin_manual = st.sidebar.slider("net_io_in (%)", 0.0, 100.0, 43.0)
    netout_manual = st.sidebar.slider("net_io_out (%)", 0.0, 100.0, 33.0)

st.sidebar.markdown("---")
MAX_HISTORY = st.sidebar.slider(
    "Session history (readings)",
    min_value=20,
    max_value=500,
    value=100,
    step=10,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: IF + TCN Ensemble | Test AUC-ROC: 0.899 | Weights: IF=0.40, TCN=0.60"
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

if "api_ok" not in st.session_state:
    st.session_state.api_ok = False

if "reference_df" not in st.session_state:
    try:
        st.session_state.reference_df = load_reference_data()
    except Exception:
        st.session_state.reference_df = None


# ---------------------------------------------------------------------------
# Telemetry generation
# ---------------------------------------------------------------------------


def _generate_reading(mode: str) -> dict:
    """Generate one telemetry snapshot in the requested mode."""
    rng = np.random.default_rng()

    if mode == "Anomaly" or (mode == "Mixed (50/50)" and rng.random() < 0.5):
        # High utilization — anomalous
        return {
            "cpu_util": float(np.clip(rng.normal(88, 5), 0, 100)),
            "mem_util": float(np.clip(rng.normal(92, 4), 0, 100)),
            "net_io_in": float(np.clip(rng.normal(85, 8), 0, 100)),
            "net_io_out": float(np.clip(rng.normal(78, 7), 0, 100)),
            "disk_io": float(np.clip(rng.normal(60, 10), 0, 100)),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    else:
        # Baseline normal
        return {
            "cpu_util": float(np.clip(rng.normal(40, 10), 0, 100)),
            "mem_util": float(np.clip(rng.normal(60, 10), 0, 100)),
            "net_io_in": float(np.clip(rng.normal(43, 8), 0, 100)),
            "net_io_out": float(np.clip(rng.normal(33, 6), 0, 100)),
            "disk_io": float(np.clip(rng.normal(10, 8), 0, 100)),
            "timestamp": datetime.now(UTC).isoformat(),
        }


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def _check_ready(api_url: str) -> bool:
    try:
        r = requests.get(f"{api_url}/ready", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _detect(api_url: str, snapshot: dict) -> dict | None:
    try:
        r = requests.post(
            f"{api_url}/detect",
            json=snapshot,
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_SEVERITY_COLOURS = {
    "Critical": "🔴",
    "Warning": "🟡",
    "Normal": "🟢",
}
_SEVERITY_BG = {
    "Critical": "#ff4444",
    "Warning": "#ffaa00",
    "Normal": "#00cc44",
}


# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------

st.title("🔍 CloudDrift — Live Anomaly Dashboard")
st.caption(
    "Real-time cloud infrastructure anomaly detection | "
    "IF + TCN Ensemble (Test AUC-ROC = 0.899)"
)

# ---------------------------------------------------------------------------
# API status banner
# ---------------------------------------------------------------------------

api_status = _check_ready(API_URL)
if api_status:
    st.success(f"✅ API ready at {API_URL}", icon=None)
else:
    st.error(
        f"❌ API not reachable at {API_URL}  |  "
        f"Start with: `uv run uvicorn api.main:app --port 8000`"
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Controls row
# ---------------------------------------------------------------------------

col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 1, 3])

with col_btn1:
    single_shot = st.button(
        "▶  Send one reading",
        disabled=not api_status,
        width="stretch",
    )

with col_btn2:
    burst = st.button(
        "▶▶  Send 20 readings",
        disabled=not api_status,
        width="stretch",
    )

with col_btn3:
    clear_hist = st.button(
        "🗑  Clear history",
        width="stretch",
    )

# ---------------------------------------------------------------------------
# Handle button actions
# ---------------------------------------------------------------------------

if clear_hist:
    st.session_state.history = []
    st.rerun()


def _submit_reading():
    if MODE == "Manual input":
        snap = {
            "cpu_util": cpu_manual,
            "mem_util": mem_manual,
            "net_io_in": netin_manual,
            "net_io_out": netout_manual,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    else:
        snap = _generate_reading(MODE)

    result = _detect(API_URL, snap)
    if result is None:
        st.warning("API call failed — check connection")
        return

    entry = {**snap, **result}
    st.session_state.history.append(entry)
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]


if single_shot:
    _submit_reading()
    st.rerun()

if burst:
    progress = st.progress(0, text="Sending readings...")
    for i in range(20):
        _submit_reading()
        time.sleep(0.05)
        progress.progress((i + 1) / 20, text=f"Sending readings... {i + 1}/20")
    progress.empty()
    st.rerun()

# ---------------------------------------------------------------------------
# KPI metrics
# ---------------------------------------------------------------------------

history = st.session_state.history

if not history:
    st.info(
        "No readings yet. Click **▶ Send one reading** or "
        "**▶▶ Send 20 readings** to begin.",
        icon="ℹ️",
    )
    st.stop()

latest = history[-1]
latest_score = latest.get("anomaly_score", 0.0)
latest_severity = latest.get("severity_label", "Normal")
latest_latency = latest.get("inference_latency_ms", 0.0)
latest_features = latest.get("top_contributing_features", [])
latest_deviations = latest.get("feature_deviation_scores", {})

st.markdown("### Latest Reading")
km1, km2, km3, km4, km5 = st.columns(5)
km1.metric("Anomaly Score", f"{latest_score:.3f}")
km2.metric(
    "Severity", f"{_SEVERITY_COLOURS.get(latest_severity, '')} {latest_severity}"
)
km3.metric("CPU util (%)", f"{latest.get('cpu_util', 0):.1f}")
km4.metric("Mem util (%)", f"{latest.get('mem_util', 0):.1f}")
km5.metric("Latency (ms)", f"{latest_latency:.1f}")

# ---------------------------------------------------------------------------
# Score history chart
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### Anomaly Score History")

history_df = pd.DataFrame(history)
if "anomaly_score" in history_df.columns:
    chart_df = history_df[["timestamp", "anomaly_score"]].copy()
    chart_df["timestamp"] = pd.to_datetime(
        chart_df["timestamp"], errors="coerce", utc=True
    )
    chart_df = chart_df.sort_values("timestamp")
    chart_df.index = range(len(chart_df))

    # Add threshold lines using st.line_chart
    chart_df["Critical threshold"] = 0.8
    chart_df["Warning threshold"] = 0.5

    st.line_chart(
        chart_df.set_index("timestamp")[
            ["anomaly_score", "Critical threshold", "Warning threshold"]
        ],
        color=["#ff4444", "#ffaa00", "#00cc44"],
        height=250,
    )

# ---------------------------------------------------------------------------
# Top contributing features (latest reading)
# ---------------------------------------------------------------------------

col_feat, col_table = st.columns([1, 2])

with col_feat:
    st.markdown("### Contributing Metrics (latest)")
    if latest_deviations:
        feat_df = pd.DataFrame(
            [
                {"Metric": k, "Z-score deviation": v}
                for k, v in sorted(
                    latest_deviations.items(), key=lambda x: x[1], reverse=True
                )
            ]
        )
        st.bar_chart(
            feat_df.set_index("Metric"),
            y="Z-score deviation",
            color="#ff4444" if latest_severity == "Critical" else "#ffaa00",
            height=250,
        )
    else:
        st.info("No feature deviation data available")

with col_table:
    st.markdown("### Recent Readings")
    display_cols = [
        "timestamp",
        "cpu_util",
        "mem_util",
        "anomaly_score",
        "severity_label",
    ]
    available_display = [c for c in display_cols if c in history_df.columns]
    recent_df = history_df[available_display].tail(10).copy()
    if "anomaly_score" in recent_df.columns:
        recent_df["anomaly_score"] = recent_df["anomaly_score"].round(4)
    st.dataframe(recent_df, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Z-score drift (always available once there are readings)
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### Live Drift vs Training Distribution (Z-score)")
st.caption(
    "Shows how far the current session's mean readings have drifted from "
    "the Alibaba training distribution. Values > 2.0 indicate meaningful drift."
)

snapshot_dicts = [
    {
        c: row.get(c)
        for c in ["cpu_util", "mem_util", "net_io_in", "net_io_out", "disk_io"]
    }
    for row in history
]
zscore_drift = compute_zscore_drift(snapshot_dicts)

if zscore_drift:
    drift_df = pd.DataFrame(
        [
            {
                "Metric": k,
                "Drift (|z-score|)": v,
                "Status": "⚠️ Drifted" if v >= 2.0 else "✅ Stable",
            }
            for k, v in sorted(zscore_drift.items(), key=lambda x: x[1], reverse=True)
        ]
    )
    st.dataframe(drift_df, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Evidently drift report (on demand)
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### Evidently Data Drift Report (on demand)")
n_current = len(snapshot_dicts)
n_needed = MIN_CURRENT_ROWS

if n_current < n_needed:
    st.info(
        f"Need at least {n_needed} readings to generate an Evidently report. "
        f"You have {n_current}. Send more readings then click below."
    )
    st.button("Generate Evidently Drift Report", disabled=True)
else:
    if st.button(f"Generate Evidently Drift Report ({n_current} readings)"):
        with st.spinner("Running Evidently DataDriftPreset..."):
            try:
                html_path, summary = generate_drift_report(
                    snapshot_dicts,
                    reference_df=st.session_state.reference_df,
                )
                st.success(f"Report saved: `{html_path}`")

                # Display summary table
                if summary.get("columns"):
                    st.markdown(
                        f"**Dataset drifted:** "
                        f"{'Yes ⚠️' if summary['dataset_drifted'] else 'No ✅'}  |  "
                        f"**Drifted columns:** {summary['n_drifted']}"
                    )
                    col_summary = pd.DataFrame(
                        [
                            {
                                "Metric": col,
                                "Drift score": vals["drift_score"],
                                "Drifted": "⚠️ Yes" if vals["drifted"] else "✅ No",
                                "Test": vals["stattest"],
                            }
                            for col, vals in summary["columns"].items()
                        ]
                    )
                    st.dataframe(col_summary, width="stretch", hide_index=True)

                # Embed the HTML report inline
                st.markdown("#### Full Evidently Report")
                st.iframe(html_path, height=600, scrolling=True)

            except Exception as exc:
                st.error(f"Failed to generate drift report: {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    f"Session: {len(history)} readings | "
    f"Critical: {sum(1 for h in history if h.get('severity_label') == 'Critical')} | "
    f"Warning: {sum(1 for h in history if h.get('severity_label') == 'Warning')} | "
    f"Normal: {sum(1 for h in history if h.get('severity_label') == 'Normal')}"
)

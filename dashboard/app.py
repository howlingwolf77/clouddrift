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
import streamlit.components.v1 as components

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
# Constants — centralized so tuning requires one edit, not a file-wide search
# ---------------------------------------------------------------------------

CRITICAL_THRESHOLD = 0.75  # anomaly score ≥ this → Critical (3σ convention)
WARNING_THRESHOLD = 0.50  # anomaly score ≥ this → Warning
BURST_SIZE = 20  # readings sent by "Send 20 readings" button
API_TIMEOUT = 5  # seconds for /detect calls
READY_TIMEOUT = 3  # seconds for /ready liveness calls

# Synthetic telemetry parameters — normal baseline
NORMAL_CPU_MEAN, NORMAL_CPU_STD = 40, 10
NORMAL_MEM_MEAN, NORMAL_MEM_STD = 60, 10
NORMAL_NET_IN_MEAN, NORMAL_NET_IN_STD = 43, 8
NORMAL_NET_OUT_MEAN, NORMAL_NET_OUT_STD = 33, 6
NORMAL_DISK_MEAN, NORMAL_DISK_STD = 10, 8

# Synthetic telemetry parameters — anomalous high-utilization
ANOMALY_CPU_MEAN, ANOMALY_CPU_STD = 88, 5
ANOMALY_MEM_MEAN, ANOMALY_MEM_STD = 92, 4
ANOMALY_NET_IN_MEAN, ANOMALY_NET_IN_STD = 85, 8
ANOMALY_NET_OUT_MEAN, ANOMALY_NET_OUT_STD = 78, 7
ANOMALY_DISK_MEAN, ANOMALY_DISK_STD = 60, 10


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
# Severity helpers — defined early so they can be reused throughout
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
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("CloudDrift")
st.sidebar.caption("Cloud Infrastructure Anomaly Detector")
st.sidebar.markdown("---")

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

# machine_id_manual initialised here (outside the Manual block) so all code
# paths can reference it safely — fixes NameError in _generate_reading().
machine_id_manual: str | None = None

if MODE == "Manual input":
    st.sidebar.markdown("#### Enter telemetry values")
    cpu_manual = st.sidebar.slider("cpu_util (%)", 0.0, 100.0, 41.0)
    mem_manual = st.sidebar.slider("mem_util (%)", 0.0, 100.0, 72.0)
    netin_manual = st.sidebar.slider("net_io_in (%)", 0.0, 100.0, 43.0)
    netout_manual = st.sidebar.slider("net_io_out (%)", 0.0, 100.0, 33.0)
    disk_manual = st.sidebar.slider("disk_io (%)", 0.0, 100.0, 5.0)
    machine_id_manual = (
        st.sidebar.text_input("machine_id (optional)", "machine-1-1") or None
    )

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
    "Track 1: z-score (/detect) | Track 2: IF+TCN ensemble (/batch_detect ≥30 snapshots with same machine_id) | AUC-ROC: 0.899"
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

if "reference_df" not in st.session_state:
    try:
        st.session_state.reference_df = load_reference_data()
    except Exception:
        st.session_state.reference_df = None


# ---------------------------------------------------------------------------
# Telemetry generation
# ---------------------------------------------------------------------------


def _generate_reading(mode: str, machine_id: str | None = None) -> dict:
    """
    Generate one synthetic telemetry snapshot.

    Args:
        mode:       "Normal", "Anomaly", or "Mixed (50/50)"
        machine_id: Optional machine identifier included in the payload.

    Returns:
        Dict matching the TelemetrySnapshot schema.
    """
    rng = np.random.default_rng()
    is_anomaly = mode == "Anomaly" or (mode == "Mixed (50/50)" and rng.random() < 0.5)

    if is_anomaly:
        snap = {
            "cpu_util": float(
                np.clip(rng.normal(ANOMALY_CPU_MEAN, ANOMALY_CPU_STD), 0, 100)
            ),
            "mem_util": float(
                np.clip(rng.normal(ANOMALY_MEM_MEAN, ANOMALY_MEM_STD), 0, 100)
            ),
            "net_io_in": float(
                np.clip(rng.normal(ANOMALY_NET_IN_MEAN, ANOMALY_NET_IN_STD), 0, 100)
            ),
            "net_io_out": float(
                np.clip(rng.normal(ANOMALY_NET_OUT_MEAN, ANOMALY_NET_OUT_STD), 0, 100)
            ),
            "disk_io": float(
                np.clip(rng.normal(ANOMALY_DISK_MEAN, ANOMALY_DISK_STD), 0, 100)
            ),
        }
    else:
        snap = {
            "cpu_util": float(
                np.clip(rng.normal(NORMAL_CPU_MEAN, NORMAL_CPU_STD), 0, 100)
            ),
            "mem_util": float(
                np.clip(rng.normal(NORMAL_MEM_MEAN, NORMAL_MEM_STD), 0, 100)
            ),
            "net_io_in": float(
                np.clip(rng.normal(NORMAL_NET_IN_MEAN, NORMAL_NET_IN_STD), 0, 100)
            ),
            "net_io_out": float(
                np.clip(rng.normal(NORMAL_NET_OUT_MEAN, NORMAL_NET_OUT_STD), 0, 100)
            ),
            "disk_io": float(
                np.clip(rng.normal(NORMAL_DISK_MEAN, NORMAL_DISK_STD), 0, 100)
            ),
        }

    snap["timestamp"] = datetime.now(UTC).isoformat()
    snap["machine_id"] = machine_id
    return snap


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def _check_ready(api_url: str) -> bool:
    try:
        r = requests.get(f"{api_url}/ready", timeout=READY_TIMEOUT)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _detect(api_url: str, snapshot: dict) -> dict | None:
    try:
        r = requests.post(
            f"{api_url}/detect",
            json=snapshot,
            timeout=API_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------

st.title("🔍 CloudDrift — Live Anomaly Dashboard")
st.caption(
    "Real-time cloud infrastructure anomaly detection | "
    "z-score inference | Ensemble AUC-ROC = 0.899 (batch evaluation)"
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
        f"▶▶  Send {BURST_SIZE} readings",
        disabled=not api_status,
        width="stretch",
    )

with col_btn3:
    if st.button("🗑  Clear history", width="stretch"):
        st.session_state.history = []
        st.rerun()


def _submit_reading() -> bool:
    """
    Build one telemetry snapshot, call /detect, and append to session history.

    Returns True on success, False if the API call failed.
    """
    if MODE == "Manual input":
        snap = {
            "cpu_util": cpu_manual,
            "mem_util": mem_manual,
            "net_io_in": netin_manual,
            "net_io_out": netout_manual,
            "disk_io": disk_manual,
            "timestamp": datetime.now(UTC).isoformat(),
            "machine_id": machine_id_manual,
        }
    else:
        snap = _generate_reading(MODE, machine_id=machine_id_manual)

    result = _detect(API_URL, snap)
    if result is None:
        return False

    entry = {**snap, **result}
    st.session_state.history.append(entry)
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]
    return True


if single_shot:
    if not _submit_reading():
        st.warning("API call failed — check connection")
    st.rerun()

if burst:
    progress = st.progress(0, text="Sending readings...")
    succeeded = 0
    failed = 0
    for i in range(BURST_SIZE):
        if _submit_reading():
            succeeded += 1
        else:
            failed += 1
        time.sleep(0.05)
        progress.progress(
            (i + 1) / BURST_SIZE,
            text=f"Sending readings... {i + 1}/{BURST_SIZE}",
        )
    progress.empty()
    if failed > 0:
        st.warning(
            f"{succeeded} readings succeeded, {failed} failed — check API connection"
        )
    st.rerun()

# ---------------------------------------------------------------------------
# KPI metrics
# ---------------------------------------------------------------------------

history = st.session_state.history

if not history:
    st.info(
        "No readings yet. Click **▶ Send one reading** or "
        f"**▶▶ Send {BURST_SIZE} readings** to begin.",
        icon="ℹ️",
    )
    st.stop()

latest = history[-1]
latest_score = latest.get("anomaly_score", 0.0)
latest_severity = latest.get("severity_label", "Normal")
latest_latency = latest.get("inference_latency_ms", 0.0)
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

    chart_df["Critical threshold"] = CRITICAL_THRESHOLD
    chart_df["Warning threshold"] = WARNING_THRESHOLD

    # Color order matches column order:
    #   anomaly_score    → #4fc3f7 (light blue — neutral score line)
    #   Critical threshold → #ff4444 (red)
    #   Warning threshold  → #ffaa00 (orange)
    st.line_chart(
        chart_df.set_index("timestamp")[
            ["anomaly_score", "Critical threshold", "Warning threshold"]
        ],
        color=["#4fc3f7", "#ff4444", "#ffaa00"],
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
        # Reuse _SEVERITY_BG so bar color stays in sync with severity thresholds
        bar_color = _SEVERITY_BG.get(latest_severity, _SEVERITY_BG["Warning"])
        st.bar_chart(
            feat_df.set_index("Metric"),
            y="Z-score deviation",
            color=bar_color,
            height=250,
        )
    else:
        st.info("No feature deviation data available")

with col_table:
    st.markdown("### Recent Readings")
    display_cols = [
        "timestamp",
        "machine_id",
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
    "the SMD training distribution. Values > 2.0 indicate meaningful drift."
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

                st.markdown("#### Full Evidently Report")
                with open(html_path, encoding="utf-8") as _f:
                    components.html(_f.read(), height=650, scrolling=True)

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

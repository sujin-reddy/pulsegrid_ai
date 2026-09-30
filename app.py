"""PulseGrid AI — Live Stream & Autonomous Self-Healing Infrastructure Dashboard.

Real-time streaming telemetry with LangGraph multi-agent detection, diagnosis,
and preventive intervention. Features live charting, spike injection, and
persistent SQLite memory inspection.
"""

from __future__ import annotations

import random
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from db_memory import DB_PATH, init_db
from main import run_pipeline
from replay import replay_csv_stream, EXAMPLE_CSV_CONTENT

# Ensure database is initialized
init_db()

# Page Configuration
st.set_page_config(
    page_title="PulseGrid AI — Live Stream",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-alert {
        color: #ef4444;
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-ok {
        color: #10b981;
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .info-card {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .action-card {
        background: #1e293b;
        border-left: 4px solid #10b981;
        padding: 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_raw_database_entries() -> list[dict]:
    """Fetch all raw records directly from interventions.db."""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, anomaly_type, action_taken, outcome_score, timestamp "
                "FROM intervention_memory ORDER BY id DESC"
            )
            return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return []


def generate_telemetry(spike: bool = False) -> dict:
    """Generate a realistic telemetry reading with optional load spike."""
    if spike:
        val = random.randint(85, 98)
        voltage = round(random.uniform(208.0, 216.0), 1)
        temperature = round(random.uniform(82.0, 92.0), 1)
        load_mw = round(random.uniform(168.0, 178.0), 1)
        freq = round(random.uniform(59.65, 59.78), 2)
        pf = round(random.uniform(0.78, 0.88), 2)
    else:
        val = random.randint(50, 78)
        voltage = round(random.uniform(225.0, 235.0), 1)
        temperature = round(random.uniform(38.0, 55.0), 1)
        load_mw = round(random.uniform(90.0, 145.0), 1)
        freq = round(random.uniform(59.92, 60.08), 2)
        pf = round(random.uniform(0.93, 0.98), 2)

    return {
        "substation_id": f"SUB-{random.choice(['NORTH', 'SOUTH', 'EAST', 'WEST', 'CENTRAL'])}-{random.randint(1, 12):02d}",
        "value": float(val),
        "voltage_kv": voltage,
        "temperature_c": temperature,
        "load_mw": load_mw,
        "frequency_hz": freq,
        "power_factor": pf,
    }


# Title
st.title("⚡ PulseGrid AI — Live Stream & Autonomous Self-Healing")
st.caption("Real-time LangGraph Agent Orchestration • Qwen 2.5 Diagnostics • SQLite Persistent Memory")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["Timestamp", "Sensor_Value", "Status", "Substation"]
    )
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "spike_requested" not in st.session_state:
    st.session_state.spike_requested = False

# Sidebar Controls
st.sidebar.header("🎛️ Stream Controls")
streaming = st.sidebar.toggle("Start Live Telemetry Stream", value=False)
refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 1, 5, 2)

if st.sidebar.button("💥 Inject Artificial Load Spike (>80)", use_container_width=True):
    st.session_state.spike_requested = True

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Manual One-Shot Simulation")
manual_val = st.sidebar.slider("Manual Sensor Value", 0, 100, 85)
if st.sidebar.button("🚀 Run Single Pipeline", use_container_width=True):
    manual_telemetry = generate_telemetry(spike=(manual_val > 80))
    manual_telemetry["value"] = float(manual_val)
    with st.spinner("Executing pipeline..."):
        res = run_pipeline(manual_telemetry)
        st.session_state.last_result = res
        now = pd.Timestamp.now().strftime("%H:%M:%S")
        status_label = "CRITICAL" if res.get("anomaly_detected") else "NORMAL"
        new_row = pd.DataFrame(
            [{"Timestamp": now, "Sensor_Value": manual_val, "Status": status_label, "Substation": manual_telemetry["substation_id"]}]
        )
        st.session_state.history = pd.concat([st.session_state.history, new_row]).tail(30)
    st.sidebar.success("Pipeline executed!")

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Anomaly Threshold:** `> 80`  \n"
    f"**History Buffer:** Last {len(st.session_state.history)} readings  \n"
    f"**Database Path:** `interventions.db`"
)

# ── CSV Replay Section ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📂 CSV Replay")
st.sidebar.caption("Upload a telemetry CSV to replay through the pipeline.")

uploaded_csv = st.sidebar.file_uploader(
    "Upload telemetry CSV",
    type=["csv"],
    help="Columns: domain, device_id, + any numeric metric columns",
    key="csv_uploader",
)

if st.sidebar.button("⬇️ Download Example CSV", use_container_width=True):
    st.sidebar.download_button(
        label="📥 example_telemetry.csv",
        data=EXAMPLE_CSV_CONTENT,
        file_name="example_telemetry.csv",
        mime="text/csv",
        use_container_width=True,
    )

if uploaded_csv is not None:
    if st.sidebar.button("▶️ Run CSV Replay", use_container_width=True):
        import tempfile, os  # noqa: E401
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
            tmp.write(uploaded_csv.read())
            tmp_path = tmp.name
        st.subheader("📂 CSV Replay Results")
        replay_placeholder = st.empty()
        replay_rows = []
        try:
            for result in replay_csv_stream(tmp_path, run_pipeline):
                replay_rows.append(result)
                replay_placeholder.dataframe(
                    pd.DataFrame(replay_rows)[
                        ["row_index", "domain", "device_id", "anomaly_detected",
                         "resolved_domain", "action", "cost", "latency_ms"]
                    ],
                    use_container_width=True,
                )
        finally:
            os.unlink(tmp_path)
        st.success(f"Replay complete — {len(replay_rows)} events processed.")

# Live chart placeholder
chart_placeholder = st.empty()

# 3-Column layout
col1, col2, col3 = st.columns(3)

# Live Streaming Loop
if streaming:
    st.toast("📡 Telemetry Stream Active...", icon="📡")

    # Determine if this tick should be a spike
    is_spike = st.session_state.spike_requested
    if is_spike:
        st.session_state.spike_requested = False

    telemetry = generate_telemetry(spike=is_spike)

    # Run LangGraph pipeline
    result_state = run_pipeline(telemetry)
    st.session_state.last_result = result_state

    # Append to history
    now = pd.Timestamp.now().strftime("%H:%M:%S")
    sensor_val = telemetry["value"]
    status_label = "CRITICAL" if result_state.get("anomaly_detected") else "NORMAL"
    new_row = pd.DataFrame(
        [{"Timestamp": now, "Sensor_Value": sensor_val, "Status": status_label, "Substation": telemetry["substation_id"]}]
    )
    st.session_state.history = pd.concat([st.session_state.history, new_row]).tail(30)

    # Update live chart
    with chart_placeholder.container():
        st.subheader("📈 Real-Time Telemetry Feed")
        if len(st.session_state.history) > 1:
            chart_df = st.session_state.history.set_index("Timestamp")["Sensor_Value"].astype(float)
            st.line_chart(chart_df)
        else:
            st.info("Gathering first data points...")

    # Update 3 columns
    with col1:
        st.subheader("1️⃣ Telemetry & Detection")
        delta_color = "inverse" if result_state.get("anomaly_detected") else "normal"
        st.metric(
            "Current Sensor Value",
            f"{sensor_val:.0f}",
            delta=status_label,
            delta_color=delta_color,
        )
        st.metric("Substation", telemetry["substation_id"])
        st.metric("Voltage", f"{telemetry['voltage_kv']:.1f} kV")
        st.metric("Frequency", f"{telemetry['frequency_hz']:.2f} Hz")
        st.metric("Load", f"{telemetry['load_mw']:.1f} MW")
        st.metric("Temperature", f"{telemetry['temperature_c']:.1f} C")

    with col2:
        st.subheader("2️⃣ Root Cause Analysis")
        if result_state.get("anomaly_detected"):
            st.error(f"**AI Diagnosis (Qwen 2.5):**\n\n{result_state.get('root_cause', 'Diagnosing...')}")
        else:
            st.success("Grid Operating Within Normal Thresholds")

    with col3:
        st.subheader("3️⃣ Preventive Action Plan")
        if result_state.get("anomaly_detected"):
            proposed = result_state.get("proposed_intervention", {})
            st.markdown(
                f"""
                <div class="action-card">
                    <p><b>Action:</b> {proposed.get('action', 'N/A')}</p>
                    <p><b>Cost:</b> ${proposed.get('cost', 0):,.2f}</p>
                    <p><b>Impact:</b> {proposed.get('expected_impact', 'N/A')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            past = result_state.get("past_interventions", [])
            if past:
                st.markdown("**Retrieved Memory Precedents:**")
                df_past = pd.DataFrame(past)
                display_cols = [c for c in ["anomaly_type", "action_taken", "outcome_score"] if c in df_past.columns]
                if display_cols:
                    st.dataframe(df_past[display_cols], use_container_width=True)
        else:
            st.info("No Intervention Required")

    # Sleep then rerun for next tick
    time.sleep(refresh_rate)
    st.rerun()

# When NOT streaming, show last result if available
elif st.session_state.last_result is not None:
    res = st.session_state.last_result

    with chart_placeholder.container():
        st.subheader("📈 Telemetry History")
        if len(st.session_state.history) > 0:
            chart_df = st.session_state.history.set_index("Timestamp")["Sensor_Value"].astype(float)
            st.line_chart(chart_df)
        else:
            st.info("No telemetry data yet. Start a stream or run a manual simulation.")

    with col1:
        st.subheader("1️⃣ Telemetry & Detection")
        anomaly = res.get("anomaly_detected", False)
        tel = res.get("telemetry_data", {})
        st.metric("Last Sensor Value", f"{tel.get('value', 0):.0f}", delta="CRITICAL" if anomaly else "NORMAL", delta_color="inverse" if anomaly else "normal")
        st.json(tel)

    with col2:
        st.subheader("2️⃣ Root Cause Analysis")
        if res.get("anomaly_detected"):
            st.error(f"**Diagnosis:**\n\n{res.get('root_cause', 'N/A')}")
        else:
            st.success("All parameters nominal.")

    with col3:
        st.subheader("3️⃣ Preventive Action")
        if res.get("anomaly_detected"):
            proposed = res.get("proposed_intervention", {})
            st.markdown(
                f"""
                <div class="action-card">
                    <p><b>Action:</b> {proposed.get('action', 'N/A')}</p>
                    <p><b>Cost:</b> ${proposed.get('cost', 0):,.2f}</p>
                    <p><b>Impact:</b> {proposed.get('expected_impact', 'N/A')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No Intervention Required")

else:
    with chart_placeholder.container():
        st.info("Toggle **Start Live Telemetry Stream** in the sidebar, or run a **Manual One-Shot Simulation** to begin.")

# Divider
st.divider()

# Raw SQLite Database Expander
with st.expander("🗄️ Raw SQLite Database Entries (interventions.db)", expanded=False):
    st.markdown("Direct read of persistent table `intervention_memory`:")
    raw_records = get_raw_database_entries()
    if raw_records:
        df_raw = pd.DataFrame(raw_records)
        st.dataframe(
            df_raw[["id", "anomaly_type", "action_taken", "outcome_score", "timestamp"]],
            use_container_width=True,
        )
        st.caption(f"Total records in SQLite: **{len(raw_records)}**")
    else:
        st.info("No records present in interventions.db.")

# Telemetry History Table
with st.expander("📋 Session Telemetry History Log", expanded=False):
    if len(st.session_state.history) > 0:
        st.dataframe(st.session_state.history, use_container_width=True)
    else:
        st.info("No telemetry readings captured yet.")

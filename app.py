"""PulseGrid AI — Autonomous Self-Healing Infrastructure Dashboard.

Interactive Streamlit interface showcasing real-time agent workflow execution,
local LLM root-cause analysis, and persistent SQLite memory.
"""

from __future__ import annotations

import sqlite3
import pandas as pd
import streamlit as st

from db_memory import DB_PATH, init_db
from main import run_pipeline

# Ensure database is initialized
init_db()

# Page Configuration
st.set_page_config(
    page_title="PulseGrid AI — Autonomous Self-Healing Infrastructure",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Control Room Aesthetics
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
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, anomaly_type, action_taken, outcome_score, timestamp FROM intervention_memory ORDER BY id DESC")
        return [dict(row) for row in cursor.fetchall()]


# Header
st.title("⚡ PulseGrid AI — Autonomous Self-Healing Infrastructure")
st.caption("LangGraph Multi-Agent Orchestration • Local Qwen 2.5 Reasoner • SQLite Persistent Memory")

# Sidebar Controls
st.sidebar.header("🎛️ Telemetry Input Parameters")
st.sidebar.markdown("Configure incoming sensor stream for agent monitoring:")

sensor_val = st.sidebar.slider(
    "Sensor Value (Severity Metric)",
    min_value=0,
    max_value=100,
    value=85,
    help="Threshold is 80. Values > 80 trigger an automated anomaly investigation.",
)

anomaly_type = st.sidebar.selectbox(
    "Simulated Anomaly Category",
    [
        "thermal_overload",
        "voltage_sag",
        "frequency_decay",
        "transformer_overheat",
        "reactive_power_deficit",
    ],
    help="Select the specific grid physical disturbance pattern.",
)

substation_id = st.sidebar.text_input("Substation Identifier", value="SUB-CENTRAL-04")

# Map anomaly category to realistic physical parameters
if anomaly_type == "thermal_overload":
    voltage = 226.5
    temperature = 84.0 if sensor_val > 80 else 48.0
    load_mw = 172.0 if sensor_val > 80 else 125.0
    freq = 59.98
    pf = 0.93
elif anomaly_type == "voltage_sag":
    voltage = 212.0 if sensor_val > 80 else 230.2
    temperature = 52.0
    load_mw = 145.0
    freq = 59.96
    pf = 0.82 if sensor_val > 80 else 0.95
elif anomaly_type == "frequency_decay":
    voltage = 224.0
    temperature = 50.0
    load_mw = 158.0
    freq = 59.72 if sensor_val > 80 else 60.01
    pf = 0.91
elif anomaly_type == "transformer_overheat":
    voltage = 229.0
    temperature = 88.0 if sensor_val > 80 else 46.0
    load_mw = 160.0
    freq = 60.00
    pf = 0.94
else:
    voltage = 222.0
    temperature = 55.0
    load_mw = 140.0
    freq = 59.95
    pf = 0.81 if sensor_val > 80 else 0.96

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Threshold:** `80.0`")
st.sidebar.markdown(
    f"**Status Preview:** "
    + ("<span class='badge-alert'>ANOMALOUS (>80)</span>" if sensor_val > 80 else "<span class='badge-ok'>NOMINAL (<=80)</span>"),
    unsafe_allow_html=True,
)

run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Session state management for simulation results
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if run_simulation:
    input_telemetry = {
        "substation_id": substation_id,
        "value": float(sensor_val),
        "anomaly_type": anomaly_type,
        "voltage_kv": voltage,
        "temperature_c": temperature,
        "load_mw": load_mw,
        "frequency_hz": freq,
        "power_factor": pf,
    }

    with st.spinner("⚡ Executing PulseGrid Multi-Agent Pipeline (Monitor ➔ Investigate ➔ Intervene)..."):
        res = run_pipeline(input_telemetry)
        st.session_state.pipeline_result = res
    st.success("Simulation completed successfully!")

# Display Results across 3 Columns
if st.session_state.pipeline_result is not None:
    res = st.session_state.pipeline_result
    anomaly_detected = res.get("anomaly_detected", False)
    status = res.get("status", "complete")
    root_cause = res.get("root_cause", "No anomaly identified.")
    proposed = res.get("proposed_intervention", {})
    past_actions = res.get("past_interventions", [])
    messages = res.get("messages", [])

    col1, col2, col3 = st.columns(3)

    # -------------------------------------------------------------
    # Column 1: Telemetry & Detection Status
    # -------------------------------------------------------------
    with col1:
        st.subheader("1️⃣ Telemetry & Detection")
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Sensor Metric: {sensor_val} / 100</h3>
                <p>Threshold Limit: <b>80.0</b></p>
                <p>Detection Status: {'<span class="badge-alert">⚠️ ANOMALY DETECTED</span>' if anomaly_detected else '<span class="badge-ok">✅ NOMINAL OPERATION</span>'}</p>
                <p>Workflow Phase: <code>{status.upper()}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Real-Time Telemetry Payload:**")
        st.json(res.get("telemetry_data", {}))

    # -------------------------------------------------------------
    # Column 2: Root Cause Analysis & Evidence
    # -------------------------------------------------------------
    with col2:
        st.subheader("2️⃣ Root Cause Analysis")
        if anomaly_detected:
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>🔍 AI Diagnostic Reasoning (Qwen 2.5)</h4>
                    <p>{root_cause}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Detection & Diagnostic Logs:**")
            for msg in messages:
                st.code(str(msg), language="markdown")
        else:
            st.info("✅ All metrics within normal operating envelope. No root-cause investigation required.")

    # -------------------------------------------------------------
    # Column 3: Proposed Action & Historical Memory
    # -------------------------------------------------------------
    with col3:
        st.subheader("3️⃣ Intervention & Memory")
        if anomaly_detected:
            action_name = proposed.get("action", "Automatic reactive adjustment")
            cost_val = proposed.get("cost", 450.0)
            impact_desc = proposed.get("expected_impact", "Parameter stabilization within 60s")

            st.markdown(
                f"""
                <div class="action-card">
                    <h4>⚡ Proposed Preventive Action</h4>
                    <p><b>Action:</b> {action_name}</p>
                    <p><b>Est. Cost:</b> ${cost_val:,.2f}</p>
                    <p><b>Expected Impact:</b> {impact_desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**🧠 Retrieved Memory Precedents (Score ≥ 0.7):**")
            if past_actions:
                df_past = pd.DataFrame(past_actions)
                display_cols = [c for c in ["anomaly_type", "action_taken", "outcome_score"] if c in df_past.columns]
                st.dataframe(df_past[display_cols], use_container_width=True)
            else:
                st.caption("No prior memory records found matching criteria.")
        else:
            st.success("System healthy. Preventive action standby.")

st.divider()

# -------------------------------------------------------------
# Expandable Section: Raw SQLite Database Entries
# -------------------------------------------------------------
with st.expander("🗄️ Raw SQLite Database Entries (interventions.db)", expanded=False):
    st.markdown("Direct read of persistent table `intervention_memory`:")
    raw_records = get_raw_database_entries()
    if raw_records:
        df_raw = pd.DataFrame(raw_records)
        st.dataframe(
            df_raw[["id", "anomaly_type", "action_taken", "outcome_score", "timestamp"]],
            use_container_width=True,
        )
        st.caption(f"Total historical intervention records in SQLite: {len(raw_records)}")
    else:
        st.info("No records present in interventions.db.")

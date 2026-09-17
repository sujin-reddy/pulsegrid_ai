"""PulseGrid AI — Streamlit Operational Dashboard.

Interactive control room for monitoring grid telemetry, triggering LangGraph
anomaly workflows, inspecting local LLM (Qwen 2.5) reasoning, and executing interventions.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from db_memory import get_past_interventions, init_db
from graph import run_pulsegrid_workflow

# Initialize DB on startup
init_db()

# Page configuration
st.set_page_config(
    page_title="PulseGrid AI — Grid Resilience Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern dark-mode aesthetic
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
    }
    .metric-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .badge-normal {
        color: #10b981;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.15);
        padding: 4px 10px;
        border-radius: 6px;
    }
    .badge-alert {
        color: #ef4444;
        font-weight: 700;
        background: rgba(239, 68, 68, 0.15);
        padding: 4px 10px;
        border-radius: 6px;
    }
    .agent-card {
        border-left: 4px solid #3b82f6;
        background: #1e293b;
        padding: 16px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Telemetry controls
st.sidebar.title("⚡ PulseGrid AI Control")
st.sidebar.markdown("**Autonomous Grid Multi-Agent System**")
st.sidebar.caption("LangGraph + Ollama (`qwen2.5`) + SQLite")

st.sidebar.subheader("📡 Telemetry Scenario")
preset = st.sidebar.selectbox(
    "Choose Preset Scenario",
    [
        "Nominal Grid State",
        "Undervoltage Sag & Low PF",
        "Feeder Thermal Overload",
        "Frequency Decay Event",
        "Custom Manual Inputs",
    ],
)

substation = st.sidebar.text_input("Substation ID", value="SUB-CENTRAL-01")

if preset == "Nominal Grid State":
    v_val, f_val, l_val, pf_val, t_val = 230.4, 60.01, 115.0, 0.96, 42.0
elif preset == "Undervoltage Sag & Low PF":
    v_val, f_val, l_val, pf_val, t_val = 214.2, 59.95, 145.0, 0.83, 62.0
elif preset == "Feeder Thermal Overload":
    v_val, f_val, l_val, pf_val, t_val = 227.0, 59.98, 174.0, 0.92, 88.5
elif preset == "Frequency Decay Event":
    v_val, f_val, l_val, pf_val, t_val = 222.0, 59.72, 162.0, 0.88, 54.0
else:
    v_val = st.sidebar.slider("Voltage (kV)", 200.0, 250.0, 230.0, 0.5)
    f_val = st.sidebar.slider("Frequency (Hz)", 59.0, 61.0, 60.0, 0.05)
    l_val = st.sidebar.slider("Load (MW)", 50.0, 200.0, 120.0, 1.0)
    pf_val = st.sidebar.slider("Power Factor", 0.70, 1.00, 0.95, 0.01)
    t_val = st.sidebar.slider("Transformer Temp (°C)", 20.0, 110.0, 45.0, 1.0)

run_button = st.sidebar.button("🚀 Run PulseGrid Workflow", type="primary", use_container_width=True)

# Main Header
st.title("⚡ PulseGrid AI — Smart Grid Resilience Orchestrator")
st.markdown(
    "Autonomous monitoring, root-cause diagnosis via **Qwen 2.5**, and memory-augmented intervention dispatching."
)

# Real-Time Telemetry Display
st.subheader("📊 Substation Telemetry Readings")
col1, col2, col3, col4, col5 = st.columns(5)

v_status = "badge-normal" if 218.5 <= v_val <= 241.5 else "badge-alert"
f_status = "badge-normal" if 59.8 <= f_val <= 60.2 else "badge-alert"
l_status = "badge-normal" if l_val <= 165.0 else "badge-alert"
pf_status = "badge-normal" if pf_val >= 0.90 else "badge-alert"
t_status = "badge-normal" if t_val <= 80.0 else "badge-alert"

with col1:
    st.metric("Bus Voltage", f"{v_val:.1f} kV", delta=f"{v_val - 230.0:+.1f} kV vs 230kV")
    st.markdown(f"<span class='{v_status}'>Limit: 218.5 - 241.5 kV</span>", unsafe_allow_html=True)
with col2:
    st.metric("Frequency", f"{f_val:.2f} Hz", delta=f"{f_val - 60.00:+.2f} Hz vs 60Hz")
    st.markdown(f"<span class='{f_status}'>Limit: 59.8 - 60.2 Hz</span>", unsafe_allow_html=True)
with col3:
    st.metric("Feeder Load", f"{l_val:.1f} MW", delta=f"{l_val / 180 * 100:.0f}% Cap")
    st.markdown(f"<span class='{l_status}'>Max: 165.0 MW</span>", unsafe_allow_html=True)
with col4:
    st.metric("Power Factor", f"{pf_val:.2f}", delta=f"{pf_val - 0.95:+.2f}")
    st.markdown(f"<span class='{pf_status}'>Min: 0.90</span>", unsafe_allow_html=True)
with col5:
    st.metric("Core Temp", f"{t_val:.1f} °C", delta=f"{t_val - 45.0:+.1f} °C")
    st.markdown(f"<span class='{t_status}'>Alarm: > 80.0 °C</span>", unsafe_allow_html=True)

st.divider()

# Session State for Workflow Results
if "workflow_result" not in st.session_state:
    st.session_state.workflow_result = None

if run_button:
    current_telemetry = {
        "substation_id": substation,
        "voltage_kv": float(v_val),
        "frequency_hz": float(f_val),
        "load_mw": float(l_val),
        "power_factor": float(pf_val),
        "temperature_c": float(t_val),
    }
    with st.spinner("Executing LangGraph Agent Workflow (Monitoring ➔ Investigating ➔ Intervening)..."):
        result = run_pulsegrid_workflow(current_telemetry)
        st.session_state.workflow_result = result
    st.success("Workflow cycle completed!")

# Workflow output visualization
if st.session_state.workflow_result:
    res = st.session_state.workflow_result
    anomaly_detected = res.get("anomaly_detected", False)
    status = res.get("status", "complete")

    st.subheader("🤖 LangGraph Agent Workflow Execution")

    # Workflow Status Bar
    wcol1, wcol2, wcol3, wcol4 = st.columns(4)
    with wcol1:
        st.info("1. **Monitor Node**\nThresholds & limits evaluation")
    with wcol2:
        if anomaly_detected:
            st.warning("2. **Investigate Node**\nQwen2.5 Root Cause Diagnosis")
        else:
            st.caption("2. **Investigate Node**\nSkipped (Nominal)")
    with wcol3:
        if anomaly_detected:
            st.warning("3. **Intervene Node**\nAction & Cost Formulation")
        else:
            st.caption("3. **Intervene Node**\nSkipped (Nominal)")
    with wcol4:
        st.success(f"4. **Status**\n`{status.upper()}`")

    if anomaly_detected:
        st.error("⚠️ **Grid Anomaly Detected** — Automatic Mitigations Activated")

        tab_diag, tab_act, tab_logs = st.tabs(["🔍 Root Cause Diagnosis", "⚡ Proposed Intervention", "📜 Agent Message Stream"])

        with tab_diag:
            st.markdown("### LLM Diagnostic Analysis (`qwen2.5`)")
            st.write(res.get("root_cause", "Diagnosis unavailable"))

            with st.expander("Retrieved Memory Precedents (SQLite)"):
                mems = res.get("past_interventions", [])
                if mems:
                    st.dataframe(pd.DataFrame(mems)[["anomaly_type", "action_taken", "outcome_score", "timestamp"]], width=800)
                else:
                    st.write("No prior records retrieved.")

        with tab_act:
            st.markdown("### Recommended Mitigation Plan")
            action_data = res.get("proposed_intervention", {})
            icol1, icol2 = st.columns([2, 1])
            with icol1:
                st.markdown(f"**Action:** {action_data.get('action', 'N/A')}")
                st.markdown(f"**Expected Impact:** {action_data.get('expected_impact', 'N/A')}")
            with icol2:
                st.metric("Estimated Cost", f"${action_data.get('cost', 0):,.2f}")
                if st.button("✅ Confirm & Dispatch Grid Command", type="primary"):
                    st.balloons()
                    st.success(f"Command dispatched to {substation} SCADA gateway!")

        with tab_logs:
            st.markdown("### Appended State Message Logs (`operator.add`)")
            for msg in res.get("messages", []):
                st.code(str(msg), language="markdown")

    else:
        st.success("✅ **All Grid Metrics Nominal** — No human or autonomous intervention required.")
        with st.expander("Inspection Logs"):
            for msg in res.get("messages", []):
                st.code(str(msg))

# Historical Interventions View
st.divider()
st.subheader("📚 Grid Memory: Historical Interventions (SQLite)")
past_records = get_past_interventions("voltage_sag", limit=10)
if past_records:
    df_history = pd.DataFrame(past_records)
    st.dataframe(
        df_history[["id", "anomaly_type", "action_taken", "outcome_score", "timestamp"]],
        width=1000,
    )
else:
    st.info("No historical intervention records found.")

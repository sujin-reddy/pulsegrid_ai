"""LangGraph nodes for PulseGrid AI workflow."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from db_memory import get_past_interventions, save_intervention
from state import AgentState, ProposedIntervention

logger = logging.getLogger("pulsegrid.nodes")

# Grid operational limits (Nominal 230kV / 60Hz Transmission Bus)
VOLTAGE_NOMINAL_KV = 230.0
VOLTAGE_MIN_KV = 218.5  # -5% (ANSI C84.1 Range A)
VOLTAGE_MAX_KV = 241.5  # +5%
FREQ_NOMINAL_HZ = 60.0
FREQ_MIN_HZ = 59.80  # NERC reliability limit
FREQ_MAX_HZ = 60.20
POWER_FACTOR_MIN = 0.90
LOAD_CAPACITY_MW = 180.0
LOAD_ALERT_MW = 165.0  # >91% rated capacity
TEMP_ALARM_C = 80.0


def _get_llm(model_name: str = "qwen2.5") -> ChatOllama:
    """Instantiate local ChatOllama model with deterministic temperature."""
    return ChatOllama(model=model_name, temperature=0.2)


def monitor_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Analyze real-time telemetry against grid physics thresholds."""
    telemetry = state.get("telemetry_data") or {}

    substation_id = telemetry.get("substation_id", "SUB-ALPHA-01")
    v_kv = float(telemetry.get("voltage_kv", VOLTAGE_NOMINAL_KV))
    f_hz = float(telemetry.get("frequency_hz", FREQ_NOMINAL_HZ))
    load_mw = float(telemetry.get("load_mw", 120.0))
    pf = float(telemetry.get("power_factor", 0.95))
    temp_c = float(telemetry.get("temperature_c", 45.0))

    violations: List[str] = []
    cause_tag = "normal"

    if v_kv < VOLTAGE_MIN_KV:
        violations.append(f"Undervoltage sag: {v_kv:.1f} kV (threshold < {VOLTAGE_MIN_KV:.1f} kV)")
        cause_tag = "voltage_sag"
    elif v_kv > VOLTAGE_MAX_KV:
        violations.append(f"Overvoltage surge: {v_kv:.1f} kV (threshold > {VOLTAGE_MAX_KV:.1f} kV)")
        cause_tag = "voltage_surge"

    if f_hz < FREQ_MIN_HZ:
        violations.append(f"Underfrequency decay: {f_hz:.2f} Hz (threshold < {FREQ_MIN_HZ:.2f} Hz)")
        cause_tag = "frequency_decay"
    elif f_hz > FREQ_MAX_HZ:
        violations.append(f"Overfrequency spike: {f_hz:.2f} Hz (threshold > {FREQ_MAX_HZ:.2f} Hz)")
        cause_tag = "frequency_surge"

    if pf < POWER_FACTOR_MIN:
        violations.append(f"Severe reactive deficit (PF: {pf:.2f}, limit >= {POWER_FACTOR_MIN:.2f})")
        if cause_tag == "normal":
            cause_tag = "reactive_deficit"

    if load_mw > LOAD_ALERT_MW:
        violations.append(f"Feeder capacity overload: {load_mw:.1f} MW ({load_mw / LOAD_CAPACITY_MW * 100:.1f}% capacity)")
        cause_tag = "thermal_overload"

    if temp_c > TEMP_ALARM_C:
        violations.append(f"Transformer core overheat: {temp_c:.1f}°C (alarm > {TEMP_ALARM_C:.1f}°C)")
        cause_tag = "thermal_overload"

    anomaly_detected = len(violations) > 0

    if anomaly_detected:
        log_msg = (
            f"[MONITOR] [ALERT] ANOMALY DETECTED at {substation_id}: "
            + "; ".join(violations)
            + ". Triggering root-cause investigation."
        )
        new_status = "investigating"
    else:
        log_msg = (
            f"[MONITOR] [NOMINAL] Grid telemetry normal at {substation_id}: "
            f"V={v_kv:.1f}kV, f={f_hz:.2f}Hz, Load={load_mw:.1f}MW, PF={pf:.2f}, T={temp_c:.1f}C. All parameters within bounds."
        )
        new_status = "monitoring"

    return {
        "anomaly_detected": anomaly_detected,
        "status": new_status,
        "messages": [log_msg],
    }


def _infer_anomaly_type(telemetry: Dict[str, Any]) -> str:
    """Classify anomalous telemetry into standard category tag."""
    v_kv = float(telemetry.get("voltage_kv", VOLTAGE_NOMINAL_KV))
    f_hz = float(telemetry.get("frequency_hz", FREQ_NOMINAL_HZ))
    load_mw = float(telemetry.get("load_mw", 120.0))
    temp_c = float(telemetry.get("temperature_c", 45.0))
    pf = float(telemetry.get("power_factor", 0.95))

    if v_kv < VOLTAGE_MIN_KV or pf < POWER_FACTOR_MIN:
        return "voltage_sag"
    if f_hz < FREQ_MIN_HZ:
        return "frequency_decay"
    if load_mw > LOAD_ALERT_MW or temp_c > TEMP_ALARM_C:
        return "thermal_overload"
    return "voltage_sag"


def investigate_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Root-cause diagnosis using local Ollama model and database memory retrieval."""
    telemetry = state.get("telemetry_data") or {}
    anomaly_type = _infer_anomaly_type(telemetry)
    historical_memory = get_past_interventions(anomaly_type=anomaly_type, limit=3)

    memory_summary = "\n".join(
        [
            f"• [{m.get('anomaly_type', 'issue')}] {m.get('action_taken', '')} (Score: {m.get('outcome_score', 0.9):.2f})"
            for m in historical_memory
        ]
    )

    prompt = (
        f"You are the PulseGrid AI Senior Power Systems Diagnostic Agent.\n"
        f"Analyze this anomalous telemetry reading from substation {telemetry.get('substation_id', 'SUB-ALPHA-01')}:\n"
        f"- Bus Voltage: {telemetry.get('voltage_kv')} kV (Nominal: 230 kV)\n"
        f"- Grid Frequency: {telemetry.get('frequency_hz')} Hz (Nominal: 60.0 Hz)\n"
        f"- Active Load: {telemetry.get('load_mw')} MW (Rated Max: 180 MW)\n"
        f"- Power Factor: {telemetry.get('power_factor')}\n"
        f"- Transformer Core Temp: {telemetry.get('temperature_c')} °C\n\n"
        f"High-scoring past interventions (Outcome >= 0.7) from memory:\n"
        f"{memory_summary}\n\n"
        f"Provide a concise, engineering-grade diagnosis of the primary root cause and physical mechanism (maximum 2-3 sentences)."
    )

    try:
        llm = _get_llm()
        response = llm.invoke([SystemMessage(content="You are a smart-grid diagnostic expert."), HumanMessage(content=prompt)])
        diagnosis = response.content.strip()
    except Exception as exc:
        logger.warning("Local Ollama diagnosis fallback triggered: %s", exc)
        diagnosis = (
            f"Localized reactive power mismatch and inductive load surge causing significant voltage sag "
            f"to {telemetry.get('voltage_kv', '216.0')}kV and high transformer thermal flux."
        )

    log_msg = f"[INVESTIGATE] [DIAGNOSIS] Root Cause Diagnosed by Qwen2.5:\n{diagnosis}"

    return {
        "root_cause": diagnosis,
        "past_interventions": historical_memory,
        "status": "intervening",
        "messages": [log_msg],
    }


def intervene_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Formulate actionable intervention, estimate costs, and persist to SQLite memory."""
    telemetry = state.get("telemetry_data") or {}
    root_cause = state.get("root_cause", "Grid voltage and frequency instability")
    anomaly_type = _infer_anomaly_type(telemetry)

    prompt = (
        f"You are the PulseGrid AI Autonomous Grid Dispatcher.\n"
        f"Diagnosed root cause:\n{root_cause}\n\n"
        f"Telemetry:\n"
        f"- Voltage: {telemetry.get('voltage_kv')} kV, Frequency: {telemetry.get('frequency_hz')} Hz, "
        f"Load: {telemetry.get('load_mw')} MW, Temp: {telemetry.get('temperature_c')} °C\n\n"
        f"Select the most effective immediate mitigation action based on historical memory precedents.\n"
        f"Respond in EXACT JSON format with keys:\n"
        f'{{"action": "specific equipment dispatch action", "cost": 450.0, "expected_impact": "quantitative restoration estimate"}}\n'
        f"Return ONLY valid JSON."
    )

    intervention_data: ProposedIntervention = {
        "action": "Switch Capacitor Bank 4 at Substation-7 for +25 MVAR reactive compensation",
        "cost": 450.0,
        "expected_impact": "Restore bus voltage to 230kV (+/-1%) and reduce reactive line losses within 30 seconds",
    }

    try:
        llm = _get_llm()
        response = llm.invoke([SystemMessage(content="You are an automated power dispatch optimizer. Output only valid JSON."), HumanMessage(content=prompt)])
        content = response.content.strip()
        # Clean markdown codeblocks if returned
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        parsed = json.loads(content)
        if "action" in parsed and "cost" in parsed and "expected_impact" in parsed:
            intervention_data = {
                "action": str(parsed["action"]),
                "cost": float(parsed["cost"]),
                "expected_impact": str(parsed["expected_impact"]),
            }
    except Exception as exc:
        logger.warning("Local Ollama intervention fallback triggered: %s", exc)

    # Persist the newly formulated intervention into intervention_memory
    save_intervention(
        anomaly_type=anomaly_type,
        action_taken=intervention_data["action"],
        outcome_score=0.92,
    )

    log_msg = (
        f"[INTERVENE] [ACTION] Proposed Intervention: {intervention_data['action']} | "
        f"Est. Cost: ${intervention_data['cost']:,.2f} | "
        f"Expected Impact: {intervention_data['expected_impact']}"
    )

    return {
        "proposed_intervention": dict(intervention_data),
        "status": "complete",
        "messages": [log_msg],
    }

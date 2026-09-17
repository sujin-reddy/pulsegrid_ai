"""Asynchronous worker nodes for PulseGrid AI LangGraph workflow.

Agents for Detection, Analysis & Preventive Interventions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from db_memory import get_past_interventions, save_intervention
from state import AgentState, ProposedIntervention

logger = logging.getLogger("pulsegrid.nodes")


def _get_llm(model_name: str = "qwen2.5") -> ChatOllama:
    """Instantiate local ChatOllama model with deterministic temperature."""
    return ChatOllama(model=model_name, temperature=0.2)


def _extract_anomaly_type(telemetry: Dict[str, Any]) -> str:
    """Derive standard anomaly category tag from telemetry values."""
    if "temperature_c" in telemetry and float(telemetry["temperature_c"]) > 80:
        return "thermal_overload"
    if "voltage_kv" in telemetry and (float(telemetry["voltage_kv"]) < 218.5 or float(telemetry.get("power_factor", 1.0)) < 0.90):
        return "voltage_sag"
    if "frequency_hz" in telemetry and float(telemetry["frequency_hz"]) < 59.80:
        return "frequency_decay"
    if "load_mw" in telemetry and float(telemetry["load_mw"]) > 165.0:
        return "thermal_overload"
    if "value" in telemetry and float(telemetry["value"]) > 80:
        return "threshold_exceeded"
    return "general_anomaly"


# =====================================================================
# Node 1: Monitoring Node
# =====================================================================
async def monitoring_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Monitor telemetry data and detect threshold violations."""
    telemetry = state.get("telemetry_data") or {}

    anomaly_detected = False
    violations: List[str] = []

    # Check generic 'value' key if present (e.g. value > 80)
    if "value" in telemetry:
        val = float(telemetry["value"])
        if val > 80:
            anomaly_detected = True
            violations.append(f"Metric value {val:.1f} exceeded threshold 80")

    # Also check standard grid telemetry metrics if present
    if "voltage_kv" in telemetry:
        v_kv = float(telemetry["voltage_kv"])
        if v_kv < 218.5:
            anomaly_detected = True
            violations.append(f"Undervoltage sag: {v_kv:.1f} kV (< 218.5 kV)")
        elif v_kv > 241.5:
            anomaly_detected = True
            violations.append(f"Overvoltage surge: {v_kv:.1f} kV (> 241.5 kV)")

    if "temperature_c" in telemetry:
        temp_c = float(telemetry["temperature_c"])
        if temp_c > 80.0:
            anomaly_detected = True
            violations.append(f"Core temperature alarm: {temp_c:.1f}°C (> 80.0°C)")

    if "frequency_hz" in telemetry:
        f_hz = float(telemetry["frequency_hz"])
        if f_hz < 59.80:
            anomaly_detected = True
            violations.append(f"Frequency decay: {f_hz:.2f} Hz (< 59.80 Hz)")

    if "load_mw" in telemetry:
        load_mw = float(telemetry["load_mw"])
        if load_mw > 165.0:
            anomaly_detected = True
            violations.append(f"Feeder capacity overload: {load_mw:.1f} MW (> 165.0 MW)")

    if "power_factor" in telemetry:
        pf = float(telemetry["power_factor"])
        if pf < 0.90:
            anomaly_detected = True
            violations.append(f"Reactive deficit PF: {pf:.2f} (< 0.90)")

    if anomaly_detected:
        log_msg = f"[MONITOR] [ALERT] Anomaly detected: {'; '.join(violations)}."
        new_status = "investigating"
    else:
        log_msg = f"[MONITOR] [NOMINAL] Telemetry within acceptable thresholds."
        new_status = "monitoring"

    return {
        "anomaly_detected": anomaly_detected,
        "status": new_status,
        "messages": [log_msg],
    }


# =====================================================================
# Node 2: Investigation Node
# =====================================================================
async def investigation_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Perform root-cause analysis via local LLM or deterministic rule fallback."""
    if not state.get("anomaly_detected", False):
        return {
            "root_cause": "No anomaly detected during monitoring.",
            "status": "monitoring",
            "messages": ["[INVESTIGATE] Skipped root cause analysis as no anomaly was detected."],
        }

    telemetry = state.get("telemetry_data") or {}
    anomaly_type = _extract_anomaly_type(telemetry)

    prompt = (
        f"You are the PulseGrid AI Senior Power Systems Diagnostic Agent.\n"
        f"Analyze this anomalous telemetry dataset:\n"
        f"{json.dumps(telemetry, indent=2)}\n\n"
        f"Anomaly Classification: {anomaly_type}\n\n"
        f"Provide a concise, engineering-grade diagnosis of the primary root cause and physical mechanism (maximum 2-3 sentences)."
    )

    try:
        llm = _get_llm()
        response = await llm.ainvoke(
            [
                SystemMessage(content="You are a smart-grid diagnostic expert."),
                HumanMessage(content=prompt),
            ]
        )
        root_cause = response.content.strip()
    except Exception as exc:
        logger.warning("Local LLM unavailable or timed out; utilizing deterministic rule fallback: %s", exc)
        # Deterministic mock rule fallback
        if anomaly_type == "thermal_overload":
            root_cause = (
                "Sustained heavy load on feeder line compounded by elevated ambient conditions, "
                "causing localized thermal congestion and core temperature elevation above rated limits."
            )
        elif anomaly_type == "voltage_sag":
            root_cause = (
                "Sudden inductive load ramp or remote line tripping resulting in severe reactive power deficit "
                "and depressed bus voltage."
            )
        elif anomaly_type == "frequency_decay":
            root_cause = (
                "Unplanned generation deficit creating an instantaneous supply-demand mismatch "
                "and rate-of-change-of-frequency (RoCoF) decline."
            )
        else:
            root_cause = (
                f"Telemetry reading exceeded operational limits (anomaly: {anomaly_type}), "
                "indicating imminent component stress requiring preventive intervention."
            )

    log_msg = f"[INVESTIGATE] [DIAGNOSIS] Root cause identified:\n{root_cause}"

    return {
        "root_cause": root_cause,
        "status": "intervening",
        "messages": [log_msg],
    }


# =====================================================================
# Node 3: Preventive Intervention Node
# =====================================================================
async def preventive_intervention_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Retrieve memory precedents, formulate preventive plan, and log to database."""
    telemetry = state.get("telemetry_data") or {}
    root_cause = state.get("root_cause", "Threshold exceeded in monitored telemetry")
    anomaly_type = _extract_anomaly_type(telemetry)

    # Retrieve relevant past actions with outcome score >= 0.7 from persistent memory
    past_actions = get_past_interventions(anomaly_type=anomaly_type, limit=3)

    memory_context = "\n".join(
        [
            f"• Action: {item.get('action_taken')} | Outcome Score: {item.get('outcome_score', 0.8):.2f}"
            for item in past_actions
        ]
    )

    prompt = (
        f"You are the PulseGrid AI Autonomous Grid Intervention Dispatcher.\n"
        f"Root cause:\n{root_cause}\n\n"
        f"Telemetry snapshot: {json.dumps(telemetry)}\n\n"
        f"Historical interventions with outcome score >= 0.7 from persistent memory:\n"
        f"{memory_context}\n\n"
        f"Formulate a preventive action plan considering these historical outcome scores.\n"
        f"Respond in EXACT JSON format with keys:\n"
        f'{{"action": "specific preventive equipment or dispatch action", "cost": 450.0, "expected_impact": "quantitative expected recovery"}}\n'
        f"Return ONLY valid JSON."
    )

    proposed_plan: ProposedIntervention = {
        "action": "Dispatch autonomous reactive compensation and shed non-critical auxiliary feeder",
        "cost": 500.0,
        "expected_impact": "Stabilize telemetry parameters within nominal threshold in < 60 seconds",
    }

    try:
        llm = _get_llm()
        response = await llm.ainvoke(
            [
                SystemMessage(content="You are an automated power dispatch optimizer. Output only valid JSON."),
                HumanMessage(content=prompt),
            ]
        )
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        parsed = json.loads(content)
        if "action" in parsed and "cost" in parsed and "expected_impact" in parsed:
            proposed_plan = {
                "action": str(parsed["action"]),
                "cost": float(parsed["cost"]),
                "expected_impact": str(parsed["expected_impact"]),
            }
    except Exception as exc:
        logger.warning("Local LLM unavailable or invalid format; applying rule-based plan: %s", exc)
        if past_actions:
            best_precedent = past_actions[0]
            proposed_plan = {
                "action": best_precedent.get("action_taken", proposed_plan["action"]),
                "cost": 450.0,
                "expected_impact": f"Replicate verified historical mitigation (past score: {best_precedent.get('outcome_score', 0.9):.2f})",
            }

    # Log proposed intervention to db_memory.py with save_intervention()
    save_intervention(
        anomaly_type=anomaly_type,
        action_taken=proposed_plan["action"],
        outcome_score=0.95,
    )

    log_msg = (
        f"[INTERVENE] [ACTION] Preventive Plan: {proposed_plan['action']} | "
        f"Cost: ${proposed_plan['cost']:,.2f} | "
        f"Expected Impact: {proposed_plan['expected_impact']}"
    )

    return {
        "proposed_intervention": dict(proposed_plan),
        "past_interventions": past_actions,
        "status": "complete",
        "messages": [log_msg],
    }


# Backward-compatible aliases
monitor_node = monitoring_node
investigate_node = investigation_node
intervene_node = preventive_intervention_node

__all__ = [
    "monitoring_node",
    "investigation_node",
    "preventive_intervention_node",
    "monitor_node",
    "investigate_node",
    "intervene_node",
]

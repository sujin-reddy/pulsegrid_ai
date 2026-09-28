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
from domain_config import DOMAIN_REGISTRY, DomainClassification, get_domain_config
from state import AgentState, ProposedIntervention, TelemetryPayload

logger = logging.getLogger("pulsegrid.nodes")


def _get_llm(model_name: str = "qwen2.5") -> ChatOllama:
    """Instantiate local ChatOllama model with deterministic temperature."""
    return ChatOllama(model=model_name, temperature=0.2)


def _normalize_telemetry(telemetry: Dict[str, Any]) -> TelemetryPayload:
    """Normalize incoming telemetry dictionary to a generic TelemetryPayload.

    - If incoming dict has "metrics" dict, extract numeric metrics from it.
    - Else (legacy flat grid-only dict with substation_id, voltage_kv, etc.), wrap it:
      domain="energy_grid", device_id=telemetry.get("substation_id", "UNKNOWN"),
      metrics = every numeric key except substation_id / device_id / domain.
    """
    if not isinstance(telemetry, dict):
        return {
            "domain": "energy_grid",
            "device_id": "UNKNOWN",
            "metrics": {},
            "unit_map": {},
        }

    domain = str(telemetry.get("domain", "energy_grid"))
    device_id = str(telemetry.get("device_id", telemetry.get("substation_id", "UNKNOWN")))
    unit_map = (
        {str(k): str(v) for k, v in telemetry.get("unit_map", {}).items()}
        if isinstance(telemetry.get("unit_map"), dict)
        else {}
    )

    if "metrics" in telemetry and isinstance(telemetry["metrics"], dict):
        raw_metrics = telemetry["metrics"]
        metrics = {
            str(k): float(v)
            for k, v in raw_metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    else:
        metrics = {
            str(k): float(v)
            for k, v in telemetry.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("substation_id", "device_id", "domain")
        }

    return {
        "domain": domain,
        "device_id": device_id,
        "metrics": metrics,
        "unit_map": unit_map,
    }


def _extract_anomaly_type(telemetry: Dict[str, Any]) -> str:
    """Derive standard anomaly category tag from telemetry values."""
    if not isinstance(telemetry, dict):
        return "general_anomaly"
    metrics = telemetry.get("metrics") if isinstance(telemetry.get("metrics"), dict) else telemetry
    if "temperature_c" in metrics and float(metrics["temperature_c"]) > 80:
        return "thermal_overload"
    if "voltage_kv" in metrics and (float(metrics["voltage_kv"]) < 218.5 or float(metrics.get("power_factor", 1.0)) < 0.90):
        return "voltage_sag"
    if "frequency_hz" in metrics and float(metrics["frequency_hz"]) < 59.80:
        return "frequency_decay"
    if "load_mw" in metrics and float(metrics["load_mw"]) > 165.0:
        return "thermal_overload"
    if "value" in metrics and float(metrics["value"]) > 80:
        return "threshold_exceeded"
    return "general_anomaly"


# =====================================================================
# Node 0: Domain Router Node
# =====================================================================
async def domain_router_node(state: AgentState) -> Dict[str, Any]:
    """Node 0: Route incoming telemetry to the appropriate operating domain.

    Resolution order:
      a. FAST PATH (fast-tag): Explicit 'domain' key in payload matching DOMAIN_REGISTRY.
      b. FAST PATH (fast-alias): device_id or metric keys match tag_aliases.
      c. LLM FALLBACK (llm-fallback): Structured output classification via Qwen 2.5.
      d. DEFAULT (default): Fallback to 'energy_grid' with warning log.
    """
    raw_telemetry = state.get("telemetry_data") or {}
    resolved: str | None = None
    path_used: str = "default"
    log_msg: str = ""

    # a. FAST PATH: explicit domain key matching DOMAIN_REGISTRY id
    if "domain" in raw_telemetry and str(raw_telemetry["domain"]).strip().lower() in DOMAIN_REGISTRY:
        resolved = str(raw_telemetry["domain"]).strip().lower()
        path_used = "fast-tag"
        log_msg = f"[DOMAIN_ROUTER] [FAST-PATH: fast-tag] Telemetry payload explicitly specifies domain: '{resolved}'."

    # b. FAST PATH: check payload keys/device_id against domain tag_aliases
    if not resolved:
        device_id = str(raw_telemetry.get("device_id", raw_telemetry.get("substation_id", ""))).lower()
        if "metrics" in raw_telemetry and isinstance(raw_telemetry["metrics"], dict):
            metric_keys = [str(k).lower() for k in raw_telemetry["metrics"].keys()]
        else:
            metric_keys = [str(k).lower() for k in raw_telemetry.keys() if k not in ("substation_id", "device_id", "domain")]

        search_tokens = [device_id] + metric_keys
        combined_search = f"{device_id} {' '.join(metric_keys)}"

        for domain_id, config in DOMAIN_REGISTRY.items():
            for alias in config.tag_aliases:
                alias_lower = alias.lower()
                if any(alias_lower in tok for tok in search_tokens) or (alias_lower in combined_search):
                    resolved = domain_id
                    path_used = "fast-alias"
                    log_msg = f"[DOMAIN_ROUTER] [FAST-PATH: fast-alias] Matched tag alias '{alias}' in device/metrics; routed to domain: '{resolved}'."
                    break
            if resolved:
                break

    # c. LLM FALLBACK: call LLM once with DomainClassification structured output
    if not resolved:
        try:
            llm = _get_llm(model_name="qwen2.5")
            structured_llm = llm.with_structured_output(DomainClassification)
            prompt = (
                f"Analyze this telemetry packet metadata and classify it into the best operating domain:\n"
                f"Device ID: {raw_telemetry.get('device_id', raw_telemetry.get('substation_id', 'UNKNOWN'))}\n"
                f"Payload Content: {raw_telemetry}\n"
                f"Candidate domains: {list(DOMAIN_REGISTRY.keys())}"
            )
            res: DomainClassification = await structured_llm.ainvoke(prompt)
            if res and res.domain_id in DOMAIN_REGISTRY:
                resolved = res.domain_id
                path_used = "llm-fallback"
                log_msg = f"[DOMAIN_ROUTER] [FALLBACK: llm-fallback] Classified by Qwen 2.5 as '{resolved}' (confidence: {res.confidence:.2f})."
            else:
                raise ValueError(f"Unknown domain returned by LLM: {res}")
        except Exception as exc:
            resolved = "energy_grid"
            path_used = "default"
            log_msg = f"[DOMAIN_ROUTER] [WARNING: default] LLM domain classification failed ({exc}); defaulted to '{resolved}'."

    if not resolved:
        resolved = "energy_grid"
        path_used = "default"
        log_msg = f"[DOMAIN_ROUTER] [WARNING: default] No match found; defaulted to '{resolved}'."

    return {
        "resolved_domain": resolved,
        "messages": [log_msg],
    }


# =====================================================================
# Node 1: Monitoring Node
# =====================================================================
async def monitoring_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Monitor telemetry data and detect threshold violations based on domain_config."""
    raw_telemetry = state.get("telemetry_data") or {}
    resolved_domain = state.get("resolved_domain") or "energy_grid"
    domain_cfg = get_domain_config(resolved_domain)

    normalized = _normalize_telemetry(raw_telemetry)
    if "domain" not in raw_telemetry:
        normalized["domain"] = resolved_domain

    metrics = normalized.get("metrics", {})
    thresholds = domain_cfg.metric_thresholds

    anomaly_detected = False
    violations: List[str] = []

    for metric_name, val in metrics.items():
        if metric_name not in thresholds:
            continue
        rule = thresholds[metric_name]
        val_float = float(val)
        unit = rule.get("unit", "")
        unit_str = f" {unit}" if unit and not unit.startswith("°") else unit
        label = rule.get("label", metric_name)
        op = rule.get("op", ">")
        threshold_val = float(rule.get("value", 0.0))

        violation_found = False
        active_label = label
        active_op = op
        active_threshold = threshold_val

        # Primary rule check
        if op == ">" and val_float > threshold_val:
            violation_found = True
        elif op == "<" and val_float < threshold_val:
            violation_found = True

        # Secondary rule check (e.g. overvoltage surge on voltage_kv)
        if not violation_found and "secondary_op" in rule and "secondary_value" in rule:
            sec_op = rule["secondary_op"]
            sec_val = float(rule["secondary_value"])
            sec_label = rule.get("secondary_label", label)
            if sec_op == ">" and val_float > sec_val:
                violation_found = True
                active_label = sec_label
                active_op = sec_op
                active_threshold = sec_val
            elif sec_op == "<" and val_float < sec_val:
                violation_found = True
                active_label = sec_label
                active_op = sec_op
                active_threshold = sec_val

        if violation_found:
            anomaly_detected = True
            if resolved_domain == "energy_grid":
                if metric_name == "value":
                    violations.append(f"Metric value {val_float:.1f} exceeded threshold {int(threshold_val)}")
                elif metric_name == "voltage_kv":
                    violations.append(f"{active_label}: {val_float:.1f} kV ({active_op} {active_threshold:.1f} kV)")
                elif metric_name == "temperature_c":
                    violations.append(f"{active_label}: {val_float:.1f}°C ({active_op} {active_threshold:.1f}°C)")
                elif metric_name == "frequency_hz":
                    violations.append(f"{active_label}: {val_float:.2f} Hz ({active_op} {active_threshold:.2f} Hz)")
                elif metric_name == "load_mw":
                    violations.append(f"{active_label}: {val_float:.1f} MW ({active_op} {active_threshold:.1f} MW)")
                elif metric_name == "power_factor":
                    violations.append(f"{active_label}: {val_float:.2f} ({active_op} {active_threshold:.2f})")
                else:
                    violations.append(f"{active_label}: {val_float:.1f}{unit_str} ({active_op} {active_threshold:.1f}{unit_str})")
            else:
                unit_fmt = f"{unit_str}" if unit_str else ""
                violations.append(f"{metric_name} ({active_label}): {val_float:.1f}{unit_fmt} ({active_op} {active_threshold:.1f}{unit_fmt})")

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

    domain = telemetry.get("domain", "energy_grid") if isinstance(telemetry, dict) else "energy_grid"

    # Log proposed intervention to db_memory.py with save_intervention()
    save_intervention(
        anomaly_type=anomaly_type,
        action_taken=proposed_plan["action"],
        outcome_score=0.95,
        domain=domain,
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
    "domain_router_node",
    "monitoring_node",
    "investigation_node",
    "preventive_intervention_node",
    "monitor_node",
    "investigate_node",
    "intervene_node",
]

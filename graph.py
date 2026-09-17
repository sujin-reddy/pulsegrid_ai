"""LangGraph state machine graph for PulseGrid AI."""

from __future__ import annotations

from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from nodes import investigate_node, intervene_node, monitor_node
from state import AgentState


def route_after_monitoring(state: AgentState) -> Literal["investigate", "end"]:
    """Conditional router: proceed to investigation if anomaly detected, else terminate."""
    if state.get("anomaly_detected", False):
        return "investigate"
    return "end"


def build_pulsegrid_graph() -> StateGraph:
    """Build and compile the PulseGrid AI LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Register workflow nodes
    workflow.add_node("monitor", monitor_node)
    workflow.add_node("investigate", investigate_node)
    workflow.add_node("intervene", intervene_node)

    # Set workflow entry point
    workflow.set_entry_point("monitor")

    # Conditional routing from monitor node
    workflow.add_conditional_edges(
        "monitor",
        route_after_monitoring,
        {
            "investigate": "investigate",
            "end": END,
        },
    )

    # Deterministic progression once an anomaly is being handled
    workflow.add_edge("investigate", "intervene")
    workflow.add_edge("intervene", END)

    return workflow.compile()


# Singleton compiled graph instance
pulsegrid_app = build_pulsegrid_graph()


def run_pulsegrid_workflow(telemetry: Dict[str, Any]) -> AgentState:
    """Execute the full workflow for an incoming telemetry snapshot."""
    initial_state: AgentState = {
        "messages": [f"Initiating PulseGrid cycle for substation {telemetry.get('substation_id', 'UNKNOWN')}"],
        "telemetry_data": telemetry,
        "anomaly_detected": False,
        "root_cause": "",
        "proposed_intervention": {},
        "past_interventions": [],
        "status": "monitoring",
    }
    final_state = pulsegrid_app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    test_telemetry = {
        "substation_id": "SUB-NORTH-04",
        "voltage_kv": 214.2,  # Anomaly: undervoltage sag
        "frequency_hz": 59.95,
        "load_mw": 145.0,
        "power_factor": 0.84,  # Anomaly: low power factor
        "temperature_c": 62.0,
    }
    print("Executing PulseGrid workflow...")
    result = run_pulsegrid_workflow(test_telemetry)
    print("Workflow Final Status:", result.get("status"))
    print("Anomaly Detected:", result.get("anomaly_detected"))
    print("Root Cause:", result.get("root_cause"))
    print("Proposed Intervention:", result.get("proposed_intervention"))

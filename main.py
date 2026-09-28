"""Main LangGraph workflow wiring and execution pipeline for PulseGrid AI."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from nodes import (
    domain_router_node,
    investigation_node,
    monitoring_node,
    preventive_intervention_node,
)
from state import AgentState


def check_anomaly_condition(state: AgentState) -> Literal["investigate", "__end__"]:
    """Conditional router from monitor node.

    If anomaly_detected is True -> proceed to investigate.
    Else -> route to END.
    """
    if state.get("anomaly_detected", False):
        return "investigate"
    return END


# 1. Instantiate StateGraph with shared AgentState
workflow = StateGraph(AgentState)

# 2. Add nodes: domain_router, monitor, investigate, intervene
workflow.add_node("domain_router", domain_router_node)
workflow.add_node("monitor", monitoring_node)
workflow.add_node("investigate", investigation_node)
workflow.add_node("intervene", preventive_intervention_node)

# 3. Set entry point to domain_router
workflow.set_entry_point("domain_router")

# 4. Add edge from domain_router to monitor
workflow.add_edge("domain_router", "monitor")

# 5. Add conditional edge from monitor
workflow.add_conditional_edges(
    "monitor",
    check_anomaly_condition,
    {
        "investigate": "investigate",
        END: END,
    },
)

# 6. Add direct edges: investigate -> intervene -> END
workflow.add_edge("investigate", "intervene")
workflow.add_edge("intervene", END)

# 7. Compile the graph into app_graph
app_graph = workflow.compile()


async def arun_pipeline(input_telemetry: Dict[str, Any]) -> AgentState:
    """Asynchronously execute the compiled graph pipeline."""
    initial_state: AgentState = {
        "messages": [f"[START] Initializing PulseGrid pipeline execution."],
        "telemetry_data": input_telemetry,
        "resolved_domain": "",
        "anomaly_detected": False,
        "root_cause": "",
        "proposed_intervention": {},
        "past_interventions": [],
        "status": "monitoring",
    }
    final_state = await app_graph.ainvoke(initial_state)
    return final_state


def run_pipeline(input_telemetry: Dict[str, Any]) -> AgentState:
    """Execute the graph and return the final state.

    Handles execution whether invoked from a sync context, CLI script, or
    inside an existing running asyncio event loop (e.g., Streamlit / Jupyter).
    """
    initial_state: AgentState = {
        "messages": [f"[START] Initializing PulseGrid pipeline execution."],
        "telemetry_data": input_telemetry,
        "resolved_domain": "",
        "anomaly_detected": False,
        "root_cause": "",
        "proposed_intervention": {},
        "past_interventions": [],
        "status": "monitoring",
    }

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, app_graph.ainvoke(initial_state)).result()
    else:
        return asyncio.run(app_graph.ainvoke(initial_state))


if __name__ == "__main__":
    sample_telemetry = {
        "substation_id": "SUB-NORTH-04",
        "value": 86.5,
        "temperature_c": 84.0,
        "voltage_kv": 213.5,
        "frequency_hz": 59.75,
        "load_mw": 169.0,
        "power_factor": 0.82,
    }

    print("Running PulseGrid AI pipeline via main.py...")
    result_state = run_pipeline(sample_telemetry)
    print("\n--- Execution Finished ---")
    print(f"Status: {result_state.get('status')}")
    print(f"Anomaly Detected: {result_state.get('anomaly_detected')}")
    print(f"Root Cause: {result_state.get('root_cause')}")
    print(f"Proposed Intervention: {result_state.get('proposed_intervention')}")
    print(f"Message Count: {len(result_state.get('messages', []))}")

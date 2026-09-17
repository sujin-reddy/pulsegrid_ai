"""State definition for the PulseGrid AI multi-agent workflow."""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


class ProposedIntervention(TypedDict, total=False):
    """Schema for a proposed grid intervention action."""
    action: str
    cost: float
    expected_impact: str


# Status values representing the lifecycle of the workflow
WorkflowStatus = Literal["monitoring", "investigating", "intervening", "complete"]


class AgentState(TypedDict, total=False):
    """Shared state object passed across nodes in the PulseGrid AI workflow.

    Attributes:
        messages: List of messages/logs appended via operator.add reducer across nodes.
        telemetry_data: Dict containing sensor/time-series reading.
        anomaly_detected: Bool flag indicating whether an anomaly was detected.
        root_cause: Str explanation of the diagnosed issue.
        proposed_intervention: Dict containing intervention details (action, cost, expected_impact).
        past_interventions: List of dicts retrieved from database memory.
        status: Str representing current phase (monitoring, investigating, intervening, complete).
    """

    messages: Annotated[List[Any], operator.add]
    telemetry_data: Dict[str, Any]
    anomaly_detected: bool
    root_cause: str
    proposed_intervention: Dict[str, Any]
    past_interventions: List[Dict[str, Any]]
    status: Literal["monitoring", "investigating", "intervening", "complete"] | str


__all__ = ["AgentState", "ProposedIntervention", "WorkflowStatus"]

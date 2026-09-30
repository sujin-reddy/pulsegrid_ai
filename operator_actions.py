"""PulseGrid AI — Operator Action Taxonomy.

Maps each (domain, anomaly_type) pair to expected canonical operator responses.
Used by run_experiment.py to compare the agent's proposed_intervention against
a known gold-standard action catalogue for evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OperatorAction:
    """A gold-standard operator action for a given domain and anomaly type."""

    domain: str
    anomaly_type: str
    canonical_action: str          # Reference action string for keyword matching
    keywords: List[str]            # Keywords that must appear in a valid agent action
    cost_range: tuple[float, float] = (0.0, 10_000.0)  # Expected cost band ($)
    severity: str = "medium"       # low | medium | high | critical


# ---------------------------------------------------------------------------
# Operator Action Catalogue
# ---------------------------------------------------------------------------

OPERATOR_ACTIONS: Dict[str, List[OperatorAction]] = {
    # ── Energy Grid ────────────────────────────────────────────────────────
    "energy_grid": [
        OperatorAction(
            domain="energy_grid",
            anomaly_type="voltage_sag",
            canonical_action="Switch capacitor bank or tap changer to compensate reactive deficit",
            keywords=["capacitor", "tap", "reactive", "compensation", "statcom", "var"],
            cost_range=(200.0, 2_000.0),
            severity="high",
        ),
        OperatorAction(
            domain="energy_grid",
            anomaly_type="thermal_overload",
            canonical_action="Shed non-critical load or reroute via Dynamic Line Rating",
            keywords=["shed", "load", "reroute", "bess", "battery", "dlr", "feeder"],
            cost_range=(300.0, 3_000.0),
            severity="high",
        ),
        OperatorAction(
            domain="energy_grid",
            anomaly_type="frequency_decay",
            canonical_action="Activate Fast Frequency Response governor and demand response",
            keywords=["frequency", "governor", "demand", "ffr", "response", "inertia"],
            cost_range=(100.0, 1_500.0),
            severity="critical",
        ),
        OperatorAction(
            domain="energy_grid",
            anomaly_type="general_anomaly",
            canonical_action="Dispatch autonomous reactive compensation and auxiliary shed",
            keywords=["reactive", "compensat", "dispatch", "shed", "stabiliz"],
            cost_range=(100.0, 5_000.0),
            severity="medium",
        ),
        OperatorAction(
            domain="energy_grid",
            anomaly_type="threshold_exceeded",
            canonical_action="Dispatch autonomous reactive compensation and auxiliary shed",
            keywords=["reactive", "compensat", "dispatch", "shed", "stabiliz"],
            cost_range=(100.0, 5_000.0),
            severity="medium",
        ),
    ],
    # ── Data Center ────────────────────────────────────────────────────────
    "datacenter": [
        OperatorAction(
            domain="datacenter",
            anomaly_type="thermal_overload",
            canonical_action="Activate cooling fans, throttle GPU workload, or migrate pod",
            keywords=["cool", "throttl", "migrat", "pod", "gpu", "thermal", "fan"],
            cost_range=(50.0, 800.0),
            severity="high",
        ),
        OperatorAction(
            domain="datacenter",
            anomaly_type="general_anomaly",
            canonical_action="Increase cooling capacity and schedule workload rebalancing",
            keywords=["cool", "rebalanc", "workload", "cluster", "scale"],
            cost_range=(50.0, 1_000.0),
            severity="medium",
        ),
    ],
    # ── Water / Hydrology ──────────────────────────────────────────────────
    "water_env": [
        OperatorAction(
            domain="water_env",
            anomaly_type="general_anomaly",
            canonical_action="Open spillway gates, issue flood alert, activate evacuation corridor",
            keywords=["spillway", "gate", "alert", "evacuati", "barrier", "flood"],
            cost_range=(500.0, 15_000.0),
            severity="critical",
        ),
        OperatorAction(
            domain="water_env",
            anomaly_type="threshold_exceeded",
            canonical_action="Open spillway and issue downstream flood warning",
            keywords=["spillway", "downstream", "warning", "alert", "evacuati"],
            cost_range=(500.0, 10_000.0),
            severity="critical",
        ),
    ],
    # ── Traffic ────────────────────────────────────────────────────────────
    "traffic": [
        OperatorAction(
            domain="traffic",
            anomaly_type="general_anomaly",
            canonical_action="Adjust signal timing and divert via emergency corridor",
            keywords=["signal", "timing", "divert", "corridor", "reroute", "traffic"],
            cost_range=(0.0, 500.0),
            severity="medium",
        ),
    ],
}


def get_operator_action(domain: str, anomaly_type: str) -> Optional[OperatorAction]:
    """Return the best-matching OperatorAction for a given domain + anomaly_type.

    Falls back to the first action in the domain catalogue, then to None.
    """
    actions = OPERATOR_ACTIONS.get(domain, [])
    # Exact match on anomaly_type first
    for action in actions:
        if action.anomaly_type == anomaly_type:
            return action
    # Fallback: first entry in domain
    return actions[0] if actions else None


def action_keyword_match(proposed_action: str, operator_action: OperatorAction) -> float:
    """Return keyword coverage score: fraction of expected keywords found in proposed action.

    Score range: 0.0 (none matched) → 1.0 (all matched).
    """
    if not proposed_action or not operator_action.keywords:
        return 0.0
    proposed_lower = proposed_action.lower()
    matched = sum(1 for kw in operator_action.keywords if kw.lower() in proposed_lower)
    return round(matched / len(operator_action.keywords), 4)


def cost_in_range(cost: float, operator_action: OperatorAction) -> bool:
    """Check whether the proposed cost falls within the expected operator cost band."""
    lo, hi = operator_action.cost_range
    return lo <= cost <= hi


if __name__ == "__main__":
    oa = get_operator_action("energy_grid", "voltage_sag")
    print(f"Domain:    {oa.domain}")
    print(f"Anomaly:   {oa.anomaly_type}")
    print(f"Canonical: {oa.canonical_action}")
    print(f"Keywords:  {oa.keywords}")
    sample = "Switch Capacitor Bank for reactive compensation"
    score = action_keyword_match(sample, oa)
    print(f"Keyword match score for sample action: {score:.2%}")

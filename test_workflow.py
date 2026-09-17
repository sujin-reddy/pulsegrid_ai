import json
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from db_memory import get_past_interventions, init_db
from graph import run_pulsegrid_workflow
from state import AgentState


def test_nominal_grid_scenario():
    print("\n" + "=" * 60)
    print("TEST 1: NOMINAL GRID TELEMETRY SCENARIO")
    print("=" * 60)

    nominal_telemetry = {
        "substation_id": "SUB-WEST-02",
        "voltage_kv": 230.1,  # Nominal (218.5 - 241.5 kV)
        "frequency_hz": 60.01,  # Nominal (59.8 - 60.2 Hz)
        "load_mw": 110.5,  # Nominal (< 165 MW)
        "power_factor": 0.96,  # Nominal (>= 0.90)
        "temperature_c": 44.0,  # Nominal (< 80 °C)
    }

    result: AgentState = run_pulsegrid_workflow(nominal_telemetry)

    print(f"Status: {result.get('status')}")
    print(f"Anomaly Detected: {result.get('anomaly_detected')}")
    print(f"Messages Logged ({len(result.get('messages', []))}):")
    for msg in result.get("messages", []):
        print(f"  -> {msg}")

    assert result.get("anomaly_detected") is False, "Expected no anomaly for nominal telemetry"
    assert result.get("status") == "monitoring", "Expected status to remain 'monitoring'"
    print("✅ TEST 1 PASSED: Nominal scenario stayed in monitoring phase.")


def test_anomalous_grid_scenario():
    print("\n" + "=" * 60)
    print("TEST 2: SEVERE ANOMALY (VOLTAGE SAG & REACTIVE DEFICIT)")
    print("=" * 60)

    anomaly_telemetry = {
        "substation_id": "SUB-EAST-07",
        "voltage_kv": 212.4,  # Anomaly: severe sag below 218.5 kV
        "frequency_hz": 59.78,  # Anomaly: frequency decay below 59.80 Hz
        "load_mw": 168.2,  # Anomaly: overload > 165 MW
        "power_factor": 0.81,  # Anomaly: low power factor < 0.90
        "temperature_c": 83.5,  # Anomaly: overheating > 80 °C
    }

    result: AgentState = run_pulsegrid_workflow(anomaly_telemetry)

    print(f"Status: {result.get('status')}")
    print(f"Anomaly Detected: {result.get('anomaly_detected')}")
    print(f"Root Cause (Qwen 2.5):\n{result.get('root_cause')}")
    print(f"\nProposed Intervention:\n{json.dumps(result.get('proposed_intervention'), indent=2)}")
    print(f"\nRetrieved Memory Precedents Count: {len(result.get('past_interventions', []))}")
    print(f"\nMessage Trail ({len(result.get('messages', []))} entries):")
    for msg in result.get("messages", []):
        print(f"  • {msg}")

    assert result.get("anomaly_detected") is True, "Expected anomaly to be detected"
    assert result.get("status") == "complete", "Expected workflow status to reach 'complete'"
    assert result.get("root_cause") != "", "Expected non-empty root cause"
    assert "action" in result.get("proposed_intervention", {}), "Expected proposed action"
    print("✅ TEST 2 PASSED: Anomaly triggered full investigation, intervention, and completion.")


if __name__ == "__main__":
    init_db()
    test_nominal_grid_scenario()
    test_anomalous_grid_scenario()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! 🚀")
    print("=" * 60)

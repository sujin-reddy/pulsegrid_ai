"""Manual test suite verifying Domain Router Node and Multi-Domain Evaluation."""

import asyncio
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from main import run_pipeline
from state import AgentState


def test_datacenter_fast_tag():
    print("\n" + "=" * 60)
    print("MANUAL TEST 1: DATACENTER FAST-TAG RESOLUTION & GPU OVERHEAT")
    print("=" * 60)
    payload = {
        "domain": "datacenter",
        "device_id": "GPU-CLUSTER-01",
        "metrics": {"gpu_temp_c": 91.0},
    }
    res: AgentState = run_pipeline(payload)
    print(f"Resolved Domain: {res.get('resolved_domain')}")
    print(f"Anomaly Detected: {res.get('anomaly_detected')}")
    print(f"Status: {res.get('status')}")
    print("Messages:")
    for m in res.get("messages", []):
        print(f"  • {m}")

    assert res.get("resolved_domain") == "datacenter", "Expected resolved_domain == 'datacenter'"
    assert res.get("anomaly_detected") is True, "Expected anomaly_detected == True"
    assert any("gpu_temp_c" in m for m in res.get("messages", [])), "Expected violation message referencing gpu_temp_c"
    assert any("[FAST-PATH: fast-tag]" in m for m in res.get("messages", [])), "Expected fast-tag log in messages"
    print("✅ MANUAL TEST 1 PASSED: Fast-tag datacenter resolved & detected gpu_temp_c anomaly.")


def test_water_env_alias():
    print("\n" + "=" * 60)
    print("MANUAL TEST 2: WATER_ENV ALIAS RESOLUTION & RIVER LEVEL BREACH")
    print("=" * 60)
    # No explicit domain key; device_id is "RIVER-GAUGE-03"
    payload = {
        "device_id": "RIVER-GAUGE-03",
        "metrics": {"river_level_m": 14.8},
    }
    res: AgentState = run_pipeline(payload)
    print(f"Resolved Domain: {res.get('resolved_domain')}")
    print(f"Anomaly Detected: {res.get('anomaly_detected')}")
    print(f"Status: {res.get('status')}")
    print("Messages:")
    for m in res.get("messages", []):
        print(f"  • {m}")

    assert res.get("resolved_domain") == "water_env", "Expected resolved_domain == 'water_env'"
    assert res.get("anomaly_detected") is True, "Expected anomaly_detected == True"
    assert any("river_level_m" in m for m in res.get("messages", [])), "Expected violation message referencing river_level_m"
    assert any("[FAST-PATH: fast-alias]" in m for m in res.get("messages", [])), "Expected fast-alias log in messages"
    print("✅ MANUAL TEST 2 PASSED: Fast-alias water_env resolved & detected river_level_m anomaly.")


def test_llm_fallback_resolution():
    print("\n" + "=" * 60)
    print("MANUAL TEST 3: LLM FALLBACK RESOLUTION (NO DOMAIN, NO TAG ALIASES)")
    print("=" * 60)
    # Device ID and metric keys have no keywords matching tag aliases,
    # but semantic metrics refer to hydrological/environmental parameters.
    payload = {
        "device_id": "UNIT-771",
        "metrics": {"aquifer_salinity_ppt": 32.5, "estuary_turbidity_ntu": 88.0},
    }
    res: AgentState = run_pipeline(payload)
    print(f"Resolved Domain: {res.get('resolved_domain')}")
    print(f"Anomaly Detected: {res.get('anomaly_detected')}")
    print(f"Status: {res.get('status')}")
    print("Messages:")
    for m in res.get("messages", []):
        print(f"  • {m}")

    # Resolves via LLM fallback to water_env (or default energy_grid if LLM unavailable)
    assert res.get("resolved_domain") in ["water_env", "energy_grid"]
    assert any("[FALLBACK: llm-fallback]" in m for m in res.get("messages", [])), "Expected LLM fallback log in messages"
    print("✅ MANUAL TEST 3 PASSED: LLM fallback resolution verified.")


if __name__ == "__main__":
    test_datacenter_fast_tag()
    test_water_env_alias()
    test_llm_fallback_resolution()
    print("\n" + "=" * 60)
    print("ALL ROUTER MANUAL TESTS PASSED SUCCESSFULLY! 🚀")
    print("=" * 60)

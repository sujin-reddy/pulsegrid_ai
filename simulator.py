"""PulseGrid AI — Deterministic Telemetry Simulator.

Generates realistic telemetry payloads for all four registered domains
(energy_grid, datacenter, water_env, traffic).  All values are drawn from
seeded random distributions so experiments are fully reproducible.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Literal, Optional

# ---------------------------------------------------------------------------
# Domain-specific telemetry factories
# ---------------------------------------------------------------------------

DomainID = Literal["energy_grid", "datacenter", "water_env", "traffic"]

DEVICE_PREFIXES: Dict[str, List[str]] = {
    "energy_grid": ["SUB-NORTH", "SUB-SOUTH", "SUB-EAST", "SUB-WEST", "SUB-CENTRAL"],
    "datacenter": ["GPU-CLUSTER", "CPU-NODE", "K8S-POD", "HOST-RACK"],
    "water_env": ["RIVER-GAUGE", "DAM-SENSOR", "SPILLWAY-CTRL", "CATCHMENT-STA"],
    "traffic": ["INTERSECTION", "SIGNAL-CTL", "CORRIDOR", "TRANSIT-NODE"],
}

SCENARIO_LABELS: Dict[str, Dict[str, str]] = {
    "energy_grid": {
        "nominal": "grid_nominal",
        "anomaly": "grid_fault",
    },
    "datacenter": {
        "nominal": "dc_nominal",
        "anomaly": "dc_overheat",
    },
    "water_env": {
        "nominal": "hydro_nominal",
        "anomaly": "flood_warning",
    },
    "traffic": {
        "nominal": "traffic_nominal",
        "anomaly": "traffic_congestion",
    },
}


def _device_id(domain: str, rng: random.Random) -> str:
    prefix = rng.choice(DEVICE_PREFIXES.get(domain, ["DEVICE"]))
    return f"{prefix}-{rng.randint(1, 20):02d}"


def _energy_grid(anomaly: bool, rng: random.Random) -> Dict[str, Any]:
    if anomaly:
        return {
            "voltage_kv": round(rng.uniform(205.0, 217.0), 1),
            "frequency_hz": round(rng.uniform(59.60, 59.78), 2),
            "load_mw": round(rng.uniform(167.0, 180.0), 1),
            "power_factor": round(rng.uniform(0.76, 0.88), 2),
            "temperature_c": round(rng.uniform(82.0, 95.0), 1),
        }
    return {
        "voltage_kv": round(rng.uniform(222.0, 237.0), 1),
        "frequency_hz": round(rng.uniform(59.92, 60.08), 2),
        "load_mw": round(rng.uniform(85.0, 155.0), 1),
        "power_factor": round(rng.uniform(0.93, 0.99), 2),
        "temperature_c": round(rng.uniform(35.0, 72.0), 1),
    }


def _datacenter(anomaly: bool, rng: random.Random) -> Dict[str, Any]:
    if anomaly:
        return {
            "gpu_temp_c": round(rng.uniform(87.0, 98.0), 1),
            "memory_leak_pct": round(rng.uniform(91.0, 99.0), 1),
            "disk_io_wait_ms": round(rng.uniform(210.0, 450.0), 1),
        }
    return {
        "gpu_temp_c": round(rng.uniform(55.0, 80.0), 1),
        "memory_leak_pct": round(rng.uniform(30.0, 75.0), 1),
        "disk_io_wait_ms": round(rng.uniform(5.0, 150.0), 1),
    }


def _water_env(anomaly: bool, rng: random.Random) -> Dict[str, Any]:
    if anomaly:
        return {
            "river_level_m": round(rng.uniform(13.0, 18.5), 2),
            "flow_rate_m3s": round(rng.uniform(460.0, 700.0), 1),
            "rainfall_mm_24h": round(rng.uniform(80.0, 160.0), 1),
        }
    return {
        "river_level_m": round(rng.uniform(4.0, 11.0), 2),
        "flow_rate_m3s": round(rng.uniform(50.0, 380.0), 1),
        "rainfall_mm_24h": round(rng.uniform(0.0, 55.0), 1),
    }


def _traffic(anomaly: bool, rng: random.Random) -> Dict[str, Any]:
    """Traffic domain — no domain_config thresholds yet; still produces payload."""
    if anomaly:
        return {
            "congestion_index": round(rng.uniform(0.82, 1.0), 2),
            "queue_length_m": round(rng.uniform(420.0, 900.0), 1),
            "avg_speed_kmh": round(rng.uniform(3.0, 15.0), 1),
        }
    return {
        "congestion_index": round(rng.uniform(0.1, 0.55), 2),
        "queue_length_m": round(rng.uniform(10.0, 180.0), 1),
        "avg_speed_kmh": round(rng.uniform(40.0, 90.0), 1),
    }


_METRIC_FACTORIES = {
    "energy_grid": _energy_grid,
    "datacenter": _datacenter,
    "water_env": _water_env,
    "traffic": _traffic,
}


def generate_payload(
    domain: DomainID,
    anomaly: bool,
    rng: random.Random,
    unit_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build one TelemetryPayload-compatible dict for the given domain."""
    factory = _METRIC_FACTORIES.get(domain, _energy_grid)
    metrics = factory(anomaly, rng)
    return {
        "domain": domain,
        "device_id": _device_id(domain, rng),
        "metrics": metrics,
        "unit_map": unit_map or {},
    }


def generate_experiment_batch(
    n: int = 20,
    seed: int = 42,
    domain_weights: Optional[Dict[str, float]] = None,
    anomaly_rate: float = 0.5,
) -> List[Dict[str, Any]]:
    """Generate a reproducible batch of N telemetry events across all domains.

    Args:
        n:              Number of events to generate.
        seed:           RNG seed for full reproducibility.
        domain_weights: Relative sampling probability per domain. Defaults to uniform.
        anomaly_rate:   Fraction of events that are anomalous (default 0.5).

    Returns:
        List of dicts, each containing keys:
            event_id, domain, device_id, scenario, is_anomaly, payload
    """
    rng = random.Random(seed)
    domains = list(_METRIC_FACTORIES.keys())
    weights = [domain_weights.get(d, 1.0) for d in domains] if domain_weights else None

    events: List[Dict[str, Any]] = []
    for i in range(n):
        domain: DomainID = rng.choices(domains, weights=weights, k=1)[0]  # type: ignore[assignment]
        is_anomaly = rng.random() < anomaly_rate
        payload = generate_payload(domain, is_anomaly, rng)
        scenario = SCENARIO_LABELS[domain]["anomaly" if is_anomaly else "nominal"]
        events.append(
            {
                "event_id": i + 1,
                "domain": domain,
                "device_id": payload["device_id"],
                "scenario": scenario,
                "is_anomaly": is_anomaly,
                "payload": payload,
            }
        )
    return events


if __name__ == "__main__":
    batch = generate_experiment_batch(n=5, seed=42)
    for ev in batch:
        print(f"[{ev['event_id']}] {ev['domain']} | {ev['scenario']} | anomaly={ev['is_anomaly']}")
        print(f"      device={ev['device_id']} metrics={ev['payload']['metrics']}")

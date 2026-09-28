"""Domain Configuration Registry for PulseGrid AI multi-domain orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field


class DomainClassification(BaseModel):
    """Structured output schema for LLM-based fallback domain classification."""

    domain_id: Literal["energy_grid", "datacenter", "water_env", "traffic"] = Field(
        description="Classify the operating domain based on metric names, tags, or device identifier."
    )
    confidence: float = Field(
        default=0.8,
        description="Classification confidence score between 0.0 and 1.0."
    )


@dataclass
class DomainConfig:
    """Configuration schema defining operating domain attributes and thresholds."""

    domain_id: str  # "energy_grid", "datacenter", "water_env", "traffic"
    display_name: str
    metric_thresholds: Dict[str, dict]  # e.g. {"gpu_temp_c": {"op": ">", "value": 85.0, "label": "GPU thermal overload"}}
    anomaly_prompt_context: str  # short domain-specific framing injected into the diagnostic LLM prompt
    tag_aliases: List[str] = field(default_factory=list)  # strings that map incoming payloads to this domain


DOMAIN_REGISTRY: Dict[str, DomainConfig] = {
    "energy_grid": DomainConfig(
        domain_id="energy_grid",
        display_name="Energy Grid Infrastructure",
        metric_thresholds={
            "voltage_kv": {
                "op": "<",
                "value": 218.5,
                "label": "Undervoltage sag",
                "unit": "kV",
                "secondary_op": ">",
                "secondary_value": 241.5,
                "secondary_label": "Overvoltage surge",
            },
            "temperature_c": {
                "op": ">",
                "value": 80.0,
                "label": "Core temperature alarm",
                "unit": "°C",
            },
            "frequency_hz": {
                "op": "<",
                "value": 59.80,
                "label": "Frequency decay",
                "unit": "Hz",
            },
            "load_mw": {
                "op": ">",
                "value": 165.0,
                "label": "Feeder capacity overload",
                "unit": "MW",
            },
            "power_factor": {
                "op": "<",
                "value": 0.90,
                "label": "Reactive deficit PF",
                "unit": "",
            },
            "value": {
                "op": ">",
                "value": 80.0,
                "label": "Metric value",
                "unit": "",
            },
        },
        anomaly_prompt_context=(
            "You are the PulseGrid AI Senior Power Systems Diagnostic Agent. "
            "Analyze electrical grid telemetry, transformer loadings, voltage stability, and reactive power dynamics."
        ),
        tag_aliases=[
            "energy_grid",
            "grid",
            "substation",
            "sub-",
            "elec",
            "voltage",
            "power",
            "transformer",
            "feeder",
            "energy",
            "reactive",
        ],
    ),
    "datacenter": DomainConfig(
        domain_id="datacenter",
        display_name="Data Center & Cloud Infrastructure",
        metric_thresholds={
            "gpu_temp_c": {
                "op": ">",
                "value": 85.0,
                "label": "GPU thermal overload",
                "unit": "°C",
            },  # Placeholder-tunable engineering default
            "memory_leak_pct": {
                "op": ">",
                "value": 90.0,
                "label": "Memory utilization threshold breach",
                "unit": "%",
            },  # Placeholder-tunable engineering default
            "disk_io_wait_ms": {
                "op": ">",
                "value": 200.0,
                "label": "Storage I/O latency stall",
                "unit": "ms",
            },  # Placeholder-tunable engineering default
        },
        anomaly_prompt_context=(
            "You are a DevOps and AI Cloud Infrastructure Site Reliability Engineer. "
            "Investigate pod crashes, memory leaks, GPU thermal throttling, and cluster storage bottlenecks."
        ),
        tag_aliases=[
            "datacenter",
            "data_center",
            "gpu",
            "pod",
            "aws",
            "cloud",
            "cluster",
            "server",
            "node",
            "cpu",
            "host",
            "k8s",
        ],
    ),
    "water_env": DomainConfig(
        domain_id="water_env",
        display_name="Hydrological & Water Environment",
        metric_thresholds={
            "river_level_m": {
                "op": ">",
                "value": 12.5,
                "label": "River stage breach flood warning",
                "unit": "m",
            },  # Tunable hydrological breach threshold
            "flow_rate_m3s": {
                "op": ">",
                "value": 450.0,
                "label": "Excessive discharge rate",
                "unit": "m³/s",
            },  # Tunable river discharge threshold
            "rainfall_mm_24h": {
                "op": ">",
                "value": 75.0,
                "label": "Heavy precipitation alert",
                "unit": "mm",
            },  # Tunable precipitation limit
        },
        anomaly_prompt_context=(
            "You are a hydrological disaster management and flood operations specialist. "
            "Analyze catchment sensor readings, river gauge elevations, spillway flows, and flash flood indicators."
        ),
        tag_aliases=[
            "water_env",
            "water",
            "river",
            "flood",
            "gauge",
            "dam",
            "valve",
            "hydrology",
            "spillway",
            "catchment",
            "drainage",
        ],
    ),
    "traffic": DomainConfig(
        domain_id="traffic",
        display_name="Urban Traffic Network",
        metric_thresholds={},  # TODO: Traffic metrics (e.g. congestion_index, queue_length_m) out of scope for this step
        anomaly_prompt_context=(
            "You are an urban traffic orchestration agent. "
            "Manage signal timings, emergency corridors, and intersection bottlenecks."
        ),
        tag_aliases=[
            "traffic",
            "intersection",
            "signal",
            "vehicle",
            "transit",
            "road",
            "corridor",
        ],
    ),
}


def get_domain_config(domain_id: str) -> DomainConfig:
    """Retrieve domain configuration by domain_id with fallback to energy_grid."""
    return DOMAIN_REGISTRY.get(domain_id, DOMAIN_REGISTRY["energy_grid"])

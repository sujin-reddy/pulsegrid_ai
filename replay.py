"""PulseGrid AI — CSV Replay Module.

Reads a CSV file of telemetry records and replays them through the LangGraph
pipeline, producing a results DataFrame.  Designed for both offline batch use
and Streamlit sidebar integration in app.py.

Expected CSV columns (flexible — other columns are passed through as metrics):
    domain          str   (energy_grid | datacenter | water_env | traffic)
    device_id       str   optional — auto-generated if absent
    <metric_cols>   float any numeric column not in RESERVED_COLS is treated as a metric

Example minimal CSV:
    domain,device_id,voltage_kv,temperature_c,frequency_hz
    energy_grid,SUB-NORTH-01,213.5,84.0,59.75
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESERVED_COLS = {"domain", "device_id", "substation_id", "unit_map", "metrics", "event_id"}

EXAMPLE_CSV_CONTENT = """\
domain,device_id,voltage_kv,temperature_c,frequency_hz,load_mw,power_factor
energy_grid,SUB-NORTH-01,213.5,84.0,59.75,170.0,0.83
energy_grid,SUB-SOUTH-07,229.0,55.0,59.97,120.0,0.96
datacenter,GPU-CLUSTER-03,,,,,
datacenter,GPU-CLUSTER-03,,,,,
water_env,RIVER-GAUGE-05,,,,,
"""


# ---------------------------------------------------------------------------
# CSV → payload conversion
# ---------------------------------------------------------------------------


def _row_to_payload(row: pd.Series) -> Dict[str, Any]:
    """Convert a DataFrame row to a TelemetryPayload-compatible dict."""
    domain = str(row.get("domain", "energy_grid")).strip() or "energy_grid"
    device_id = str(
        row.get("device_id", row.get("substation_id", f"REPLAY-{int(time.time())%9999:04d}"))
    ).strip()

    metrics: Dict[str, float] = {}
    for col, val in row.items():
        if col in RESERVED_COLS:
            continue
        try:
            fval = float(val)
            if not pd.isna(fval):
                metrics[str(col)] = fval
        except (ValueError, TypeError):
            pass  # skip non-numeric or empty cells

    return {
        "domain": domain,
        "device_id": device_id,
        "metrics": metrics,
        "unit_map": {},
    }


# ---------------------------------------------------------------------------
# Core replay function
# ---------------------------------------------------------------------------


def replay_csv(
    csv_path: str | Path,
    run_pipeline_fn: Callable[[Dict[str, Any]], Any],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    delay_s: float = 0.0,
) -> pd.DataFrame:
    """Replay all rows in a CSV file through the PulseGrid pipeline.

    Args:
        csv_path:           Path to input CSV.
        run_pipeline_fn:    Callable matching ``main.run_pipeline`` signature.
        progress_callback:  Optional fn(current_row, total_rows) for UI progress.
        delay_s:            Optional sleep between events (useful for Streamlit live demo).

    Returns:
        DataFrame with columns:
            row_index, domain, device_id, anomaly_detected, resolved_domain,
            root_cause, action, cost, status, latency_ms, error
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df_in = pd.read_csv(csv_path)
    total = len(df_in)
    results: List[Dict[str, Any]] = []

    for idx, row in df_in.iterrows():
        payload = _row_to_payload(row)
        t0 = time.perf_counter()
        error_msg = ""
        state: Dict[str, Any] = {}

        try:
            state = run_pipeline_fn(payload)
        except Exception as exc:
            error_msg = str(exc)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        proposed = state.get("proposed_intervention", {}) if state else {}

        results.append(
            {
                "row_index": int(idx),  # type: ignore[arg-type]
                "domain": payload["domain"],
                "device_id": payload["device_id"],
                "anomaly_detected": state.get("anomaly_detected", False),
                "resolved_domain": state.get("resolved_domain", ""),
                "root_cause": (state.get("root_cause", ""))[:200],
                "action": proposed.get("action", ""),
                "cost": proposed.get("cost", 0.0),
                "status": state.get("status", "error" if error_msg else ""),
                "latency_ms": latency_ms,
                "error": error_msg,
            }
        )

        if progress_callback:
            progress_callback(int(idx) + 1, total)  # type: ignore[arg-type]

        if delay_s > 0:
            time.sleep(delay_s)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Stream (generator) variant for live Streamlit updates
# ---------------------------------------------------------------------------


def replay_csv_stream(
    csv_path: str | Path,
    run_pipeline_fn: Callable[[Dict[str, Any]], Any],
) -> Iterator[Dict[str, Any]]:
    """Yield one result dict per CSV row — for live Streamlit st.dataframe updates."""
    csv_path = Path(csv_path)
    df_in = pd.read_csv(csv_path)
    for idx, row in df_in.iterrows():
        payload = _row_to_payload(row)
        t0 = time.perf_counter()
        error_msg = ""
        state: Dict[str, Any] = {}
        try:
            state = run_pipeline_fn(payload)
        except Exception as exc:
            error_msg = str(exc)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        proposed = state.get("proposed_intervention", {}) if state else {}
        yield {
            "row_index": int(idx),  # type: ignore[arg-type]
            "domain": payload["domain"],
            "device_id": payload["device_id"],
            "anomaly_detected": state.get("anomaly_detected", False),
            "resolved_domain": state.get("resolved_domain", ""),
            "root_cause": (state.get("root_cause", ""))[:200],
            "action": proposed.get("action", ""),
            "cost": proposed.get("cost", 0.0),
            "status": state.get("status", "error" if error_msg else ""),
            "latency_ms": latency_ms,
            "error": error_msg,
        }


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from main import run_pipeline

    parser = argparse.ArgumentParser(description="Replay a CSV telemetry file through PulseGrid AI.")
    parser.add_argument("csv", help="Path to input CSV file")
    parser.add_argument("--out", default="results/replay_output.csv", help="Output CSV path")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between events (seconds)")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def _progress(cur: int, tot: int) -> None:
        print(f"\r  Replaying {cur}/{tot}...", end="", flush=True)

    print(f"Replaying CSV: {args.csv}")
    df_out = replay_csv(args.csv, run_pipeline, progress_callback=_progress, delay_s=args.delay)
    print(f"\nComplete. {len(df_out)} events processed.")
    df_out.to_csv(args.out, index=False)
    print(f"Results saved → {args.out}")
    print(df_out.to_string(index=False))

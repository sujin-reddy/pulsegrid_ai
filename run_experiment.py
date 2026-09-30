"""PulseGrid AI — Experiment Runner.

Generates N reproducible telemetry events via simulator.py, runs each through
the full LangGraph pipeline, evaluates agent outputs against the operator
action catalogue, and writes three result artefacts to results/:

  events.csv               -- per-event detail (N rows)
  summary_by_scenario.csv  -- aggregated metrics grouped by scenario label
  summary_by_domain.csv    -- aggregated metrics grouped by domain

Usage:
    python run_experiment.py --n 20 --seed 42
    python run_experiment.py --n 20 --seed 42 --no-llm   # forces rule-based fallback
    python run_experiment.py --n 50 --seed 7 --out results/run2
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

# Add project root to path so script is runnable from any cwd
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from db_memory import init_db
from operator_actions import action_keyword_match, cost_in_range, get_operator_action
from simulator import generate_experiment_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_llm_patch() -> None:
    """Monkey-patch _get_llm in nodes.py to always raise — forces rule fallback."""
    import nodes  # noqa: PLC0415

    def _broken_llm(model_name: str = "qwen2.5"):  # noqa: ANN202
        raise RuntimeError("[no-llm mode] LLM call suppressed by --no-llm flag.")

    nodes._get_llm = _broken_llm  # type: ignore[attr-defined]
    print("⚠️  --no-llm mode: all LLM calls suppressed; using deterministic rule fallback.")


def _run_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the pipeline for a single payload, returning the final AgentState."""
    from main import run_pipeline  # imported late so monkey-patch lands first
    return run_pipeline(payload)


def _evaluate_event(
    event: Dict[str, Any],
    state: Dict[str, Any],
    latency_ms: float,
    error: str,
) -> Dict[str, Any]:
    """Combine event metadata, agent outputs, and evaluation scores into one flat dict."""
    domain = event["domain"]
    is_anomaly = event["is_anomaly"]
    resolved = state.get("resolved_domain", "")
    detected = state.get("anomaly_detected", False)
    proposed = state.get("proposed_intervention", {})
    action_str = proposed.get("action", "")
    cost = float(proposed.get("cost", 0.0))
    root_cause = str(state.get("root_cause", ""))[:300]

    # Domain routing accuracy
    domain_correct = int(resolved == domain)

    # Detection accuracy (TP/TN/FP/FN)
    if is_anomaly and detected:
        det_label = "TP"
    elif not is_anomaly and not detected:
        det_label = "TN"
    elif not is_anomaly and detected:
        det_label = "FP"
    else:
        det_label = "FN"

    # Keyword match score (only meaningful when anomaly was correctly detected)
    keyword_score: float = 0.0
    cost_ok: bool = False
    if detected and is_anomaly:
        # Find anomaly_type via nodes helper (import lazily)
        from nodes import _extract_anomaly_type  # noqa: PLC0415
        anomaly_type = _extract_anomaly_type(event["payload"])
        oa = get_operator_action(domain, anomaly_type)
        if oa:
            keyword_score = action_keyword_match(action_str, oa)
            cost_ok = cost_in_range(cost, oa)

    return {
        "event_id": event["event_id"],
        "domain": domain,
        "device_id": event["device_id"],
        "scenario": event["scenario"],
        "is_anomaly": is_anomaly,
        "resolved_domain": resolved,
        "domain_correct": domain_correct,
        "anomaly_detected": detected,
        "detection_label": det_label,
        "root_cause_snippet": root_cause[:120].replace("\n", " "),
        "proposed_action": action_str[:160],
        "cost_usd": cost,
        "cost_in_range": cost_ok,
        "keyword_match_score": keyword_score,
        "pipeline_status": state.get("status", "error" if error else ""),
        "latency_ms": latency_ms,
        "error": error,
    }


def _print_row(row: Dict[str, Any], verbose: bool = False) -> None:
    icon = "[A]" if row["is_anomaly"] else "[N]"
    det = row["detection_label"]
    kw = row["keyword_match_score"]
    print(
        f"  [{row['event_id']:02d}] {icon} {row['domain']:<12} | "
        f"scenario={row['scenario']:<20} | det={det} | "
        f"domain_ok={bool(row['domain_correct'])} | "
        f"kw={kw:.2f} | {row['latency_ms']:.0f}ms"
    )
    if verbose and row.get("error"):
        print(f"       [WARN] error: {row['error']}")


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Group df by group_col and compute summary metrics."""
    agg = (
        df.groupby(group_col)
        .agg(
            n_events=("event_id", "count"),
            n_anomalies=("is_anomaly", "sum"),
            TP=("detection_label", lambda s: (s == "TP").sum()),
            TN=("detection_label", lambda s: (s == "TN").sum()),
            FP=("detection_label", lambda s: (s == "FP").sum()),
            FN=("detection_label", lambda s: (s == "FN").sum()),
            domain_accuracy=("domain_correct", "mean"),
            avg_keyword_match=("keyword_match_score", "mean"),
            avg_cost_usd=("cost_usd", "mean"),
            pct_cost_in_range=("cost_in_range", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            n_errors=("error", lambda s: (s != "").sum()),
        )
        .reset_index()
    )

    # Derived metrics
    denom_prec = agg["TP"] + agg["FP"]
    denom_rec = agg["TP"] + agg["FN"]
    agg["precision"] = (agg["TP"] / denom_prec.replace(0, float("nan"))).round(4)
    agg["recall"] = (agg["TP"] / denom_rec.replace(0, float("nan"))).round(4)

    denom_f1 = agg["precision"] + agg["recall"]
    agg["f1_score"] = (
        (2 * agg["precision"] * agg["recall"]) / denom_f1.replace(0, float("nan"))
    ).round(4)

    agg["detection_accuracy"] = (
        (agg["TP"] + agg["TN"]) / agg["n_events"]
    ).round(4)

    # Round floats
    for col in ["domain_accuracy", "avg_keyword_match", "pct_cost_in_range",
                "avg_latency_ms", "avg_cost_usd"]:
        agg[col] = agg[col].round(4)

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(
    n: int = 20,
    seed: int = 42,
    out_dir: str = "results",
    no_llm: bool = False,
    verbose: bool = False,
    anomaly_rate: float = 0.5,
) -> Dict[str, Path]:
    """Run the full experiment, write CSVs, and return paths.

    Returns:
        dict with keys: events_csv, summary_scenario_csv, summary_domain_csv
    """
    init_db()
    out_path = _ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    if no_llm:
        _no_llm_patch()

    # Generate reproducible event batch
    print("[PulseGrid AI Experiment Runner]")
    print(f"   N={n}  seed={seed}  anomaly_rate={anomaly_rate}  no_llm={no_llm}")
    print(f"   Output -> {out_path.resolve()}\n")

    events = generate_experiment_batch(n=n, seed=seed, anomaly_rate=anomaly_rate)
    rows: List[Dict[str, Any]] = []

    t_total = time.perf_counter()
    for event in events:
        payload = event["payload"]
        t0 = time.perf_counter()
        error = ""
        state: Dict[str, Any] = {}
        try:
            state = _run_event(payload)
        except Exception as exc:
            error = str(exc)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        row = _evaluate_event(event, state, latency_ms, error)
        rows.append(row)
        _print_row(row, verbose=verbose)

    elapsed = round(time.perf_counter() - t_total, 2)
    print(f"\n[OK] {n} events processed in {elapsed}s  ({elapsed/n*1000:.0f}ms avg)")

    # Write events.csv
    df_events = pd.DataFrame(rows)
    events_path = out_path / "events.csv"
    df_events.to_csv(events_path, index=False)
    print(f"[CSV] events.csv              -> {events_path}")

    # Write summary_by_scenario.csv
    df_scenario = _aggregate(df_events, "scenario")
    scenario_path = out_path / "summary_by_scenario.csv"
    df_scenario.to_csv(scenario_path, index=False)
    print(f"[CSV] summary_by_scenario.csv -> {scenario_path}")

    # Write summary_by_domain.csv
    df_domain = _aggregate(df_events, "domain")
    domain_path = out_path / "summary_by_domain.csv"
    df_domain.to_csv(domain_path, index=False)
    print(f"[CSV] summary_by_domain.csv   -> {domain_path}")

    # Console summary
    print("\n── Summary by Domain ─────────────────────────────────────────")
    print(
        df_domain[["domain", "n_events", "detection_accuracy", "precision",
                   "recall", "f1_score", "avg_keyword_match", "avg_latency_ms"]].to_string(index=False)
    )
    print("\n── Summary by Scenario ───────────────────────────────────────")
    print(
        df_scenario[["scenario", "n_events", "TP", "TN", "FP", "FN",
                     "detection_accuracy", "f1_score"]].to_string(index=False)
    )

    return {
        "events_csv": events_path,
        "summary_scenario_csv": scenario_path,
        "summary_domain_csv": domain_path,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PulseGrid AI Experiment Runner — generates evaluation CSVs."
    )
    parser.add_argument("--n", type=int, default=20, help="Number of events (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--out", default="results", help="Output directory (default: results/)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Suppress all LLM calls; use deterministic rule-based fallback only",
    )
    parser.add_argument("--verbose", action="store_true", help="Print error details per event")
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.5,
        help="Fraction of events that are anomalies (default: 0.5)",
    )
    args = parser.parse_args()

    run_experiment(
        n=args.n,
        seed=args.seed,
        out_dir=args.out,
        no_llm=args.no_llm,
        verbose=args.verbose,
        anomaly_rate=args.anomaly_rate,
    )

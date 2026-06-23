"""
Consolidate per-scenario aggregated JSON files into a single canonical CSV.

Each scenario's `*_aggregated.json` contains per-scheduler, per-metric statistics
produced by the statistical benchmark.  This script reads all matching files from
a given directory and writes one CSV with all scenarios and schedulers, making the
benchmark results self-contained in a single artifact rather than spread across
per-scenario files.

Usage:
    .venv/bin/python -m assessment.scripts.consolidate_benchmark_results \\
        --input-dir results/stage3_extended \\
        --output results/stage3_extended/consolidated.csv

The canonical ten benchmark scenarios are selected when --canonical is given,
ignoring any extra JSON files in the directory.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Canonical scenario keys in display order (matches REFERENCE_All_Workload_Scenarios.md)
CANONICAL_SCENARIOS: List[Tuple[str, str]] = [
    ("alarm_under_burst",                       "Alarm Under Burst"),
    ("alarm_under_burst_phased",                "Alarm Under Burst (Phased)"),
    ("device_monopolization",                   "Device Monopolization"),
    ("device_monopolization_sweep",             "Device Monopolization (Sweep)"),
    ("multi_zone_emergency",                    "Multi-Zone Emergency"),
    ("multi_zone_emergency_cascade",            "Multi-Zone Emergency (Cascade)"),
    ("alarm_flood_attack",                      "Alarm Flood Attack"),
    ("alarm_malfunction_surge",                 "Alarm Malfunction Surge (Forced Drops)"),
    ("legit_extreme_emergency",                 "Legitimate Extreme Emergency"),
    ("alarm_load_near_saturation_constrained",  "Alarm Load Regime (ρ≈0.95 Token-Constrained)"),
]

CORE_METRICS = [
    "alarm_avg_latency",
    "alarm_p95_latency",
    "band_0_fairness",
    "band_1_fairness",
    "band_2_fairness",
    "band_3_fairness",
    "device_latency_fairness",
    "device_throughput_fairness",
    "min_device_rate",
    "avg_device_rate",
    "high_avg_latency",
    "high_p95_latency",
    "avg_waiting_time",
    "p95_waiting_time",
]

PROTECTION_METRICS = [
    "alarm_dropped",
    "alarm_dropped_rate",
    "protection_enabled",
    "alarm_source_fairness",
    "alarm_source_count",
    "alarm_source_latency_cv",
    "alarm_protection_activations",
    "alarm_protection_deactivations",
]


def _load_scenario(input_dir: str, scenario_key: str) -> Optional[dict]:
    path = Path(input_dir) / f"{scenario_key}_aggregated.json"
    if not path.exists():
        print(f"  [warn] missing: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def consolidate(
    input_dir: str,
    output_path: str,
    canonical: bool = False,
) -> None:
    """Read per-scenario JSON files and write a unified CSV."""
    if canonical:
        scenario_list = CANONICAL_SCENARIOS
    else:
        files = sorted(Path(input_dir).glob("*_aggregated.json"))
        scenario_list = [
            (f.stem.replace("_aggregated", ""), f.stem.replace("_aggregated", "").replace("_", " ").title())
            for f in files
        ]

    schedulers: Optional[List[str]] = None
    rows = []

    for scenario_key, scenario_name in scenario_list:
        agg = _load_scenario(input_dir, scenario_key)
        if agg is None:
            continue

        if schedulers is None:
            schedulers = [k for k in agg.keys() if k not in ("phase_metrics", "phase_boundaries")]

        for sched in schedulers:
            if sched not in agg:
                continue
            row = {
                "scenario_key": scenario_key,
                "scenario_name": scenario_name,
                "scheduler": sched,
            }

            for metric in CORE_METRICS:
                if metric in agg[sched]:
                    row[f"{metric}_mean"] = agg[sched][metric]["mean"]
                    row[f"{metric}_std"] = agg[sched][metric]["std"]
                    row[f"{metric}_ci_lower"] = agg[sched][metric]["ci_lower"]
                    row[f"{metric}_ci_upper"] = agg[sched][metric]["ci_upper"]
                else:
                    row[f"{metric}_mean"] = ""
                    row[f"{metric}_std"] = ""
                    row[f"{metric}_ci_lower"] = ""
                    row[f"{metric}_ci_upper"] = ""

            # Protection metrics for all schedulers: use 0.0 when the metric exists
            # (non-TRIAGE/4 schedulers produce 0 drops and no activations), NA only
            # when the key is absent from the JSON.
            for metric in PROTECTION_METRICS:
                if metric in agg[sched]:
                    row[f"{metric}_mean"] = agg[sched][metric]["mean"]
                    row[f"{metric}_std"] = agg[sched][metric]["std"]
                else:
                    row[f"{metric}_mean"] = "NA"
                    row[f"{metric}_std"] = "NA"

            rows.append(row)

    if not rows:
        print("No data found — no CSV written.")
        return

    fieldnames = ["scenario_key", "scenario_name", "scheduler"]
    for m in CORE_METRICS:
        fieldnames.extend([f"{m}_mean", f"{m}_std", f"{m}_ci_lower", f"{m}_ci_upper"])
    for m in PROTECTION_METRICS:
        fieldnames.extend([f"{m}_mean", f"{m}_std"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(col, "")) for col in fieldnames) + "\n")

    print(f"Consolidated {len(rows)} rows across {len(scenario_list)} scenarios → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate per-scenario benchmark JSON results into one CSV."
    )
    parser.add_argument(
        "--input-dir",
        default="results/stage3_extended",
        help="Directory containing *_aggregated.json files (default: results/stage3_extended)",
    )
    parser.add_argument(
        "--output",
        default="results/stage3_extended/consolidated.csv",
        help="Output CSV path (default: results/stage3_extended/consolidated.csv)",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Only include the ten canonical scenarios (ignore extras in directory)",
    )
    args = parser.parse_args()
    consolidate(args.input_dir, args.output, canonical=args.canonical)


if __name__ == "__main__":
    main()

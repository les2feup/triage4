#!/usr/bin/env python3
"""
Compare Desktop vs RPi4 benchmark results.

Loads aggregated JSON results from both platforms and verifies
numerical equivalence within floating-point tolerance.
"""

import json
from pathlib import Path
import sys

# Configuration
DESKTOP_DIR = Path(__file__).resolve().parents[2] / "results" / "desktop_baseline"
RPI4_DIR = Path(__file__).resolve().parents[2] / "results" / "rpi4_validation"

# Key metrics to compare (most scientifically relevant)
KEY_METRICS = [
    "alarm_avg_latency",
    "alarm_p95_latency",
    "band_1_fairness",  # HIGH band fairness
    "device_latency_fairness",
    "high_avg_latency",
    "high_p95_latency",
    "min_device_rate",
]

# Schedulers to compare
SCHEDULERS = ["TRIAGE/4", "Strict", "FIFO"]

# Tolerance threshold (percentage)
TOLERANCE_PCT = 1.0  # 1% difference threshold for warnings


def load_json(path: Path) -> dict:
    """Load JSON file and return parsed data."""
    with open(path, "r") as f:
        return json.load(f)


def compare_metric(desktop_val: float, rpi4_val: float) -> tuple[float, str]:
    """
    Compare two metric values and return (diff_pct, status).

    Returns:
        (diff_pct, status) where status is 'match', 'warn', or 'error'
    """
    if desktop_val == 0 and rpi4_val == 0:
        return 0.0, "match"

    if desktop_val == 0:
        # Can't compute percentage diff, use absolute
        return abs(rpi4_val), "warn" if abs(rpi4_val) > 0.001 else "match"

    diff_pct = abs(desktop_val - rpi4_val) / abs(desktop_val) * 100

    if diff_pct > TOLERANCE_PCT:
        return diff_pct, "warn"
    elif diff_pct > 0.01:  # >0.01% but <=1%
        return diff_pct, "match"
    else:  # Essentially identical
        return diff_pct, "match"


def compare_scenario(scenario_name: str, desktop_data: dict, rpi4_data: dict) -> dict:
    """
    Compare a single scenario across both platforms.

    Returns summary statistics.
    """
    results = {
        "scenario": scenario_name,
        "schedulers": {},
        "total_comparisons": 0,
        "matches": 0,
        "warnings": 0,
    }

    for scheduler in SCHEDULERS:
        if scheduler not in desktop_data or scheduler not in rpi4_data:
            continue

        desktop_sched = desktop_data[scheduler]
        rpi4_sched = rpi4_data[scheduler]

        results["schedulers"][scheduler] = {}

        for metric in KEY_METRICS:
            if metric not in desktop_sched or metric not in rpi4_sched:
                continue

            desktop_mean = desktop_sched[metric]["mean"]
            rpi4_mean = rpi4_sched[metric]["mean"]

            diff_pct, status = compare_metric(desktop_mean, rpi4_mean)

            results["schedulers"][scheduler][metric] = {
                "desktop": desktop_mean,
                "rpi4": rpi4_mean,
                "diff_pct": diff_pct,
                "status": status,
            }

            results["total_comparisons"] += 1
            if status == "match":
                results["matches"] += 1
            else:
                results["warnings"] += 1

    return results


def print_scenario_results(results: dict, verbose: bool = False):
    """Print comparison results for a scenario."""
    scenario = results["scenario"]
    total = results["total_comparisons"]
    matches = results["matches"]
    warnings = results["warnings"]

    if warnings == 0:
        status_icon = "✅"
    else:
        status_icon = "⚠️"

    print(f"\n{status_icon} {scenario}")
    print(f"   Comparisons: {total} | Matches: {matches} | Warnings: {warnings}")

    if verbose or warnings > 0:
        for scheduler, metrics in results["schedulers"].items():
            print(f"\n   [{scheduler}]")
            for metric, data in metrics.items():
                status_mark = "✓" if data["status"] == "match" else "⚠"
                print(f"     {status_mark} {metric}:")
                print(f"         Desktop: {data['desktop']:.8f}")
                print(f"         RPi4:    {data['rpi4']:.8f}")
                print(f"         Diff:    {data['diff_pct']:.6f}%")


def main():
    """Main comparison routine."""
    print("=" * 60)
    print("Desktop vs RPi4 Benchmark Comparison")
    print("=" * 60)

    # Check directories exist
    if not DESKTOP_DIR.exists():
        print(f"ERROR: Desktop results not found: {DESKTOP_DIR}")
        sys.exit(1)

    if not RPI4_DIR.exists():
        print(f"ERROR: RPi4 results not found: {RPI4_DIR}")
        sys.exit(1)

    # Find all scenario files
    desktop_files = sorted(DESKTOP_DIR.glob("*_aggregated.json"))
    rpi4_files = sorted(RPI4_DIR.glob("*_aggregated.json"))

    print(f"\nDesktop scenarios found: {len(desktop_files)}")
    print(f"RPi4 scenarios found:    {len(rpi4_files)}")

    # Build file maps
    desktop_map = {f.stem.replace("_aggregated", ""): f for f in desktop_files}
    rpi4_map = {f.stem.replace("_aggregated", ""): f for f in rpi4_files}

    # Find common scenarios
    common_scenarios = set(desktop_map.keys()) & set(rpi4_map.keys())
    missing_on_rpi4 = set(desktop_map.keys()) - set(rpi4_map.keys())
    missing_on_desktop = set(rpi4_map.keys()) - set(desktop_map.keys())

    if missing_on_rpi4:
        print(f"\n⚠️  Missing on RPi4: {', '.join(sorted(missing_on_rpi4))}")
    if missing_on_desktop:
        print(f"\n⚠️  Missing on Desktop: {', '.join(sorted(missing_on_desktop))}")

    print(f"\nComparing {len(common_scenarios)} common scenarios...")
    print("-" * 60)

    # Compare each scenario
    all_results = []
    total_comparisons = 0
    total_matches = 0
    total_warnings = 0

    for scenario in sorted(common_scenarios):
        desktop_data = load_json(desktop_map[scenario])
        rpi4_data = load_json(rpi4_map[scenario])

        results = compare_scenario(scenario, desktop_data, rpi4_data)
        all_results.append(results)

        total_comparisons += results["total_comparisons"]
        total_matches += results["matches"]
        total_warnings += results["warnings"]

        # Print with details if warnings exist
        print_scenario_results(results, verbose=False)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Scenarios compared: {len(common_scenarios)}")
    print(f"Total metric comparisons: {total_comparisons}")
    print(f"Matches (<{TOLERANCE_PCT}% diff): {total_matches}")
    print(f"Warnings (>{TOLERANCE_PCT}% diff): {total_warnings}")

    if total_warnings == 0:
        print("\n✅ ALL RESULTS MATCH WITHIN TOLERANCE")
        print("   Desktop and RPi4 results are numerically equivalent.")
        print("   Safe to use either dataset for publication.")
    else:
        print(f"\n⚠️  {total_warnings} COMPARISONS EXCEED {TOLERANCE_PCT}% TOLERANCE")
        print("   Review warnings above for potential numerical instability.")
        print("   Consider investigating affected metrics before publication.")

    return 0 if total_warnings == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

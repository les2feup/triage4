"""
TRIAGE/4 Stress Testing Benchmark.

Tests TRIAGE/4 performance under extreme conditions to identify scalability limits
and graceful degradation behavior.

Four stress tests:
1. Device Count Scalability (10→500 devices)
2. Alarm Rate Surge (1%→50% alarm ratio)
3. Overload Graceful Degradation (ρ=0.9→1.5)
4. Token Budget Sensitivity (Q_H sweep)

Usage:
    # Run all stress tests
    python benchmarks/stress_benchmark.py --all

    # Run specific test
    python benchmarks/stress_benchmark.py --test device_scalability

    # Quick mode (fewer data points)
    python benchmarks/stress_benchmark.py --test device_scalability --quick
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assessment.baselines import FIFOScheduler, StrictPriorityScheduler
from assessment.metrics import compute_all_metrics
from triage4 import TRIAGE4Config, TRIAGE4Scheduler
from assessment.workloads import Workload

# Plot styling
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    }
)

COLORS = {
    "triage4": "#C73E1D",
    "strict": "#2E86AB",
    "fifo": "#6C757D",
}


def run_scenario_simple(scheduler, workload: Workload) -> Dict[str, float]:
    """Run scheduler and compute metrics (simplified for stress tests)."""
    result = scheduler.schedule(
        arrival_times=workload.arrival_times,
        device_ids=workload.device_ids,
        zone_priorities=workload.zone_priorities,
        is_alarm=workload.is_alarm,
    )

    metrics = compute_all_metrics(
        result=result,
        arrival_times=workload.arrival_times,
        device_ids=workload.device_ids,
        is_alarm=workload.is_alarm,
    )

    return metrics


# =============================================================================
# Stress Test 1: Device Count Scalability
# =============================================================================


def generate_scalable_workload(
    n_devices: int,
    duration: float = 60.0,
    service_rate: float = 20.0,
    seed: int | None = None,
) -> Workload:
    """
    Generate workload with variable device count.

    Keeps load constant (ρ=0.8) while scaling device count.
    Tests per-device fairness at scale.

    Args:
        n_devices: Number of devices to simulate
        duration: Simulation duration
        service_rate: Service capacity

    Returns:
        Workload with n_devices, constant load
    """
    if seed is not None:
        np.random.seed(seed)

    target_load = 0.8
    total_messages = int(service_rate * duration * target_load)
    messages_per_device = total_messages // n_devices

    arrival_times = []
    device_ids = []
    zone_priorities = []
    is_alarm = []

    interval = duration / messages_per_device

    for dev_idx in range(n_devices):
        device_id = f"device_{dev_idx}"
        zone = dev_idx % 6  # Distribute across 6 zones

        for msg_idx in range(messages_per_device):
            arrival_times.append(msg_idx * interval)
            device_ids.append(device_id)
            zone_priorities.append(zone)
            is_alarm.append(False)

    # Add fixed number of alarms (2% of total)
    n_alarms = max(2, int(0.02 * total_messages))
    alarm_interval = duration / n_alarms

    for alarm_idx in range(n_alarms):
        arrival_times.append(alarm_idx * alarm_interval)
        device_ids.append(f"alarm_sensor_{alarm_idx % 3}")
        zone_priorities.append(5)  # Low-priority zone
        is_alarm.append(True)

    # Sort
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=f"Scalability Test: {n_devices} devices, ρ=0.8",
    )


def stress_test_device_scalability(
    output_dir: str = "results/stress", quick: bool = False
) -> Dict:
    """
    Test 1: Device Count Scalability.

    Sweeps device count from 10 to 500, measures:
    - Alarm P95 latency
    - HIGH band fairness (Jain index)
    - Simulation runtime

    Args:
        output_dir: Output directory for plots
        quick: If True, use fewer data points

    Returns:
        Dictionary with results per device count
    """
    print("\n" + "=" * 100)
    print("STRESS TEST 1: DEVICE COUNT SCALABILITY")
    print("=" * 100)

    device_counts = [10, 20, 50, 100, 200] if quick else [10, 20, 50, 100, 200, 500]
    print(f"Device counts: {device_counts}")
    print()

    results = {
        "device_counts": device_counts,
        "triage4": {
            "alarm_p95": [],
            "band_0_fairness": [],
            "band_1_fairness": [],
            "band_2_fairness": [],
            "band_3_fairness": [],
            "runtime": [],
        },
        "strict": {
            "alarm_p95": [],
            "band_0_fairness": [],
            "band_1_fairness": [],
            "band_2_fairness": [],
            "band_3_fairness": [],
            "runtime": [],
        },
    }

    for n_devices in tqdm(device_counts, desc="Scalability Tests", ncols=80):
        seed = 999 + n_devices
        np.random.seed(seed)
        try:
            workload = generate_scalable_workload(n_devices, seed=seed)
        except TypeError:
            workload = generate_scalable_workload(n_devices)

        # TRIAGE/4
        import time

        triage4 = TRIAGE4Scheduler(TRIAGE4Config(), scheduler_seed=seed)
        t0 = time.time()
        triage4_metrics = run_scenario_simple(triage4, workload)
        triage4_runtime = time.time() - t0

        results["triage4"]["alarm_p95"].append(triage4_metrics["alarm_p95_latency"])
        results["triage4"]["band_0_fairness"].append(triage4_metrics["band_0_fairness"])
        results["triage4"]["band_1_fairness"].append(triage4_metrics["band_1_fairness"])
        results["triage4"]["band_2_fairness"].append(triage4_metrics["band_2_fairness"])
        results["triage4"]["band_3_fairness"].append(triage4_metrics["band_3_fairness"])
        results["triage4"]["runtime"].append(triage4_runtime)

        # Strict Priority
        strict = StrictPriorityScheduler(service_rate=20.0, scheduler_seed=seed)
        t0 = time.time()
        strict_metrics = run_scenario_simple(strict, workload)
        strict_runtime = time.time() - t0

        results["strict"]["alarm_p95"].append(strict_metrics["alarm_p95_latency"])
        results["strict"]["band_0_fairness"].append(strict_metrics["band_0_fairness"])
        results["strict"]["band_1_fairness"].append(strict_metrics["band_1_fairness"])
        results["strict"]["band_2_fairness"].append(strict_metrics["band_2_fairness"])
        results["strict"]["band_3_fairness"].append(strict_metrics["band_3_fairness"])
        results["strict"]["runtime"].append(strict_runtime)

    # Plot results
    _plot_scalability_results(results, output_dir)

    # Export results to JSON and CSV
    export_stress_results_json(results, "device_scalability", output_dir)
    export_stress_results_csv(results, "device_scalability", output_dir)

    return results


def _plot_scalability_results(results: Dict, output_dir: str):
    """Generate scalability plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    device_counts = results["device_counts"]

    # Plot 1: Alarm P95 Latency
    ax = axes[0]
    ax.plot(
        device_counts,
        results["triage4"]["alarm_p95"],
        marker="o",
        linewidth=2,
        label="TRIAGE/4",
        color=COLORS["triage4"],
    )
    ax.plot(
        device_counts,
        results["strict"]["alarm_p95"],
        marker="s",
        linewidth=2,
        label="Strict Priority",
        color=COLORS["strict"],
    )
    ax.set_xlabel("Number of Devices")
    ax.set_ylabel("Alarm P95 Latency (s)")
    ax.set_title("Alarm Latency vs. Device Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # Plot 2: HIGH Band Fairness
    ax = axes[1]
    ax.plot(
        device_counts,
        results["triage4"]["band_1_fairness"],
        marker="o",
        linewidth=2,
        label="TRIAGE/4",
        color=COLORS["triage4"],
    )
    ax.plot(
        device_counts,
        results["strict"]["band_1_fairness"],
        marker="s",
        linewidth=2,
        label="Strict Priority",
        color=COLORS["strict"],
    )
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="Target 0.8")
    ax.set_xlabel("Number of Devices")
    ax.set_ylabel("HIGH Band Fairness (Jain)")
    ax.set_title("Fairness vs. Device Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)

    # Plot 3: Runtime
    ax = axes[2]
    ax.plot(
        device_counts,
        results["triage4"]["runtime"],
        marker="o",
        linewidth=2,
        label="TRIAGE/4",
        color=COLORS["triage4"],
    )
    ax.plot(
        device_counts,
        results["strict"]["runtime"],
        marker="s",
        linewidth=2,
        label="Strict Priority",
        color=COLORS["strict"],
    )
    ax.set_xlabel("Number of Devices")
    ax.set_ylabel("Simulation Runtime (s)")
    ax.set_title("Computational Cost vs. Device Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "stress_device_scalability.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved: {output_path}")
    plt.close()


# =============================================================================
# Stress Test 2: Alarm Rate Surge
# =============================================================================


def generate_alarm_surge_workload(
    alarm_ratio: float,
    duration: float = 60.0,
    service_rate: float = 20.0,
    seed: int | None = None,
) -> Workload:
    """
    Generate workload with variable alarm ratio.

    Tests ALARM band fairness under increasing alarm density.

    Args:
        alarm_ratio: Fraction of messages that are alarms (0.01 to 0.5)
        duration: Simulation duration
        service_rate: Service capacity

    Returns:
        Workload with specified alarm ratio
    """
    if seed is not None:
        np.random.seed(seed)

    target_load = 0.8
    total_messages = int(service_rate * duration * target_load)
    n_alarms = int(total_messages * alarm_ratio)
    n_telemetry = total_messages - n_alarms

    arrival_times = []
    device_ids = []
    zone_priorities = []
    is_alarm = []

    # Telemetry messages
    telemetry_interval = duration / n_telemetry if n_telemetry > 0 else duration
    for msg_idx in range(n_telemetry):
        arrival_times.append(msg_idx * telemetry_interval)
        device_ids.append(f"telemetry_{msg_idx % 10}")
        zone_priorities.append(msg_idx % 6)
        is_alarm.append(False)

    # Alarm messages (distributed across multiple devices)
    n_alarm_devices = max(3, int(n_alarms / 10))
    alarms_per_device = n_alarms // n_alarm_devices
    alarm_interval = duration / alarms_per_device if alarms_per_device > 0 else duration

    for dev_idx in range(n_alarm_devices):
        device_id = f"alarm_sensor_{dev_idx}"
        for alarm_idx in range(alarms_per_device):
            arrival_times.append(alarm_idx * alarm_interval + dev_idx * 0.01)
            device_ids.append(device_id)
            zone_priorities.append(5)
            is_alarm.append(True)

    # Sort
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=f"Alarm Surge Test: {alarm_ratio*100:.0f}% alarms",
    )


def stress_test_alarm_rate(
    output_dir: str = "results/stress", quick: bool = False
) -> Dict:
    """
    Test 2: Alarm Rate Surge.

    Sweeps alarm ratio from 1% to 50%, measures:
    - Alarm avg latency
    - ALARM band fairness (per-device)
    - BACKGROUND band starvation

    Args:
        output_dir: Output directory for plots
        quick: If True, use fewer data points

    Returns:
        Dictionary with results per alarm ratio
    """
    print("\n" + "=" * 100)
    print("STRESS TEST 2: ALARM RATE SURGE")
    print("=" * 100)

    alarm_ratios = [0.01, 0.05, 0.10, 0.20, 0.50] if quick else [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    print(f"Alarm ratios: {[f'{r*100:.0f}%' for r in alarm_ratios]}")
    print()

    results = {
        "alarm_ratios": alarm_ratios,
        "triage4": {
            "alarm_latency": [],
            "band_0_fairness": [],
            "band_1_fairness": [],
            "band_2_fairness": [],
            "band_3_fairness": [],
        },
        "strict": {
            "alarm_latency": [],
            "band_0_fairness": [],
            "band_1_fairness": [],
            "band_2_fairness": [],
            "band_3_fairness": [],
        },
    }

    for ratio in tqdm(alarm_ratios, desc="Alarm Rate Tests", ncols=80):
        seed = int(999 + ratio * 1000)
        np.random.seed(seed)
        try:
            workload = generate_alarm_surge_workload(ratio, seed=seed)
        except TypeError:
            workload = generate_alarm_surge_workload(ratio)

        # TRIAGE/4
        triage4 = TRIAGE4Scheduler(TRIAGE4Config(), scheduler_seed=seed)
        triage4_metrics = run_scenario_simple(triage4, workload)

        results["triage4"]["alarm_latency"].append(triage4_metrics["alarm_avg_latency"])
        results["triage4"]["band_0_fairness"].append(triage4_metrics["band_0_fairness"])
        results["triage4"]["band_1_fairness"].append(triage4_metrics["band_1_fairness"])
        results["triage4"]["band_2_fairness"].append(triage4_metrics["band_2_fairness"])
        results["triage4"]["band_3_fairness"].append(triage4_metrics["band_3_fairness"])

        # Strict Priority
        strict = StrictPriorityScheduler(service_rate=20.0, scheduler_seed=seed)
        strict_metrics = run_scenario_simple(strict, workload)

        results["strict"]["alarm_latency"].append(strict_metrics["alarm_avg_latency"])
        results["strict"]["band_0_fairness"].append(strict_metrics["band_0_fairness"])
        results["strict"]["band_1_fairness"].append(strict_metrics["band_1_fairness"])
        results["strict"]["band_2_fairness"].append(strict_metrics["band_2_fairness"])
        results["strict"]["band_3_fairness"].append(strict_metrics["band_3_fairness"])

    # Plot results
    _plot_alarm_rate_results(results, output_dir)

    # Export results to JSON and CSV
    export_stress_results_json(results, "alarm_rate", output_dir)
    export_stress_results_csv(results, "alarm_rate", output_dir)

    return results


def _plot_alarm_rate_results(results: Dict, output_dir: str):
    """Generate alarm rate surge plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    alarm_ratios_pct = [r * 100 for r in results["alarm_ratios"]]

    # Plot 1: Alarm Avg Latency
    ax = axes[0]
    ax.plot(
        alarm_ratios_pct,
        results["triage4"]["alarm_latency"],
        marker="o",
        linewidth=2,
        label="TRIAGE/4",
        color=COLORS["triage4"],
    )
    ax.plot(
        alarm_ratios_pct,
        results["strict"]["alarm_latency"],
        marker="s",
        linewidth=2,
        label="Strict Priority",
        color=COLORS["strict"],
    )
    ax.set_xlabel("Alarm Ratio (%)")
    ax.set_ylabel("Alarm Avg Latency (s)")
    ax.set_title("Alarm Latency vs. Alarm Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: ALARM Band Fairness (per-device)
    ax = axes[1]
    ax.plot(
        alarm_ratios_pct,
        results["triage4"]["band_0_fairness"],
        marker="o",
        linewidth=2,
        label="TRIAGE/4",
        color=COLORS["triage4"],
    )
    ax.plot(
        alarm_ratios_pct,
        results["strict"]["band_0_fairness"],
        marker="s",
        linewidth=2,
        label="Strict Priority",
        color=COLORS["strict"],
    )
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="Target 0.8")
    ax.set_xlabel("Alarm Ratio (%)")
    ax.set_ylabel("ALARM Band Fairness (Jain)")
    ax.set_title("ALARM Band Fairness vs. Alarm Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Plot 3: BACKGROUND Band Fairness (starvation indicator)
    ax = axes[2]
    ax.plot(
        alarm_ratios_pct,
        results["triage4"]["band_3_fairness"],
        marker="o",
        linewidth=2,
        label="TRIAGE/4",
        color=COLORS["triage4"],
    )
    ax.plot(
        alarm_ratios_pct,
        results["strict"]["band_3_fairness"],
        marker="s",
        linewidth=2,
        label="Strict Priority",
        color=COLORS["strict"],
    )
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="Target 0.8")
    ax.set_xlabel("Alarm Ratio (%)")
    ax.set_ylabel("BACKGROUND Band Fairness (Jain)")
    ax.set_title("BACKGROUND Fairness vs. Alarm Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "stress_alarm_rate_surge.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


# =============================================================================
# Export Functions
# =============================================================================


def export_stress_results_json(results: Dict, test_name: str, output_dir: str) -> str:
    """
    Export stress test results to JSON format.

    Args:
        results: Dictionary with test results (device_counts/alarm_ratios + scheduler metrics)
        test_name: Name of the stress test (e.g., "device_scalability", "alarm_rate")
        output_dir: Output directory for results

    Returns:
        Path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"stress_{test_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved JSON: {path}")
    return path


def export_stress_results_csv(results: Dict, test_name: str, output_dir: str) -> str:
    """
    Export stress test results to CSV format.

    Args:
        results: Dictionary with test results
        test_name: Name of the stress test
        output_dir: Output directory for results

    Returns:
        Path to saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"stress_{test_name}.csv")

    # Determine the sweep parameter
    if "device_counts" in results:
        sweep_key = "device_counts"
        sweep_label = "device_count"
    elif "alarm_ratios" in results:
        sweep_key = "alarm_ratios"
        sweep_label = "alarm_ratio"
    else:
        print(f"Warning: Unknown sweep parameter in results for {test_name}")
        return ""

    sweep_values = results[sweep_key]

    # Build header and rows
    schedulers = [k for k in results.keys() if k != sweep_key]
    metrics = list(results[schedulers[0]].keys()) if schedulers else []

    # Header: sweep_label, then scheduler_metric columns
    header = [sweep_label]
    for sched in schedulers:
        for metric in metrics:
            header.append(f"{sched}_{metric}")

    rows = []
    for i, sweep_val in enumerate(sweep_values):
        row = [str(sweep_val)]
        for sched in schedulers:
            for metric in metrics:
                row.append(str(results[sched][metric][i]))
        rows.append(row)

    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")

    print(f"✓ Saved CSV: {path}")
    return path


def export_comprehensive_stress_csv(all_results: Dict[str, Dict], output_dir: str) -> str:
    """
    Export comprehensive stress test results combining all tests into one CSV.

    Args:
        all_results: Dictionary mapping test_name -> results dict
        output_dir: Output directory for results

    Returns:
        Path to saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "stress_comprehensive.csv")

    rows = []
    for test_name, results in all_results.items():
        # Determine sweep parameter
        if "device_counts" in results:
            sweep_key = "device_counts"
            sweep_label = "device_count"
        elif "alarm_ratios" in results:
            sweep_key = "alarm_ratios"
            sweep_label = "alarm_ratio"
        else:
            continue

        sweep_values = results[sweep_key]
        schedulers = [k for k in results.keys() if k != sweep_key]

        for i, sweep_val in enumerate(sweep_values):
            for sched in schedulers:
                row = {
                    "test_name": test_name,
                    "sweep_param": sweep_label,
                    "sweep_value": sweep_val,
                    "scheduler": sched,
                }
                for metric, values in results[sched].items():
                    row[metric] = values[i]
                rows.append(row)

    if not rows:
        print("Warning: No results to export")
        return ""

    # Get all unique metric columns
    all_metrics = set()
    for row in rows:
        all_metrics.update(k for k in row.keys() if k not in ["test_name", "sweep_param", "sweep_value", "scheduler"])

    fieldnames = ["test_name", "sweep_param", "sweep_value", "scheduler"] + sorted(all_metrics)

    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(col, "")) for col in fieldnames) + "\n")

    print(f"✓ Saved comprehensive CSV: {path}")
    return path


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="TRIAGE/4 Stress Testing Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stress tests
  python benchmarks/stress_benchmark.py --all

  # Run specific test
  python benchmarks/stress_benchmark.py --test device_scalability

  # Quick mode (fewer data points)
  python benchmarks/stress_benchmark.py --test device_scalability --quick
        """,
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["device_scalability", "alarm_rate"],
        help="Specific stress test to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all stress tests",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer data points",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/stress",
        help="Output directory for results (default: results/stress)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define tests
    tests = {
        "device_scalability": stress_test_device_scalability,
        "alarm_rate": stress_test_alarm_rate,
    }

    # Run tests
    all_results: Dict[str, Dict] = {}

    if args.all:
        print("\n" + "=" * 100)
        print("RUNNING ALL STRESS TESTS")
        print("=" * 100)

        for test_name, test_func in tests.items():
            results = test_func(output_dir=args.output_dir, quick=args.quick)
            all_results[test_name] = results
            print()

        # Export comprehensive CSV combining all tests
        export_comprehensive_stress_csv(all_results, args.output_dir)

    elif args.test:
        results = tests[args.test](output_dir=args.output_dir, quick=args.quick)
        all_results[args.test] = results

    else:
        parser.print_help()
        return

    print("\n" + "=" * 100)
    print("Stress testing complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()

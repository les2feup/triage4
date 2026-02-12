"""
TRIAGE/4 Comparison Benchmark.

Runs TRIAGE/4, Strict Priority, and FIFO schedulers on all three evaluation
scenarios and computes comprehensive metrics.

Usage:
    python benchmarks/comparison_benchmark.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assessment.baselines import FIFOScheduler, StrictPriorityScheduler
from assessment.metrics import (
    compute_all_metrics,
    compute_order_metrics,
    export_input_order,
    export_output_order,
)
from assessment.metrics.results import SchedulerResult
from assessment.metrics.visualization import (
    plot_band_priority_heatmap,
    plot_device_fairness_timeline,
    plot_position_jump_distribution,
    plot_rank_change_comparison,
)
from triage4 import TRIAGE4Config, TRIAGE4Scheduler
from assessment.workloads import (
    generate_alarm_under_burst,
    generate_device_monopolization,
    generate_multi_zone_emergency,
)


def run_scenario(scheduler, workload, scheduler_name: str) -> Tuple[Dict[str, float], SchedulerResult]:
    """
    Run a single scheduler on a workload and compute metrics.

    Args:
        scheduler: Scheduler instance (TRIAGE/4, StrictPriority, or FIFO)
        workload: Workload instance with arrival_times, device_ids, etc.
        scheduler_name: Name for metadata

    Returns:
        Tuple of (metrics dict, SchedulerResult)
    """
    # Run simulation
    result = scheduler.schedule(
        arrival_times=workload.arrival_times,
        device_ids=workload.device_ids,
        zone_priorities=workload.zone_priorities,
        is_alarm=workload.is_alarm,
    )

    # Compute time-based metrics
    time_metrics = compute_all_metrics(
        result=result,
        arrival_times=workload.arrival_times,
        device_ids=workload.device_ids,
        is_alarm=workload.is_alarm,
    )

    # Compute order-based metrics
    order_metrics = compute_order_metrics(
        result=result,
        arrival_times=workload.arrival_times,
        device_ids=workload.device_ids,
        is_alarm=workload.is_alarm,
    )

    # Combine metrics (only float values)
    metrics = {**time_metrics, **order_metrics}
    # If scheduler_name is needed, return it separately or handle outside

    return metrics, result


def format_metric_row(
    metric_name: str, seps_val: float, strict_val: float, fifo_val: float
) -> str:
    """Format a single metric row for comparison table."""
    # Compute improvements
    if strict_val > 0:
        vs_strict = ((strict_val - seps_val) / strict_val) * 100
        vs_strict_str = f"{vs_strict:+.1f}%"
    else:
        vs_strict_str = "N/A"

    if fifo_val > 0:
        vs_fifo = ((fifo_val - seps_val) / fifo_val) * 100
        vs_fifo_str = f"{vs_fifo:+.1f}%"
    else:
        vs_fifo_str = "N/A"

    return (
        f"  {metric_name:25s} | "
        f"{seps_val:8.4f} | "
        f"{strict_val:8.4f} ({vs_strict_str:>7s}) | "
        f"{fifo_val:8.4f} ({vs_fifo_str:>7s})"
    )


def print_comparison_table(
    scenario_name: str,
    seps_metrics: Dict[str, float],
    strict_metrics: Dict[str, float],
    fifo_metrics: Dict[str, float],
) -> None:
    """
    Print formatted comparison table for a scenario.

    Args:
        scenario_name: Name of the scenario
        seps_metrics: Metrics from TRIAGE/4
        strict_metrics: Metrics from Strict Priority
        fifo_metrics: Metrics from FIFO
    """
    print(f"\n{'=' * 90}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'=' * 90}")
    print(
        f"  {'Metric':25s} | {'TRIAGE/4':>8s} | "
        f"{'Strict':>8s} {'(vs TRIAGE/4)':>8s} | "
        f"{'FIFO':>8s} {'(vs TRIAGE/4)':>8s}"
    )
    print(f"  {'-' * 25} | {'-' * 8} | {'-' * 18} | {'-' * 18}")

    # Key metrics with REFACTORING_PLAN.md success criteria + order metrics
    key_metrics = [
        ("alarm_avg_latency", "Alarm Avg Latency (s)"),
        ("alarm_p95_latency", "Alarm P95 Latency (s)"),
        ("alarm_avg_jump", "Alarm Avg Jump (pos)"),
        ("min_device_rate", "Min Device Rate (msg/s)"),
        ("band_1_fairness", "HIGH Band Fairness"),
        ("band_2_fairness", "STANDARD Band Fairness"),
        ("band_3_fairness", "BACKGROUND Band Fairness"),
        ("max_consecutive_serves", "Max Consecutive Serves"),
        ("band_inversion_count", "Band Inversions"),
        ("high_avg_latency", "HIGH Avg Latency (s)"),
        ("avg_waiting_time", "Overall Avg Wait (s)"),
        ("p95_waiting_time", "Overall P95 Wait (s)"),
    ]

    for metric_key, metric_label in key_metrics:
        if metric_key in seps_metrics:
            print(
                format_metric_row(
                    metric_label,
                    seps_metrics[metric_key],
                    strict_metrics.get(metric_key, 0.0),
                    fifo_metrics.get(metric_key, 0.0),
                )
            )

    print(f"  {'-' * 25} | {'-' * 8} | {'-' * 18} | {'-' * 18}")
    print(
        f"  {'Total Messages':25s} | {int(seps_metrics['total_messages']):8d} | "
        f"{int(strict_metrics['total_messages']):8d} {'':>8s} | "
        f"{int(fifo_metrics['total_messages']):8d} {'':>8s}"
    )


def check_success_criteria(
    scenario_name: str,
    seps_metrics: Dict[str, float],
    strict_metrics: Dict[str, float],
) -> Dict[str, bool]:
    """
    Check REFACTORING_PLAN.md success criteria.

    Criteria:
        ✅ Alarm latency reduction: TRIAGE/4 < Strict by 40-60%
        ✅ Minimum bandwidth guarantee: All devices get ≥ 0.1 msg/sec
        ✅ Fairness improvement: Jain Index > 0.8 for H/S/B bands
        ✅ Acceptable overhead: High-priority overhead < 20%

    Args:
        scenario_name: Name of scenario
        seps_metrics: TRIAGE/4 metrics
        strict_metrics: Strict Priority metrics

    Returns:
        Dictionary of pass/fail for each criterion
    """
    results = {}

    # Criterion 1: Alarm latency reduction (40-60%)
    if seps_metrics.get("alarm_count", 0) > 0:
        alarm_reduction = (
            (strict_metrics["alarm_avg_latency"] - seps_metrics["alarm_avg_latency"])
            / strict_metrics["alarm_avg_latency"]
        ) * 100
        results["alarm_latency_reduction"] = alarm_reduction >= 40.0
        alarm_reduction_val = alarm_reduction
    else:
        results["alarm_latency_reduction"] = None  # N/A
        alarm_reduction_val = 0.0

    # Criterion 2: Minimum bandwidth guarantee (≥ 0.1 msg/sec)
    min_rate = seps_metrics.get("min_device_rate", 0.0)
    results["min_bandwidth_guarantee"] = min_rate >= 0.1

    # Criterion 3: Fairness (Jain Index > 0.8 for all bands)
    fairness_pass = all(
        seps_metrics.get(f"band_{band}_fairness", 0.0) > 0.8 for band in [1, 2, 3]
    )
    results["fairness_requirement"] = fairness_pass

    # Criterion 4: High-priority overhead (< 20% vs baseline)
    # Interpret as: TRIAGE/4 high latency should be < 1.2x strict high latency
    if strict_metrics.get("high_avg_latency", 0) > 0:
        overhead_ratio = (
            seps_metrics.get("high_avg_latency", 0)
            / strict_metrics["high_avg_latency"]
        )
        results["high_priority_overhead"] = overhead_ratio < 1.2
    else:
        results["high_priority_overhead"] = True  # Pass if no high traffic

    # Print summary
    print(f"\n📊 SUCCESS CRITERIA CHECK: {scenario_name}")
    print(f"  {'Criterion':40s} | {'Status':10s} | {'Value':>15s}")
    print(f"  {'-' * 40} | {'-' * 10} | {'-' * 15}")

    status_icon = lambda x: "✅ PASS" if x else "❌ FAIL" if x is not None else "⚪ N/A"

    if results["alarm_latency_reduction"] is not None:
        print(
            f"  {'Alarm latency reduction (≥40%)':40s} | "
            f"{status_icon(results['alarm_latency_reduction']):10s} | "
            f"{alarm_reduction_val:>14.1f}%"
        )
    else:
        print(
            f"  {'Alarm latency reduction (≥40%)':40s} | "
            f"{status_icon(results['alarm_latency_reduction']):10s} | "
            f"{'No alarms':>15s}"
        )

    print(
        f"  {'Min bandwidth guarantee (≥0.1 msg/s)':40s} | "
        f"{status_icon(results['min_bandwidth_guarantee']):10s} | "
        f"{min_rate:>14.3f} msg/s"
    )

    fairness_vals = [
        seps_metrics.get(f"band_{b}_fairness", 0.0) for b in [1, 2, 3]
    ]
    fairness_str = f"{min(fairness_vals):.3f} (min)"
    print(
        f"  {'Fairness Jain Index (>0.8 all bands)':40s} | "
        f"{status_icon(results['fairness_requirement']):10s} | "
        f"{fairness_str:>15s}"
    )

    if strict_metrics.get("high_avg_latency", 0) > 0:
        overhead_ratio = (
            seps_metrics.get("high_avg_latency", 0)
            / strict_metrics["high_avg_latency"]
        )
        overhead_str = f"{overhead_ratio:.2f}x (<1.2x)"
    else:
        overhead_str = "N/A"
    print(
        f"  {'High-priority overhead (<1.2x strict)':40s} | "
        f"{status_icon(results['high_priority_overhead']):10s} | "
        f"{overhead_str:>15s}"
    )

    return results


def main():
    """Run full benchmark comparison."""
    print("\n" + "=" * 90)
    print("TRIAGE/4 EVALUATION BENCHMARK")
    print("Comparing: TRIAGE/4 vs. Strict Priority vs. FIFO")
    print("=" * 90)

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Define scenarios
    scenarios = [
        ("Alarm Under Burst Load", generate_alarm_under_burst()),
        ("Device Monopolization", generate_device_monopolization()),
        ("Multi-Zone Emergency", generate_multi_zone_emergency()),
    ]

    all_criteria_results = []

    # Run each scenario
    for scenario_name, workload in scenarios:
        # Initialize fresh schedulers for each scenario (to avoid state carryover)
        seed = 42
        seps = TRIAGE4Scheduler(TRIAGE4Config(service_rate=20.0), scheduler_seed=seed)
        strict = StrictPriorityScheduler(service_rate=20.0, scheduler_seed=seed)
        fifo = FIFOScheduler(service_rate=20.0, scheduler_seed=seed)
        # Export input order CSV (once per scenario)
        export_input_order(
            arrival_times=workload.arrival_times,
            device_ids=workload.device_ids,
            zone_priorities=workload.zone_priorities,
            is_alarm=workload.is_alarm,
            scenario_name=scenario_name,
            output_dir="results"
        )

        # Run all schedulers and collect results
        results_dict = {}  # For visualization

        # TRIAGE/4
        seps_metrics, seps_result = run_scenario(seps, workload, "TRIAGE/4")
        export_output_order(
            result=seps_result,
            arrival_times=workload.arrival_times,
            device_ids=workload.device_ids,
            zone_priorities=workload.zone_priorities,
            is_alarm=workload.is_alarm,
            scheduler_name="TRIAGE/4",
            scenario_name=scenario_name,
            output_dir="results"
        )
        results_dict["TRIAGE/4"] = (seps_result, workload.arrival_times, workload.device_ids)

        # Strict Priority
        strict_metrics, strict_result = run_scenario(strict, workload, "StrictPriority")
        export_output_order(
            result=strict_result,
            arrival_times=workload.arrival_times,
            device_ids=workload.device_ids,
            zone_priorities=workload.zone_priorities,
            is_alarm=workload.is_alarm,
            scheduler_name="Strict",
            scenario_name=scenario_name,
            output_dir="results"
        )
        results_dict["Strict"] = (strict_result, workload.arrival_times, workload.device_ids)

        # FIFO
        fifo_metrics, fifo_result = run_scenario(fifo, workload, "FIFO")
        export_output_order(
            result=fifo_result,
            arrival_times=workload.arrival_times,
            device_ids=workload.device_ids,
            zone_priorities=workload.zone_priorities,
            is_alarm=workload.is_alarm,
            scheduler_name="FIFO",
            scenario_name=scenario_name,
            output_dir="results"
        )
        results_dict["FIFO"] = (fifo_result, workload.arrival_times, workload.device_ids)

        # Print comparison table
        print_comparison_table(scenario_name, seps_metrics, strict_metrics, fifo_metrics)

        # Generate visualizations
        print(f"\n📊 Generating plots for {scenario_name}...")

        # Rank change scatter plot
        plot_rank_change_comparison(results_dict, scenario_name, "results")

        # Position jump distribution
        results_with_alarm = {
            name: (result, arrival_times, workload.is_alarm)
            for name, (result, arrival_times, _) in results_dict.items()
        }
        plot_position_jump_distribution(results_with_alarm, scenario_name, "results")

        # Device fairness timeline
        plot_device_fairness_timeline(results_dict, scenario_name, "results")

        # Band priority heatmap
        results_for_bands = {
            name: (result, arrival_times)
            for name, (result, arrival_times, _) in results_dict.items()
        }
        plot_band_priority_heatmap(results_for_bands, scenario_name, "results")

        print(f"   ✓ Plots saved to results/{scenario_name}/")

        # Check success criteria
        criteria_results = check_success_criteria(
            scenario_name, seps_metrics, strict_metrics
        )
        all_criteria_results.append((scenario_name, criteria_results))

    # Print overall summary
    print(f"\n\n{'=' * 90}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 90}")

    for scenario_name, criteria in all_criteria_results:
        passed = sum(
            1 for v in criteria.values() if v is True
        )  # Count True (not None)
        total = sum(1 for v in criteria.values() if v is not None)  # Count non-N/A
        print(f"  {scenario_name:40s}: {passed}/{total} criteria passed")

    print("\n✅ Benchmark complete!")
    print(f"📁 Results saved to: results/")
    print(f"   - CSV files: input.csv, seps_output.csv, strict_output.csv, fifo_output.csv")
    print(f"   - Plots: rank_change_comparison.png, position_jump_distribution.png,")
    print(f"            device_fairness_timeline.png, band_priority_heatmap.png\n")


if __name__ == "__main__":
    main()

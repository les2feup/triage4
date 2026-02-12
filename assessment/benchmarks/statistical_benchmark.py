"""
TRIAGE/4 Statistical Benchmark with Multi-Run Analysis.

Runs TRIAGE/4, Strict Priority, and FIFO schedulers multiple times with different
seeds to compute statistically rigorous comparisons with confidence intervals
and significance testing.

Features:
- Multi-run framework (default 50 runs)
- Statistical aggregation (mean ± std, 95% CI, CV%)
- Significance testing (Welch's t-test)
- Error bar visualizations
- Publication-ready output

Usage:
    # Quick test (5 runs)
    python benchmarks/statistical_benchmark.py --n-runs 5

    # Recommended (50 runs for statistical rigor)
    python benchmarks/statistical_benchmark.py --n-runs 50

    # Single scenario
    python benchmarks/statistical_benchmark.py --scenario alarm_under_burst --n-runs 50

    # All scenarios
    python benchmarks/statistical_benchmark.py --all --n-runs 50
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assessment.baselines import FIFOScheduler, StrictPriorityScheduler
from assessment.metrics import (
    ComparisonResult,
    DistributionData,
    StatisticsSummary,
    compare_schedulers,
    compute_all_metrics,
    compute_distribution_data,
    compute_statistics,
    format_comparison_table,
    jain_fairness_index,
)
from assessment.metrics.results import SchedulerResult
from triage4 import TRIAGE4Config, TRIAGE4Scheduler
from assessment.workloads import (
    Workload,
    generate_alarm_load_regime,
    generate_alarm_load_near_saturation_constrained,
    generate_alarm_rate_sweep,
    generate_alarm_flood_attack,
    generate_alarm_malfunction_surge,
    generate_alarm_under_burst,
    generate_alarm_under_burst_phased,
    generate_device_monopolization,
    generate_device_monopolization_sweep,
    generate_legit_extreme_emergency,
    generate_multi_zone_emergency,
    generate_multi_zone_emergency_cascade,
    generate_skewed_alarm_sources,
)


def run_scenario(
    scheduler, workload: Workload, scheduler_name: str
) -> Tuple[Dict[str, float], SchedulerResult, DistributionData]:
    """
    Run a single scheduler on a workload and compute metrics.

    Args:
        scheduler: Scheduler instance (TRIAGE/4, StrictPriority, or FIFO)
        workload: Workload instance with arrival_times, device_ids, etc.
        scheduler_name: Name for metadata

    Returns:
        Tuple of (metrics dict, SchedulerResult, DistributionData)
    """
    # Run simulation
    result = scheduler.schedule(
        arrival_times=workload.arrival_times,
        device_ids=workload.device_ids,
        zone_priorities=workload.zone_priorities,
        is_alarm=workload.is_alarm,
    )

    # Compute time-based metrics
    metrics = compute_all_metrics(
        result=result,
        arrival_times=workload.arrival_times,
        device_ids=workload.device_ids,
        is_alarm=workload.is_alarm,
        zone_priorities=workload.zone_priorities,
    )

    # Compute distribution data for plotting
    dist_data = compute_distribution_data(
        result=result,
        device_ids=workload.device_ids,
        is_alarm=workload.is_alarm,
        zone_priorities=workload.zone_priorities,
    )

    return metrics, result, dist_data


def create_triage4_default() -> TRIAGE4Config:
    """Factory for the default TRIAGE/4 configuration."""
    return TRIAGE4Config()


def compute_phase_metrics(
    workload: Workload,
    result: SchedulerResult,
    phase_boundaries,
) -> Dict[str, List[float]]:
    """Compute per-phase metrics for phased workloads."""
    metrics = {
        "phase_high_wait": [],
        "phase_overall_wait": [],
        "phase_high_fairness": [],
    }

    for start, end in phase_boundaries:
        phase_indices = [
            i
            for i, t in enumerate(workload.arrival_times)
            if start - 1e-12 <= t <= end + 1e-12
        ]

        # Overall wait
        if phase_indices:
            waits = [result.waiting_times[i] for i in phase_indices]
            metrics["phase_overall_wait"].append(float(np.mean(waits)))
        else:
            metrics["phase_overall_wait"].append(0.0)

        # HIGH band wait + fairness
        high_indices = [
            i for i in phase_indices if result.priorities[i] == 1  # BAND_HIGH
        ]
        if high_indices:
            high_waits = [result.waiting_times[i] for i in high_indices]
            metrics["phase_high_wait"].append(float(np.mean(high_waits)))

            # Fairness across devices in HIGH band within phase
            device_waits: Dict[str, List[float]] = {}
            for idx in high_indices:
                dev = workload.device_ids[idx]
                device_waits.setdefault(dev, []).append(result.waiting_times[idx])
            avg_waits_per_device = [np.mean(v) for v in device_waits.values()]
            metrics["phase_high_fairness"].append(
                float(jain_fairness_index(avg_waits_per_device))
            )
        else:
            metrics["phase_high_wait"].append(0.0)
            metrics["phase_high_fairness"].append(1.0)

    return metrics


def run_statistical_analysis(
    workload_generator,
    scenario_name: str,
    n_runs: int = 50,
    base_seed: int = 999,
    output_dir: str = "results/statistical",
    enable_alarm_protection: bool = False,
    service_rate_override: float | None = None,
) -> Tuple[
    Dict[str, Dict[str, StatisticsSummary]],
    List[ComparisonResult],
    Dict[str, Dict[str, List[StatisticsSummary]]] | None,
    List[tuple[float, float]] | None,
    Dict[str, List[DistributionData]],
]:
    """
    Run statistical benchmark with multiple iterations.

    Args:
        workload_generator: Function that generates workload (takes seed parameter)
        scenario_name: Name of scenario (e.g., "Alarm Under Burst")
        n_runs: Number of independent runs
        base_seed: Starting seed value (seeds will be base_seed, base_seed+1, ...)
        output_dir: Output directory for results
        enable_alarm_protection: If True, enable adaptive alarm protection in TRIAGE/4

    Returns:
        Tuple of (aggregated_metrics, comparisons, aggregated_phase, phase_boundaries, all_distributions)
            - aggregated_metrics: Dict[scheduler_name, Dict[metric_name, StatisticsSummary]]
            - comparisons: List[ComparisonResult] with significance tests
            - aggregated_phase: Per-phase statistics (if phased workload)
            - phase_boundaries: Phase time boundaries (if phased workload)
            - all_distributions: Raw per-device/per-source distribution data for plotting
    """
    print("=" * 100)
    print(f"STATISTICAL BENCHMARK: {scenario_name}")
    print("=" * 100)
    print(f"Number of runs: {n_runs}")
    print(f"Seeds: {base_seed} to {base_seed + n_runs - 1}")
    print()

    # Storage for all runs
    all_metrics = {
        "TRIAGE/4": [],
        "Strict": [],
        "FIFO": [],
    }
    # Storage for distribution data (per-device/per-source vectors for plotting)
    all_distributions: Dict[str, List[DistributionData]] = {
        "TRIAGE/4": [],
        "Strict": [],
        "FIFO": [],
    }
    phase_boundaries = None
    phase_metrics = None  # type: Dict[str, Dict[str, List[List[float]]]] | None

    # Run all iterations with progress bar
    for run_idx in tqdm(
        range(n_runs),
        desc="Simulations",
        unit="run",
        ncols=80,
        ascii="░█",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        seed = base_seed + run_idx

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Generate workload (pass seed if supported)
        try:
            workload = workload_generator(seed=seed)
        except TypeError:
            workload = workload_generator()

        # Initialize per-phase collectors if workload is phased
        if getattr(workload, "phase_boundaries", None):
            if phase_boundaries is None:
                phase_boundaries = workload.phase_boundaries
                phase_metrics = {
                    sched: {
                        "phase_high_wait": [[] for _ in phase_boundaries],
                        "phase_overall_wait": [[] for _ in phase_boundaries],
                        "phase_high_fairness": [[] for _ in phase_boundaries],
                    }
                    for sched in ["TRIAGE/4", "Strict", "FIFO"]
                }

        # Run TRIAGE/4
        triage4_config = create_triage4_default()
        if service_rate_override is not None:
            triage4_config.service_rate = service_rate_override

        # For the constrained near-saturation load scenario, adjust token budgets
        # so that total non-alarm capacity is ≈100% of service rate instead of 200%.
        if scenario_name.startswith("Alarm Load Regime (ρ≈0.95 Token-Constrained"):
            triage4_config.high_token_budget = 10
            triage4_config.standard_token_budget = 7
            triage4_config.background_token_budget = 3

        if enable_alarm_protection:
            triage4_config.enable_alarm_protection = True
        seps = TRIAGE4Scheduler(triage4_config, scheduler_seed=seed)
        seps_metrics, seps_result, seps_dist = run_scenario(seps, workload, "TRIAGE/4")
        all_metrics["TRIAGE/4"].append(seps_metrics)
        all_distributions["TRIAGE/4"].append(seps_dist)
        if phase_metrics is not None:
            phase_vals = compute_phase_metrics(workload, seps_result, phase_boundaries)
            for phase_idx in range(len(phase_boundaries)):
                phase_metrics["TRIAGE/4"]["phase_high_wait"][phase_idx].append(
                    phase_vals["phase_high_wait"][phase_idx]
                )
                phase_metrics["TRIAGE/4"]["phase_overall_wait"][phase_idx].append(
                    phase_vals["phase_overall_wait"][phase_idx]
                )
                phase_metrics["TRIAGE/4"]["phase_high_fairness"][phase_idx].append(
                    phase_vals["phase_high_fairness"][phase_idx]
                )

        # Run Strict Priority
        strict = StrictPriorityScheduler(
            service_rate=triage4_config.service_rate, scheduler_seed=seed
        )
        strict_metrics, strict_result, strict_dist = run_scenario(strict, workload, "Strict")
        all_metrics["Strict"].append(strict_metrics)
        all_distributions["Strict"].append(strict_dist)
        if phase_metrics is not None:
            phase_vals = compute_phase_metrics(
                workload, strict_result, phase_boundaries
            )
            for phase_idx in range(len(phase_boundaries)):
                phase_metrics["Strict"]["phase_high_wait"][phase_idx].append(
                    phase_vals["phase_high_wait"][phase_idx]
                )
                phase_metrics["Strict"]["phase_overall_wait"][phase_idx].append(
                    phase_vals["phase_overall_wait"][phase_idx]
                )
                phase_metrics["Strict"]["phase_high_fairness"][phase_idx].append(
                    phase_vals["phase_high_fairness"][phase_idx]
                )

        # Run FIFO
        fifo = FIFOScheduler(service_rate=triage4_config.service_rate, scheduler_seed=seed)
        fifo_metrics, fifo_result, fifo_dist = run_scenario(fifo, workload, "FIFO")
        all_metrics["FIFO"].append(fifo_metrics)
        all_distributions["FIFO"].append(fifo_dist)
        if phase_metrics is not None:
            phase_vals = compute_phase_metrics(workload, fifo_result, phase_boundaries)
            for phase_idx in range(len(phase_boundaries)):
                phase_metrics["FIFO"]["phase_high_wait"][phase_idx].append(
                    phase_vals["phase_high_wait"][phase_idx]
                )
                phase_metrics["FIFO"]["phase_overall_wait"][phase_idx].append(
                    phase_vals["phase_overall_wait"][phase_idx]
                )
                phase_metrics["FIFO"]["phase_high_fairness"][phase_idx].append(
                    phase_vals["phase_high_fairness"][phase_idx]
                )

    # Aggregate statistics
    print("\n" + "=" * 100)
    print("AGGREGATING RESULTS ACROSS ALL RUNS")
    print("=" * 100)

    aggregated = {}
    for scheduler_name in ["TRIAGE/4", "Strict", "FIFO"]:
        aggregated[scheduler_name] = {}
        metrics_list = all_metrics[scheduler_name]

        # Key metrics to aggregate
        metric_keys = [
            "alarm_avg_latency",
            "alarm_p95_latency",
            "band_0_fairness",
            "band_1_fairness",
            "band_2_fairness",
            "band_3_fairness",
            "band_0_wait_mean",
            "band_1_wait_mean",
            "band_2_wait_mean",
            "band_3_wait_mean",
            "band_0_wait_p95",
            "band_1_wait_p95",
            "band_2_wait_p95",
            "band_3_wait_p95",
            "device_latency_fairness",
            "device_throughput_fairness",
            "min_device_rate",
            "avg_device_rate",
            "high_avg_latency",
            "high_p95_latency",
            "avg_waiting_time",
            "p95_waiting_time",
            # Adaptive protection metrics
            "alarm_dropped",
            "alarm_dropped_rate",
            "protection_enabled",
            "alarm_source_fairness",
            "alarm_source_count",
            "alarm_source_latency_cv",
        ]

        for metric_key in metric_keys:
            values = [m[metric_key] for m in metrics_list]
            aggregated[scheduler_name][metric_key] = compute_statistics(values)

    # Aggregate per-phase statistics (if available)
    aggregated_phase: Dict[str, Dict[str, List[StatisticsSummary]]] | None = None
    if phase_metrics is not None:
        aggregated_phase = {}
        for scheduler_name, metric_map in phase_metrics.items():
            aggregated_phase[scheduler_name] = {}
            for metric_key, per_phase_values in metric_map.items():
                aggregated_phase[scheduler_name][metric_key] = [
                    compute_statistics(values) for values in per_phase_values
                ]

    # Statistical comparisons
    comparisons = []

    # Key metrics for comparison
    comparison_metrics = [
        ("alarm_avg_latency", "Alarm Avg Latency"),
        ("alarm_p95_latency", "Alarm P95 Latency"),
        ("band_1_fairness", "HIGH Band Fairness"),
        ("band_2_fairness", "STANDARD Band Fairness"),
        ("band_3_fairness", "BACKGROUND Band Fairness"),
        ("device_latency_fairness", "Device Latency Fairness"),
        ("device_throughput_fairness", "Device Throughput Fairness"),
        ("min_device_rate", "Min Device Rate"),
        ("high_avg_latency", "HIGH Avg Latency"),
    ]

    for metric_key, metric_label in comparison_metrics:
        seps_values = [m[metric_key] for m in all_metrics["TRIAGE/4"]]
        strict_values = [m[metric_key] for m in all_metrics["Strict"]]
        fifo_values = [m[metric_key] for m in all_metrics["FIFO"]]

        # TRIAGE/4 vs Strict
        comp_strict = compare_schedulers(
            scheduler_a_name="TRIAGE/4",
            scheduler_b_name="Strict",
            metric_name=metric_label,
            values_a=seps_values,
            values_b=strict_values,
        )
        comparisons.append(comp_strict)

        # TRIAGE/4 vs FIFO
        comp_fifo = compare_schedulers(
            scheduler_a_name="TRIAGE/4",
            scheduler_b_name="FIFO",
            metric_name=metric_label,
            values_a=seps_values,
            values_b=fifo_values,
        )
        comparisons.append(comp_fifo)

    return aggregated, comparisons, aggregated_phase, phase_boundaries, all_distributions


# =============================================================================
# Aggregation Export Helpers
# =============================================================================


def export_aggregated_metrics(
    aggregated: Dict[str, Dict[str, StatisticsSummary]],
    scenario_key: str,
    output_dir: str,
    aggregated_phase: Dict[str, Dict[str, List[StatisticsSummary]]] | None = None,
    phase_boundaries=None,
) -> str:
    """Persist aggregated metrics to JSON for analysis."""
    os.makedirs(output_dir, exist_ok=True)
    payload: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scheduler, metrics in aggregated.items():
        payload[scheduler] = {}
        for metric_name, stat in metrics.items():
            payload[scheduler][metric_name] = {
                "mean": stat.mean,
                "std": stat.std,
                "ci_lower": stat.ci_lower,
                "ci_upper": stat.ci_upper,
                "n": stat.n_samples,
            }

    if aggregated_phase is not None:
        payload["phase_metrics"] = {}
        for scheduler, metric_map in aggregated_phase.items():
            payload["phase_metrics"][scheduler] = {}
            for metric_name, stats_list in metric_map.items():
                payload["phase_metrics"][scheduler][metric_name] = [
                    {
                        "mean": stat.mean,
                        "std": stat.std,
                        "ci_lower": stat.ci_lower,
                        "ci_upper": stat.ci_upper,
                        "n": stat.n_samples,
                    }
                    for stat in stats_list
                ]
        if phase_boundaries is not None:
            payload["phase_boundaries"] = phase_boundaries

    path = os.path.join(output_dir, f"{scenario_key}_aggregated.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Saved aggregated metrics: {path}")
    return path


def export_fairness_table(
    aggregated_records: List[Tuple[str, str, str]], output_path: str
) -> None:
    """
    Export a CSV summarizing throughput/fairness metrics per scenario and scheduler.
    Metrics: avg_device_rate, avg_waiting_time, device_throughput_fairness, device_latency_fairness.
    """
    if not aggregated_records:
        print("Skipping fairness table (no aggregated records).")
        return

    schedulers = ["TRIAGE/4", "Strict", "FIFO"]
    metrics = [
        "avg_device_rate",
        "avg_waiting_time",
        "device_throughput_fairness",
        "device_latency_fairness",
    ]

    rows = []
    for scenario_key, scenario_name, path in aggregated_records:
        with open(path, "r", encoding="utf-8") as f:
            agg = json.load(f)
        row = {"scenario_key": scenario_key, "scenario_name": scenario_name}
        for metric in metrics:
            for sched in schedulers:
                row[f"{metric}_{sched.lower()}"] = agg[sched][metric]["mean"]
        rows.append(row)

    fieldnames = ["scenario_key", "scenario_name"] + [
        f"{metric}_{sched.lower()}" for metric in metrics for sched in schedulers
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(col, "")) for col in fieldnames) + "\n")
    print(f"✓ Saved fairness/throughput table: {output_path}")


def export_comprehensive_results(
    aggregated_records: List[Tuple[str, str, str]], output_path: str
) -> None:
    """
    Export comprehensive benchmark results to CSV with all core metrics.
    Includes alarm latency, fairness, throughput, and protection metrics.
    """
    if not aggregated_records:
        print("Skipping comprehensive export (no aggregated records).")
        return

    schedulers = ["TRIAGE/4", "Strict", "FIFO"]
    core_metrics = [
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

    # Protection-specific metrics (TRIAGE/4 only)
    protection_metrics = [
        "alarm_dropped",
        "alarm_dropped_rate",
        "protection_enabled",
        "alarm_source_fairness",
        "alarm_source_count",
        "alarm_source_latency_cv",
    ]

    rows = []
    for scenario_key, scenario_name, path in aggregated_records:
        with open(path, "r", encoding="utf-8") as f:
            agg = json.load(f)

        for sched in schedulers:
            row = {
                "scenario_key": scenario_key,
                "scenario_name": scenario_name,
                "scheduler": sched,
            }

            # Core metrics (all schedulers)
            for metric in core_metrics:
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

            # Protection metrics (TRIAGE/4 only)
            if sched == "TRIAGE/4":
                for metric in protection_metrics:
                    if metric in agg[sched]:
                        row[f"{metric}_mean"] = agg[sched][metric]["mean"]
                        row[f"{metric}_std"] = agg[sched][metric]["std"]
                    else:
                        row[f"{metric}_mean"] = ""
                        row[f"{metric}_std"] = ""

            rows.append(row)

    # Build fieldnames
    fieldnames = ["scenario_key", "scenario_name", "scheduler"]
    for metric in core_metrics:
        fieldnames.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_ci_lower",
                f"{metric}_ci_upper",
            ]
        )
    for metric in protection_metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(col, "")) for col in fieldnames) + "\n")
    print(f"✓ Saved comprehensive results: {output_path}")


def export_distribution_data(
    all_distributions: Dict[str, List[DistributionData]],
    scenario_key: str,
    output_dir: str,
) -> str:
    """
    Export per-device/per-source distribution data for plotting (box/violin/CDF).

    Saves raw distribution vectors across all runs, enabling:
    - Box plots of per-device latency distributions
    - Violin plots comparing schedulers
    - CDF plots of device latencies
    - Per-source (zone) alarm latency analysis

    Args:
        all_distributions: Dict mapping scheduler -> list of DistributionData per run
        scenario_key: Scenario identifier for filename
        output_dir: Output directory

    Returns:
        Path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)

    payload: Dict = {}

    for scheduler, dist_list in all_distributions.items():
        # Aggregate distribution data across runs
        # For plotting, we want to see the distribution of per-device averages
        # across all runs (each run contributes one value per device)

        # Collect all device latencies across runs
        all_device_latencies: Dict[str, List[float]] = {}
        all_device_counts: Dict[str, List[int]] = {}
        all_source_latencies: Dict[str, List[float]] = {}
        all_band_device_latencies: Dict[str, Dict[str, List[float]]] = {
            "0": {},
            "1": {},
            "2": {},
            "3": {},
        }

        for dist in dist_list:
            # Per-device latencies
            for dev, lat in dist.device_avg_latencies.items():
                if dev not in all_device_latencies:
                    all_device_latencies[dev] = []
                all_device_latencies[dev].append(lat)

            # Per-device message counts
            for dev, count in dist.device_msg_counts.items():
                if dev not in all_device_counts:
                    all_device_counts[dev] = []
                all_device_counts[dev].append(count)

            # Per-source (zone) alarm latencies
            for zone, lat in dist.source_avg_latencies.items():
                zone_str = str(zone)
                if zone_str not in all_source_latencies:
                    all_source_latencies[zone_str] = []
                all_source_latencies[zone_str].append(lat)

            # Per-band per-device latencies
            for band_id, dev_lats in dist.per_band_device_latencies.items():
                band_str = str(band_id)
                for dev, lat in dev_lats.items():
                    if dev not in all_band_device_latencies[band_str]:
                        all_band_device_latencies[band_str][dev] = []
                    all_band_device_latencies[band_str][dev].append(lat)

        payload[scheduler] = {
            "device_avg_latencies": all_device_latencies,
            "device_msg_counts": all_device_counts,
            "source_avg_latencies": all_source_latencies,
            "per_band_device_latencies": all_band_device_latencies,
            "n_runs": len(dist_list),
        }

    path = os.path.join(output_dir, f"{scenario_key}_distributions.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Saved distribution data: {path}")
    return path


def print_statistical_summary(
    scenario_name: str,
    aggregated: Dict[str, Dict[str, StatisticsSummary]],
    comparisons: List[ComparisonResult],
):
    """
    Print comprehensive statistical summary.

    Args:
        scenario_name: Name of scenario
        aggregated: Aggregated statistics per scheduler
        comparisons: Statistical comparison results
    """
    print("\n" + "=" * 100)
    print(f"SCENARIO: {scenario_name}")
    print("=" * 100)

    # Overall performance with confidence intervals
    print("\nOVERALL PERFORMANCE (Mean ± Std, 95% CI)")
    print("=" * 100)
    print(f"{'Metric':<30} {'TRIAGE/4':<35} {'Strict':<35} {'FIFO':<35}")
    print("-" * 100)

    key_metrics = [
        ("alarm_avg_latency", "Alarm Avg Latency (s)"),
        ("alarm_p95_latency", "Alarm P95 Latency (s)"),
        ("band_1_fairness", "HIGH Band Fairness (Jain)"),
        ("band_2_fairness", "STANDARD Band Fairness"),
        ("band_3_fairness", "BACKGROUND Band Fairness"),
        ("device_latency_fairness", "Device Latency Fairness"),
        ("device_throughput_fairness", "Device Throughput Fairness"),
        ("min_device_rate", "Min Device Rate (msg/s)"),
        ("high_avg_latency", "HIGH Avg Latency (s)"),
    ]

    for metric_key, metric_label in key_metrics:
        seps_stat = aggregated["TRIAGE/4"][metric_key]
        strict_stat = aggregated["Strict"][metric_key]
        fifo_stat = aggregated["FIFO"][metric_key]

        print(
            f"{metric_label:<30} {str(seps_stat):<35} {str(strict_stat):<35} {str(fifo_stat):<35}"
        )

    # Comparison table with significance tests
    print("\n" + format_comparison_table(comparisons, "SIGNIFICANCE TESTS"))

    # Success criteria check (vs Strict Priority)
    print("\n" + "=" * 100)
    print("SUCCESS CRITERIA VALIDATION (TRIAGE/4 vs Strict Priority)")
    print("=" * 100)

    # Extract TRIAGE/4 metrics
    seps_alarm_latency = aggregated["TRIAGE/4"]["alarm_avg_latency"].mean
    seps_min_bandwidth = aggregated["TRIAGE/4"]["min_device_rate"].mean
    seps_high_fairness = aggregated["TRIAGE/4"]["band_1_fairness"].mean
    seps_std_fairness = aggregated["TRIAGE/4"]["band_2_fairness"].mean
    seps_bg_fairness = aggregated["TRIAGE/4"]["band_3_fairness"].mean
    seps_device_latency_fairness = aggregated["TRIAGE/4"]["device_latency_fairness"].mean
    seps_device_throughput_fairness = aggregated["TRIAGE/4"][
        "device_throughput_fairness"
    ].mean
    seps_high_latency = aggregated["TRIAGE/4"]["high_avg_latency"].mean

    # Extract Strict Priority baseline metrics for relative comparison
    strict_alarm_latency = aggregated["Strict"]["alarm_avg_latency"].mean
    strict_min_bandwidth = aggregated["Strict"]["min_device_rate"].mean
    strict_high_fairness = aggregated["Strict"]["band_1_fairness"].mean
    strict_std_fairness = aggregated["Strict"]["band_2_fairness"].mean
    strict_bg_fairness = aggregated["Strict"]["band_3_fairness"].mean
    strict_device_latency_fairness = aggregated["Strict"][
        "device_latency_fairness"
    ].mean
    strict_device_throughput_fairness = aggregated["Strict"][
        "device_throughput_fairness"
    ].mean
    strict_high_latency = aggregated["Strict"]["high_avg_latency"].mean

    # Tolerance for "no worse than" comparisons (5% margin)
    tolerance = 0.95

    # All criteria are relative to Strict Priority baseline
    criteria = [
        # Alarm latency: TRIAGE/4 should be better (lower) than Strict
        (
            "Alarm latency",
            seps_alarm_latency,
            strict_alarm_latency,
            f"<Strict ({strict_alarm_latency:.3f})",
            seps_alarm_latency <= strict_alarm_latency,
        ),
        # Bandwidth: TRIAGE/4 should be no worse than Strict
        (
            "Min bandwidth",
            seps_min_bandwidth,
            strict_min_bandwidth,
            f"≥Strict ({strict_min_bandwidth:.3f})",
            seps_min_bandwidth >= strict_min_bandwidth * tolerance,
        ),
        # Band fairness: TRIAGE/4 should be no worse than Strict
        (
            "HIGH band fairness",
            seps_high_fairness,
            strict_high_fairness,
            f"≥Strict ({strict_high_fairness:.3f})",
            seps_high_fairness >= strict_high_fairness * tolerance,
        ),
        (
            "STANDARD band fairness",
            seps_std_fairness,
            strict_std_fairness,
            f"≥Strict ({strict_std_fairness:.3f})",
            seps_std_fairness >= strict_std_fairness * tolerance,
        ),
        (
            "BACKGROUND band fairness",
            seps_bg_fairness,
            strict_bg_fairness,
            f"≥Strict ({strict_bg_fairness:.3f})",
            seps_bg_fairness >= strict_bg_fairness * tolerance,
        ),
        # Device fairness: TRIAGE/4 should be no worse than Strict
        (
            "Device latency fairness",
            seps_device_latency_fairness,
            strict_device_latency_fairness,
            f"≥Strict ({strict_device_latency_fairness:.3f})",
            seps_device_latency_fairness >= strict_device_latency_fairness * tolerance,
        ),
        (
            "Device throughput fairness",
            seps_device_throughput_fairness,
            strict_device_throughput_fairness,
            f"≥Strict ({strict_device_throughput_fairness:.3f})",
            seps_device_throughput_fairness
            >= strict_device_throughput_fairness * tolerance,
        ),
        # HIGH priority overhead: TRIAGE/4 should not add excessive latency to HIGH band
        (
            "HIGH priority latency",
            seps_high_latency,
            strict_high_latency,
            f"≤1.2×Strict ({strict_high_latency:.3f})",
            (
                seps_high_latency <= strict_high_latency * 1.2
                if strict_high_latency > 0
                else True
            ),
        ),
    ]

    passed = 0
    ties = 0
    for criterion, seps_value, strict_value, target, met in criteria:
        value_str = f"{seps_value:.3f}"
        # Check if values are essentially equal (tie)
        is_tie = abs(seps_value - strict_value) < 1e-6
        if is_tie:
            status = "≈"  # Tie - values identical
            ties += 1
        elif met:
            status = "✓"  # Pass - TRIAGE/4 better or within tolerance
            passed += 1
        else:
            status = "✗"  # Fail - TRIAGE/4 worse than allowed

        print(f"{status} {criterion:<30} {value_str:<15} (target: {target})")

    print(
        f"\nSummary: {passed} passed, {ties} tied, {len(criteria) - passed - ties} failed"
    )


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="TRIAGE/4 Statistical Benchmark with Multi-Run Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 runs)
  python benchmarks/statistical_benchmark.py --n-runs 5

  # Recommended (50 runs for statistical rigor)
  python benchmarks/statistical_benchmark.py --n-runs 50

  # Single scenario
  python benchmarks/statistical_benchmark.py --scenario alarm_under_burst --n-runs 50

  # All scenarios
  python benchmarks/statistical_benchmark.py --all --n-runs 50
        """,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=[
            "alarm_under_burst",
            "alarm_under_burst_phased",
            "device_monopolization",
            "device_monopolization_sweep",
            "multi_zone_emergency",
            "multi_zone_emergency_cascade",
            "alarm_flood_attack",
            "alarm_malfunction_surge",
            "legit_extreme_emergency",
            "alarm_load_near_saturation_constrained",
        ],
        default="alarm_under_burst",
        help="Scenario to run (default: alarm_under_burst)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all scenarios",
    )
    parser.add_argument(
        "--enable-alarm-protection",
        action="store_true",
        default=True,
        help="Enable adaptive alarm protection in TRIAGE/4 during benchmarking (default: True)",
    )
    parser.add_argument(
        "--export-fairness-table",
        action="store_true",
        help="Export fairness/throughput summary table to CSV across scenarios",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=50,
        help="Number of runs with different seeds (default: 50, use 5 for quick test)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=999,
        help="Base seed for runs (seeds will be base_seed, base_seed+1, ..., base_seed+n_runs-1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/statistical",
        help="Output directory for results (default: results/statistical)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define scenarios
    scenarios = {
        # Original TRIAGE/4 core scenarios
        "alarm_under_burst": (generate_alarm_under_burst, "Alarm Under Burst", None),
        "alarm_under_burst_phased": (
            generate_alarm_under_burst_phased,
            "Alarm Under Burst (Phased)",
            None,
        ),
        "device_monopolization": (
            generate_device_monopolization,
            "Device Monopolization",
            None,
        ),
        "device_monopolization_sweep": (
            generate_device_monopolization_sweep,
            "Device Monopolization (Sweep)",
            None,
        ),
        "multi_zone_emergency": (
            generate_multi_zone_emergency,
            "Multi-Zone Emergency",
            None,
        ),
        "multi_zone_emergency_cascade": (
            generate_multi_zone_emergency_cascade,
            "Multi-Zone Emergency (Cascade)",
            None,
        ),
        # Adaptive protection scenarios from prior work
        "alarm_flood_attack": (
            generate_alarm_flood_attack,
            "Alarm Flood Attack",
            20.0,
        ),
        "alarm_malfunction_surge": (
            lambda seed=None: generate_alarm_malfunction_surge(
                heavy_zone=0,
                light_zones=[1, 2, 3, 4, 5, 6, 7],
                heavy_rate=12.0,
                light_rate=1.14,
                duration=20.0,
                legitimate_alarms=0,
                seed=seed,
            ),
            "Alarm Malfunction Surge (Forced Drops)",
            None,
        ),
        "legit_extreme_emergency": (
            generate_legit_extreme_emergency,
            "Legitimate Extreme Emergency",
            None,
        ),
        # Axis 1: Alarm rate sweep (activation / threshold)
        "alarm_rate_sweep_low": (
            lambda seed=None: generate_alarm_rate_sweep(
                alarm_rate=0.5,
                duration=30.0,
                background_rate=10.0,
                seed=seed,
            ),
            "Alarm Rate Sweep (0.5/s)",
            None,
        ),
        "alarm_rate_sweep_threshold": (
            lambda seed=None: generate_alarm_rate_sweep(
                alarm_rate=5.0,
                duration=30.0,
                background_rate=10.0,
                seed=seed,
            ),
            "Alarm Rate Sweep (5.0/s)",
            None,
        ),
        "alarm_rate_sweep_attack": (
            lambda seed=None: generate_alarm_rate_sweep(
                alarm_rate=20.0,
                duration=30.0,
                background_rate=10.0,
                seed=seed,
            ),
            "Alarm Rate Sweep (20.0/s)",
            None,
        ),
        # Axis 2: Skewed multi-source fairness
        "skewed_alarm_sources": (
            generate_skewed_alarm_sources,
            "Skewed Alarm Sources",
            None,
        ),
        # Axis 3: Load regime sweep with fixed alarm pattern
        "alarm_load_underload": (
            lambda seed=None: generate_alarm_load_regime(
                utilization=0.3,
                service_rate=20.0,
                duration=30.0,
                alarm_rate=1.0,
                seed=seed,
            ),
            "Alarm Load Regime (ρ≈0.3)",
            20.0,
        ),
        "alarm_load_optimal": (
            lambda seed=None: generate_alarm_load_regime(
                utilization=0.7,
                service_rate=20.0,
                duration=30.0,
                alarm_rate=1.0,
                seed=seed,
            ),
            "Alarm Load Regime (ρ≈0.7)",
            20.0,
        ),
        "alarm_load_near_saturation": (
            lambda seed=None: generate_alarm_load_regime(
                utilization=0.95,
                service_rate=20.0,
                duration=30.0,
                alarm_rate=1.0,
                seed=seed,
            ),
            "Alarm Load Regime (ρ≈0.95)",
            20.0,
        ),
        "alarm_load_near_saturation_constrained": (
            lambda seed=None: generate_alarm_load_near_saturation_constrained(
                utilization=0.95,
                service_rate=20.0,
                duration=30.0,
                alarm_rate=1.0,
                seed=seed,
            ),
            "Alarm Load Regime (ρ≈0.95 Token-Constrained)",
            20.0,
        ),
    }

    # Run scenarios
    aggregated_records = []

    if args.all:
        for scenario_key, (generator, name, service_rate_override) in scenarios.items():
            aggregated, comparisons, aggregated_phase, phase_boundaries, distributions = (
                run_statistical_analysis(
                    workload_generator=generator,
                    scenario_name=name,
                    n_runs=args.n_runs,
                    base_seed=args.base_seed,
                    output_dir=args.output_dir,
                    enable_alarm_protection=args.enable_alarm_protection,
                    service_rate_override=service_rate_override,
                )
            )
            print_statistical_summary(name, aggregated, comparisons)
            agg_path = export_aggregated_metrics(
                aggregated,
                scenario_key,
                args.output_dir,
                aggregated_phase,
                phase_boundaries,
            )
            export_distribution_data(distributions, scenario_key, args.output_dir)
            aggregated_records.append((scenario_key, name, agg_path))
            print("\n" * 2)
    else:
        generator, name, service_rate_override = scenarios[args.scenario]
        aggregated, comparisons, aggregated_phase, phase_boundaries, distributions = (
            run_statistical_analysis(
                workload_generator=generator,
                scenario_name=name,
                n_runs=args.n_runs,
                base_seed=args.base_seed,
                output_dir=args.output_dir,
                enable_alarm_protection=args.enable_alarm_protection,
                service_rate_override=service_rate_override,
            )
        )
        print_statistical_summary(name, aggregated, comparisons)
        agg_path = export_aggregated_metrics(
            aggregated,
            args.scenario,
            args.output_dir,
            aggregated_phase,
            phase_boundaries,
        )
        export_distribution_data(distributions, args.scenario, args.output_dir)
        aggregated_records.append((args.scenario, name, agg_path))

    # Export fairness table across scenarios
    if args.export_fairness_table and aggregated_records:
        export_fairness_table(
            aggregated_records,
            output_path=os.path.join(
                args.output_dir, "fairness_throughput_summary.csv"
            ),
        )

    # Always export comprehensive results
    if aggregated_records:
        export_comprehensive_results(
            aggregated_records,
            output_path=os.path.join(args.output_dir, "comprehensive_results.csv"),
        )

    print("\n" + "=" * 100)
    print("Statistical benchmark complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()

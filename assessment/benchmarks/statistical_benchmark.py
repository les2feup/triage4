"""
TRIAGE/4 Statistical Benchmark with Multi-Run Analysis.

Runs TRIAGE/4, Strict Priority, FIFO, WFQ, DRR, TBP, and five TRIAGE/4
ablation variants (T4-NoSemantic, T4-FIFOInBand, T4-NoTokens, T4-NoAAP,
T4-NoSourceLimit) multiple times with different seeds to compute statistically
rigorous comparisons with confidence intervals and significance testing.

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

    # Subset of schedulers (e.g. only baselines, no ablations)
    python benchmarks/statistical_benchmark.py --schedulers Strict,FIFO,WFQ,DRR,TBP
"""

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assessment.baselines import SCHEDULER_REGISTRY
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
    ROBUSTNESS_SCENARIOS,
    Workload,
    generate_alarm_load_regime,
    generate_alarm_load_near_saturation_constrained,
    generate_alarm_rate_sweep,
    generate_alarm_under_burst,
    generate_alarm_under_burst_phased,
    generate_device_monopolization,
    generate_device_monopolization_sweep,
    generate_multi_zone_emergency,
    generate_multi_zone_emergency_cascade,
    generate_skewed_alarm_sources,
)

# Leave-one-out ablation variant names.  These are constructed inline in
# run_statistical_analysis from the fully-resolved TRIAGE/4 base config so
# they inherit all scenario-specific overrides.
ABLATION_NAMES: List[str] = [
    "T4-NoSemantic",
    "T4-FIFOInBand",
    "T4-NoTokens",
    "T4-NoAAP",
    "T4-NoSourceLimit",
]


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
        source_is_legitimate=workload.source_is_legitimate,
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
    config_factory=None,
    scheduler_names: Optional[List[str]] = None,
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
        service_rate_override: If set, overrides the config service rate after factory
        config_factory: Optional callable returning a TRIAGE4Config. When provided
            the returned config is the base; service_rate_override and scenario-specific
            token overrides are then applied on top (Factory → Rate override → Scenario).
        scheduler_names: Optional subset of non-TRIAGE/4 scheduler names to run.
            Accepts registry names (Strict, FIFO, WFQ, DRR, TBP) and ablation names
            (T4-NoSemantic, T4-FIFOInBand, T4-NoTokens, T4-NoAAP,
            T4-NoSourceLimit). Defaults to all.

    Returns:
        Tuple of (aggregated_metrics, comparisons, aggregated_phase, phase_boundaries, all_distributions)
            - aggregated_metrics: Dict[scheduler_name, Dict[metric_name, StatisticsSummary]]
            - comparisons: List[ComparisonResult] with significance tests
            - aggregated_phase: Per-phase statistics (if phased workload)
            - phase_boundaries: Phase time boundaries (if phased workload)
            - all_distributions: Raw per-device/per-source distribution data for plotting
    """
    # Resolve the active scheduler list for this run
    active_registry = {
        k: v for k, v in SCHEDULER_REGISTRY.items()
        if scheduler_names is None or k in scheduler_names
    }
    active_ablations = [
        n for n in ABLATION_NAMES
        if scheduler_names is None or n in scheduler_names
    ]
    all_scheduler_names = ["TRIAGE/4"] + list(active_registry.keys()) + active_ablations

    print("=" * 100)
    print(f"STATISTICAL BENCHMARK: {scenario_name}")
    print("=" * 100)
    print(f"Number of runs: {n_runs}")
    print(f"Seeds: {base_seed} to {base_seed + n_runs - 1}")
    print(f"Schedulers: {', '.join(all_scheduler_names)}")
    print()

    # Storage for all runs
    all_metrics: Dict[str, List] = {name: [] for name in all_scheduler_names}
    all_distributions: Dict[str, List[DistributionData]] = {
        name: [] for name in all_scheduler_names
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
                    for sched in all_scheduler_names
                }

        # --- Build fully-resolved TRIAGE/4 config ---
        # Priority: Factory Config → service_rate_override → scenario constraints
        if config_factory is not None:
            triage4_config = config_factory()
        else:
            triage4_config = create_triage4_default()
        if service_rate_override is not None:
            triage4_config.service_rate = service_rate_override

        # Constrained near-saturation scenario: tighten token budgets so total
        # non-alarm capacity is ≈100% of service rate instead of the usual 200%.
        if scenario_name.startswith("Alarm Load Regime (ρ≈0.95 Token-Constrained"):
            triage4_config.high_token_budget = 10
            triage4_config.standard_token_budget = 7
            triage4_config.background_token_budget = 3

        if enable_alarm_protection:
            triage4_config.enable_alarm_protection = True

        # --- Run TRIAGE/4 ---
        seps = TRIAGE4Scheduler(triage4_config, scheduler_seed=seed)
        seps_metrics, seps_result, seps_dist = run_scenario(seps, workload, "TRIAGE/4")
        all_metrics["TRIAGE/4"].append(seps_metrics)
        all_distributions["TRIAGE/4"].append(seps_dist)
        if phase_metrics is not None:
            _accumulate_phase(phase_metrics, "TRIAGE/4", workload, seps_result, phase_boundaries)

        # --- Run leave-one-out ablations (clone base config, flip exactly one flag) ---
        ablation_flags = {
            "T4-NoSemantic": {"disable_semantic_override": True},
            "T4-FIFOInBand": {"within_band_fifo": True},
            "T4-NoTokens":   {"disable_token_buckets": True},
            "T4-NoAAP":      {"enable_alarm_protection": False},
            # Keeps AAP's band-global backstop but removes per-source limiting,
            # isolating what per-source discrimination contributes.
            "T4-NoSourceLimit": {"disable_source_rate_limit": True},
        }
        for abl_name in active_ablations:
            abl_cfg = dataclasses.replace(triage4_config, **ablation_flags[abl_name])
            abl_sched = TRIAGE4Scheduler(abl_cfg, scheduler_seed=seed)
            abl_metrics, abl_result, abl_dist = run_scenario(abl_sched, workload, abl_name)
            all_metrics[abl_name].append(abl_metrics)
            all_distributions[abl_name].append(abl_dist)
            if phase_metrics is not None:
                _accumulate_phase(phase_metrics, abl_name, workload, abl_result, phase_boundaries)

        # --- Run registry schedulers (Strict, FIFO, WFQ, DRR, TBP) ---
        # Registry factories receive only (service_rate, seed); scenario-specific token
        # overrides (e.g. constrained-saturation budgets) are NOT passed.  TBP therefore
        # always uses TRIAGE/4 default token parameters — it is an external classical
        # baseline, not a parameter-matched ablation.  T4-NoSemantic (constructed above
        # via dataclasses.replace) does inherit all scenario overrides.
        for reg_name, entry in active_registry.items():
            reg_sched = entry.factory(triage4_config.service_rate, seed)
            reg_metrics, reg_result, reg_dist = run_scenario(reg_sched, workload, reg_name)
            all_metrics[reg_name].append(reg_metrics)
            all_distributions[reg_name].append(reg_dist)
            if phase_metrics is not None:
                _accumulate_phase(phase_metrics, reg_name, workload, reg_result, phase_boundaries)

    # --- Aggregate statistics ---
    print("\n" + "=" * 100)
    print("AGGREGATING RESULTS ACROSS ALL RUNS")
    print("=" * 100)

    # Key metrics to aggregate (same set for all schedulers)
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
        "alarm_protection_activations",
        "alarm_protection_deactivations",
        # Per-source limiter and drop attribution
        "alarm_source_limit_activations",
        "alarm_sources_limited",
        "legitimate_alarm_dropped",
        "legitimate_alarm_dropped_rate",
        "abnormal_alarm_dropped_rate",
        "legitimate_alarm_n",
        "abnormal_alarm_n",
    ]

    aggregated = {}
    for scheduler_name in all_scheduler_names:
        aggregated[scheduler_name] = {}
        metrics_list = all_metrics[scheduler_name]
        for metric_key in metric_keys:
            # Drop-attribution metrics exist only for scenarios that label their
            # source classes; skip rather than invent a value, since a fabricated
            # zero would read as "nothing legitimate was shed" where the truth is
            # "the scenario draws no such distinction".
            if not all(metric_key in m for m in metrics_list):
                continue
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

    # --- Statistical comparisons (TRIAGE/4 vs every other active scheduler) ---
    comparisons = []
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

    seps_values_cache = {
        mk: [m[mk] for m in all_metrics["TRIAGE/4"]]
        for mk, _ in comparison_metrics
    }
    for other_name in all_scheduler_names:
        if other_name == "TRIAGE/4":
            continue
        for metric_key, metric_label in comparison_metrics:
            other_values = [m[metric_key] for m in all_metrics[other_name]]
            comparisons.append(compare_schedulers(
                scheduler_a_name="TRIAGE/4",
                scheduler_b_name=other_name,
                metric_name=metric_label,
                values_a=seps_values_cache[metric_key],
                values_b=other_values,
            ))

    return aggregated, comparisons, aggregated_phase, phase_boundaries, all_distributions


def _accumulate_phase(
    phase_metrics: Dict,
    scheduler_name: str,
    workload,
    result,
    phase_boundaries,
) -> None:
    """Accumulate per-phase statistics for one scheduler run."""
    phase_vals = compute_phase_metrics(workload, result, phase_boundaries)
    for phase_idx in range(len(phase_boundaries)):
        phase_metrics[scheduler_name]["phase_high_wait"][phase_idx].append(
            phase_vals["phase_high_wait"][phase_idx]
        )
        phase_metrics[scheduler_name]["phase_overall_wait"][phase_idx].append(
            phase_vals["phase_overall_wait"][phase_idx]
        )
        phase_metrics[scheduler_name]["phase_high_fairness"][phase_idx].append(
            phase_vals["phase_high_fairness"][phase_idx]
        )


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

    # Derive scheduler list from the first JSON record
    with open(aggregated_records[0][2], "r", encoding="utf-8") as _f:
        _first = json.load(_f)
    schedulers = list(_first.keys())

    metrics = [
        "avg_device_rate",
        "avg_waiting_time",
        "device_throughput_fairness",
        "device_latency_fairness",
    ]

    # Sanitise scheduler names for CSV column headers (replace / and - with _)
    def _col(sched: str) -> str:
        return sched.replace("/", "_").replace("-", "_").lower()

    rows = []
    for scenario_key, scenario_name, path in aggregated_records:
        with open(path, "r", encoding="utf-8") as f:
            agg = json.load(f)
        row = {"scenario_key": scenario_key, "scenario_name": scenario_name}
        for metric in metrics:
            for sched in schedulers:
                if sched in agg:
                    row[f"{metric}_{_col(sched)}"] = agg[sched][metric]["mean"]
        rows.append(row)

    fieldnames = ["scenario_key", "scenario_name"] + [
        f"{metric}_{_col(sched)}" for metric in metrics for sched in schedulers
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

    # Derive scheduler list dynamically from the first JSON record so new
    # baselines and ablation variants are included automatically.
    with open(aggregated_records[0][2], "r", encoding="utf-8") as _f:
        _first = json.load(_f)
    schedulers = list(_first.keys())

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
        "alarm_protection_activations",
        "alarm_protection_deactivations",
        # Per-source limiter and drop attribution
        "alarm_source_limit_activations",
        "alarm_sources_limited",
        "legitimate_alarm_dropped",
        "legitimate_alarm_dropped_rate",
        "abnormal_alarm_dropped_rate",
        "legitimate_alarm_n",
        "abnormal_alarm_n",
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

            # Protection metrics for all schedulers: use 0.0 when the metric exists
            # (non-TRIAGE/4 schedulers produce 0 drops and no activations), NA only
            # when the key is absent from the JSON (should not occur after Stage 3).
            for metric in protection_metrics:
                if metric in agg[sched]:
                    row[f"{metric}_mean"] = agg[sched][metric]["mean"]
                    row[f"{metric}_std"] = agg[sched][metric]["std"]
                else:
                    row[f"{metric}_mean"] = "NA"
                    row[f"{metric}_std"] = "NA"

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
    active_names = list(aggregated.keys())
    col_w = max(30, 100 // max(len(active_names) + 1, 1))
    header = f"{'Metric':<30}" + "".join(f"{n:<{col_w}}" for n in active_names)
    print("\nOVERALL PERFORMANCE (Mean ± Std, 95% CI)")
    print("=" * 100)
    print(header)
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
        row = f"{metric_label:<30}"
        for name in active_names:
            row += f"{str(aggregated[name][metric_key]):<{col_w}}"
        print(row)

    # Comparison table with significance tests
    print("\n" + format_comparison_table(comparisons, "SIGNIFICANCE TESTS"))

    # Success criteria check (vs Strict Priority) — only when Strict is in the active set
    if "Strict" not in aggregated:
        print("\n[Success criteria skipped: Strict Priority not in active scheduler set]")
        return

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
    parser.add_argument(
        "--schedulers",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of non-TRIAGE/4 scheduler names to run. "
            "Registry names: Strict,FIFO,WFQ,DRR,TBP. "
            "Ablation names: T4-NoSemantic,T4-FIFOInBand,T4-NoTokens,T4-NoAAP,"
            "T4-NoSourceLimit. "
            "Defaults to all."
        ),
    )

    args = parser.parse_args()

    scheduler_names = (
        [s.strip() for s in args.schedulers.split(",")]
        if args.schedulers
        else None
    )

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
        # Adaptive protection scenarios (R1-R3). Defined in assessment.workloads
        # .robustness so the sensitivity sweep runs these exact workloads.
        **ROBUSTNESS_SCENARIOS,
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
                    scheduler_names=scheduler_names,
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
                scheduler_names=scheduler_names,
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

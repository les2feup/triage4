"""
Compare TRIAGE/4 token-bucket configurations using the existing statistical benchmark.

Runs the full statistical benchmark for two configurations:
    1) Legacy budgets:  high=15, standard=10, background=5
    2) Current budgets: high=20, standard=15, background=5

For each scenario, prints the key TRIAGE/4 metrics and the relative change
between configurations. This script does not modify library code; it
temporarily overrides the `create_triage4_default` factory used by the
statistical benchmark and restores it afterwards.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

import assessment.benchmarks.statistical_benchmark as sb
from triage4 import TRIAGE4Config
from assessment.workloads import (
    Workload,
    generate_alarm_flood_attack,
    generate_alarm_malfunction_surge,
    generate_alarm_under_burst,
    generate_alarm_under_burst_phased,
    generate_device_monopolization,
    generate_device_monopolization_sweep,
    generate_legit_extreme_emergency,
    generate_multi_zone_emergency,
    generate_multi_zone_emergency_cascade,
)


# Scenario mapping mirrors statistical_benchmark.main
SCENARIOS: Dict[str, Tuple] = {
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
    "alarm_flood_attack": (
        generate_alarm_flood_attack,
        "Alarm Flood Attack",
        20.0,
    ),
    "alarm_malfunction_surge": (
        generate_alarm_malfunction_surge,
        "Alarm Malfunction Surge",
        None,
    ),
    "legit_extreme_emergency": (
        generate_legit_extreme_emergency,
        "Legitimate Extreme Emergency",
        None,
    ),
}


MetricDict = Dict[str, float]
ScenarioMetrics = Dict[str, MetricDict]


def _run_for_config(
    label: str,
    high: int,
    standard: int,
    background: int,
    n_runs: int = 30,
    base_seed: int = 999,
) -> ScenarioMetrics:
    """
    Run statistical analysis for all scenarios using a specific TRIAGE/4 token budget.

    The TRIAGE4Config factory used by the statistical benchmark is temporarily
    overridden to ensure consistent configuration for all runs.
    """

    def create_triage4_custom_default() -> TRIAGE4Config:
        return TRIAGE4Config(
            high_token_budget=high,
            standard_token_budget=standard,
            background_token_budget=background,
        )

    original_factory = sb.create_triage4_default
    sb.create_triage4_default = create_triage4_custom_default

    results_by_scenario: ScenarioMetrics = {}

    try:
        print(
            f"\n=== CONFIG {label}: "
            f"high={high}, standard={standard}, background={background} ==="
        )

        for key, (generator, name, service_rate_override) in SCENARIOS.items():
            print(f"\n--- Scenario: {key} / {name} ---")

            aggregated, _comparisons, _phase_stats, _phase_bounds = (
                sb.run_statistical_analysis(
                    workload_generator=generator,
                    scenario_name=name,
                    n_runs=n_runs,
                    base_seed=base_seed,
                    output_dir="results/statistical",  # kept for consistency
                    enable_alarm_protection=True,
                    service_rate_override=service_rate_override,
                )
            )

            seps_stats = aggregated["TRIAGE/4"]

            metrics: MetricDict = {
                "alarm_avg_latency": seps_stats["alarm_avg_latency"].mean,
                "alarm_p95_latency": seps_stats["alarm_p95_latency"].mean,
                "min_device_rate": seps_stats["min_device_rate"].mean,
                "band_1_fairness": seps_stats["band_1_fairness"].mean,
                "device_latency_fairness": seps_stats["device_latency_fairness"].mean,
                "high_avg_latency": seps_stats["high_avg_latency"].mean,
            }
            results_by_scenario[key] = metrics

            print(
                "TRIAGE/4 ({cfg})  alarm_avg={alarm_avg:.4f}s  p95={p95:.4f}s  "
                "min_rate={rate:.4f}  high_fair={hf:.4f}  "
                "dev_fair={df:.4f}  high_lat={hl:.4f}s".format(
                    cfg=label,
                    alarm_avg=metrics["alarm_avg_latency"],
                    p95=metrics["alarm_p95_latency"],
                    rate=metrics["min_device_rate"],
                    hf=metrics["band_1_fairness"],
                    df=metrics["device_latency_fairness"],
                    hl=metrics["high_avg_latency"],
                )
            )
    finally:
        sb.create_triage4_default = original_factory

    return results_by_scenario


def _print_summary(old_results: ScenarioMetrics, new_results: ScenarioMetrics) -> None:
    """Print per-scenario comparison between legacy and current budgets."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY: legacy (15/10/5) vs current (20/15/5)")
    print("=" * 80)

    metrics_order: List[str] = [
        "alarm_avg_latency",
        "alarm_p95_latency",
        "min_device_rate",
        "band_1_fairness",
        "device_latency_fairness",
        "high_avg_latency",
    ]

    for scenario_key in SCENARIOS.keys():
        old = old_results[scenario_key]
        new = new_results[scenario_key]
        print(f"\nScenario: {scenario_key}")
        for metric in metrics_order:
            old_v = old[metric]
            new_v = new[metric]
            delta = new_v - old_v
            rel = (delta / old_v * 100.0) if old_v != 0 else float("nan")
            print(
                f"  {metric:22s}: "
                f"old={old_v:9.6f}  new={new_v:9.6f}  "
                f"Δ={delta:+9.6f}  rel={rel:+7.2f}%"
            )


def main() -> None:
    """Entry point to run the comparison for both token-budget configurations."""
    # Ensure reproducible seeds inside this script as well
    np.random.seed(999)

    legacy_results = _run_for_config(
        label="legacy_15_10_5",
        high=15,
        standard=10,
        background=5,
        n_runs=30,
        base_seed=999,
    )

    current_results = _run_for_config(
        label="current_20_15_5",
        high=20,
        standard=15,
        background=5,
        n_runs=30,
        base_seed=999,
    )

    _print_summary(legacy_results, current_results)


if __name__ == "__main__":
    main()

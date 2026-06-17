#!/usr/bin/env python
"""
Inverse WCRT Analysis: Effective Burst Parameter Derivation.

Uses empirical P95 latency data to back-calculate the effective burst handling
capacity demonstrated by TRIAGE/4, showing the gap to conservative bounds.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from assessment.triage4_wcrt import (
    analyze_wcrt_gap,
    load_results_csv,
    lookup_observed_delay,
    AlarmScenario,
    SCENARIO_NAME_MAP,
)


def _resolve_results_path() -> pathlib.Path:
    candidates = [
        pathlib.Path("results/statistical_extended/comprehensive_results.csv"),
        pathlib.Path("results/statistical/comprehensive_results.csv"),
        pathlib.Path("comprehensive_results.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "comprehensive_results.csv not found in expected locations."
    )


def main() -> None:
    results_path = _resolve_results_path()
    results_df = load_results_csv(str(results_path))

    design_sigma = 10.0
    design_rho = 5.0
    service_rate = 20.0

    scenarios = [
        "Alarm Under Burst",
        "Alarm Flood Attack",
        "Legitimate Extreme Emergency",
    ]

    print("=" * 80)
    print("INVERSE WCRT ANALYSIS: Effective Burst Parameter Derivation")
    print("=" * 80)
    print()
    print("Conservative Design Parameters:")
    print(f"  sigma_A (burst):      {design_sigma} messages")
    print(f"  rho_A (rate):         {design_rho} messages/sec")
    print(f"  C (service rate):     {service_rate} messages/sec")
    print()

    for scenario_name in scenarios:
        SCENARIO_NAME_MAP.clear()
        SCENARIO_NAME_MAP.update({"Inverse": scenario_name})
        scenario = AlarmScenario(
            name="Inverse",
            C=service_rate,
            sigma_A=design_sigma,
            rho_A=design_rho,
            baseline="TRIAGE/4",
        )
        observed_p95 = lookup_observed_delay(results_df, scenario, metric="p95_mean")

        analysis = analyze_wcrt_gap(
            scenario_name=scenario_name,
            design_sigma=design_sigma,
            design_rho=design_rho,
            observed_p95=observed_p95,
            service_rate=service_rate,
        )

        print("-" * 80)
        print(f"Scenario: {scenario_name}")
        print(f"  Design WCRT (conservative):  {analysis['design_wcrt']:.4f} s")
        print(f"  Observed P95 (empirical):    {analysis['observed_p95']:.4f} s")
        print(f"  Effective sigma_eff:         {analysis['effective_sigma']:.2f} msgs")
        print(f"  Conservatism factor:         {analysis['conservatism_factor']:.1f}x")
        print(
            f"  Performance regime:          {analysis['performance_regime'].replace('_', ' ')}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()

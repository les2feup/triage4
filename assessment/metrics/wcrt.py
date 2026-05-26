"""
Analytical Worst-Case Response Time (WCRT) bounds for the ALARM band in TRIAGE/4.

Assumptions:
- Single non-preemptive server with constant service rate C [msg/s].
- Service time per message is S = 1 / C, so an ALARM message can be blocked by
  at most one in-progress non-ALARM transmission.
- Aggregate arrivals follow a leaky-bucket (sigma, rho) constraint with the
  stability condition rho < C.

The main bound implemented is:
    R_max <= (sigma + 1) / C + rho / C^2
where sigma, rho represent the burst and rate of the arriving ALARM traffic
or protection parameters when Adaptive Alarm Protection (AAP) is enabled.

Strict-priority baseline:
- An emergency flow with (sigma_E, rho_E) delayed by an aggregate of
  higher-priority classes (sigma_I, rho_I) has totals sigma_tot = sigma_E +
  sigma_I and rho_tot = rho_E + rho_I, giving:
      R_strict_max = (sigma_tot + 1) / C + rho_tot / C^2.
- Example: C = 20, sigma_E = 10, rho_E = 5, sigma_I = 5, rho_I = 4 gives
  sigma_tot = 15, rho_tot = 9, and R_strict_max = 0.8225 s.
"""

import pathlib
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import pandas as pd


def _validate_service_rate(C: float) -> None:
    if C <= 0:
        raise ValueError("Service rate C must be positive.")


def _validate_arrival_pair(sigma: float, rho: float, C: float, prefix: str) -> None:
    if sigma < 0:
        raise ValueError(f"Burst {prefix} must be non-negative.")
    if rho < 0:
        raise ValueError(f"Arrival rate {prefix} must be non-negative.")
    if rho >= C:
        raise ValueError(
            f"Unstable configuration: arrival rate {prefix} must be"
            " strictly less than service rate C."
        )


def compute_wcrt_alarm(C: float, sigma_A: float, rho_A: float) -> float:
    """
    Compute the WCRT upper bound for the ALARM band under base TRIAGE/4.

    Parameters
    ----------
    C : float
        Service rate in messages per second.
    sigma_A : float
        Burst size of ALARM arrivals in messages.
    rho_A : float
        Long-term average ALARM arrival rate in messages per second.

    Returns
    -------
    float
        WCRT upper bound in seconds.

    Raises
    ------
    ValueError
        If inputs are negative, non-positive, or violate rho_A < C.
    """
    _validate_service_rate(C)
    _validate_arrival_pair(sigma_A, rho_A, C, "sigma_A/rho_A")
    return (sigma_A + 1.0) / C + rho_A / (C**2)


def compute_wcrt_alarm_aap(C: float, b_P: float, r_P: float) -> float:
    """
    Compute the WCRT upper bound for the ALARM band when AAP is enabled.

    Parameters
    ----------
    C : float
        Service rate in messages per second.
    b_P : float
        AAP burst capacity in messages.
    r_P : float
        AAP token rate in messages per second.

    Returns
    -------
    float
        WCRT upper bound in seconds under AAP.

    Raises
    ------
    ValueError
        If inputs are negative, non-positive, or violate r_P < C.
    """
    return compute_wcrt_alarm(C, sigma_A=b_P, rho_A=r_P)


def inverse_wcrt_burst(observed_latency: float, arrival_rate: float, service_rate: float) -> float:
    """
    Compute effective burst parameter from observed latency.

    Solves: sigma_eff = (R_obs - rho/C^2) * C - 1
    """
    _validate_service_rate(service_rate)
    if observed_latency < 0:
        raise ValueError("observed_latency must be non-negative.")
    if arrival_rate < 0:
        raise ValueError("arrival_rate must be non-negative.")
    if arrival_rate >= service_rate:
        raise ValueError(
            "Unstable configuration: arrival_rate must be strictly less than service_rate."
        )

    sigma_eff = (observed_latency - arrival_rate / (service_rate**2)) * service_rate - 1
    return max(0.0, sigma_eff)


def analyze_wcrt_gap(
    scenario_name: str,
    design_sigma: float,
    design_rho: float,
    observed_p95: float,
    service_rate: float,
) -> dict:
    """
    Analyze gap between conservative design WCRT and observed performance.
    """
    design_wcrt = compute_wcrt_alarm(service_rate, sigma_A=design_sigma, rho_A=design_rho)
    sigma_eff = inverse_wcrt_burst(observed_p95, design_rho, service_rate)
    conservatism = design_wcrt / observed_p95 if observed_p95 > 0 else float("inf")

    return {
        "scenario": scenario_name,
        "design_sigma": design_sigma,
        "design_rho": design_rho,
        "design_wcrt": design_wcrt,
        "observed_p95": observed_p95,
        "effective_sigma": sigma_eff,
        "conservatism_factor": conservatism,
        "performance_regime": "tens_of_ms" if observed_p95 < 0.1 else "sub_second",
    }


def compute_wcrt_strict_emergency(
    C: float, sigma_E: float, rho_E: float, interferers: List[Tuple[float, float]]
) -> float:
    """
    Compute a WCRT upper bound for an emergency flow under strict priority.

    The emergency flow is assumed non-preemptive and can be delayed by a single
    in-service lower-priority message (at most 1 / C seconds) plus backlog from
    higher-priority interferers.

    Parameters
    ----------
    C : float
        Service rate in messages per second.
    sigma_E : float
        Burst size of the emergency flow in messages.
    rho_E : float
        Long-term average arrival rate of the emergency flow in messages per
        second.
    interferers : List[Tuple[float, float]]
        List of (sigma, rho) pairs for higher-priority classes that can delay
        the emergency flow.

    Returns
    -------
    float
        WCRT upper bound in seconds for the emergency flow.

    Raises
    ------
    ValueError
        If any parameter is invalid or the total arrival rate is unstable
        (sum of rates for emergency and interferers is >= C).
    """
    _validate_service_rate(C)
    _validate_arrival_pair(sigma_E, rho_E, C, "sigma_E/rho_E")

    sigma_hp = 0.0
    rho_hp = 0.0
    for idx, (sigma_i, rho_i) in enumerate(interferers):
        _validate_arrival_pair(sigma_i, rho_i, C, f"interferer[{idx}]")
        sigma_hp += sigma_i
        rho_hp += rho_i

    total_rho = rho_E + rho_hp
    if total_rho >= C:
        raise ValueError(
            "Unstable configuration: combined arrival rate (emergency +"
            " interferers) must be strictly less than service rate C."
        )

    sigma_total = sigma_E + sigma_hp
    return (sigma_total + 1.0) / C + total_rho / (C**2)


@dataclass
class AlarmScenario:
    """
    Container for TRIAGE/4 or strict-priority WCRT evaluation scenarios.

    baseline values:
    - "TRIAGE/4": base TRIAGE/4 ALARM bound.
    - "TRIAGE/4-AAP": TRIAGE/4 with Adaptive Alarm Protection parameters.
    - "STRICT": strict-priority baseline using sigma_A/rho_A for the emergency
      flow and optional interferers as higher-priority classes.
    """

    name: str
    C: float
    sigma_A: Optional[float] = None
    rho_A: Optional[float] = None
    b_P: Optional[float] = None
    r_P: Optional[float] = None
    deadline_s: Optional[float] = None
    observed_delay_s: Optional[float] = None
    baseline: str = "TRIAGE/4"
    interferers: Optional[List[Tuple[float, float]]] = None

    def is_aap(self) -> bool:
        return self.baseline.upper() == "TRIAGE/4-AAP"


OBS_METRIC = Literal["p95_mean", "p95_ci_upper"]
SCENARIO_NAME_MAP = {
    # Map local scenario identifiers to CSV scenario_name values.
    # By default this is empty and is populated by callers (e.g., __main__)
    # when they want to bind local names like "Base_TRIAGE4" to specific
    # scenario_name values in comprehensive_results.csv.
}


def load_results_csv(path: str) -> pd.DataFrame:
    """
    Load comprehensive simulation/prototype results.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Results with normalized scheduler and scenario_name columns.
    """
    df = pd.read_csv(path)
    if "scenario_name" not in df.columns and "scenario_key" in df.columns:
        df["scenario_name"] = df["scenario_key"]
    if "scheduler" not in df.columns:
        raise ValueError("CSV must contain a 'scheduler' column.")
    if "scenario_name" not in df.columns:
        raise ValueError("CSV must contain a 'scenario_name' column.")
    df["scheduler"] = df["scheduler"].astype(str).str.strip()
    df["scenario_name"] = df["scenario_name"].astype(str).str.strip()
    return df


def lookup_observed_delay(
    df: pd.DataFrame, scenario: AlarmScenario, metric: OBS_METRIC = "p95_mean"
) -> float:
    """
    Lookup the observed (empirical) alarm delay for a scenario from results CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Results dataframe loaded via `load_results_csv`.
    scenario : AlarmScenario
        Scenario to match against the CSV (by name and scheduler/baseline).
    metric : OBS_METRIC, optional
        Which latency statistic to use: "p95_mean" (default) or "p95_ci_upper".

    Returns
    -------
    float
        Observed delay in seconds for the selected metric.

    Raises
    ------
    ValueError
        If the baseline is unsupported, the metric is unknown, or the lookup
        does not return exactly one row.
    """
    baseline = scenario.baseline.strip().replace("_", "-").upper()
    if baseline in ("TRIAGE/4", "TRIAGE/4-AAP"):
        scheduler_labels = ["TRIAGE/4", "SEPS"]
    elif baseline == "STRICT":
        scheduler_labels = ["Strict"]
    else:
        raise ValueError(f"Unsupported baseline for lookup: {baseline}")

    original_label = scenario.name.strip()
    mapped_label = SCENARIO_NAME_MAP.get(original_label, original_label)
    labels_to_try = []
    for lbl in (mapped_label, original_label):
        if lbl not in labels_to_try:
            labels_to_try.append(lbl)
    if metric == "p95_mean":
        column = "alarm_p95_latency_mean"
    elif metric == "p95_ci_upper":
        column = "alarm_p95_latency_ci_upper"
    else:
        raise ValueError(f"Unknown metric kind: {metric}")

    if column not in df.columns:
        raise ValueError(f"CSV does not contain required column: {column}")

    matches = []
    for label in labels_to_try:
        for scheduler_label in scheduler_labels:
            mask = (df["scheduler"] == scheduler_label) & (df["scenario_name"] == label)
            subset = df.loc[mask]
            if subset.shape[0] == 1:
                matches.append((label, subset.iloc[0]))

    if len(matches) == 1:
        return float(matches[0][1][column])
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous match for scenario='{original_label}' (labels tried: {labels_to_try}) "
            f"with scheduler labels='{scheduler_labels}'."
        )
    raise ValueError(
        f"Expected exactly one CSV row for scenario='{original_label}' (labels tried: {labels_to_try}), "
        f"scheduler labels='{scheduler_labels}', found 0."
    )


def _format_deadline(deadline: Optional[float]) -> str:
    return f"{deadline:0.4f}" if deadline is not None else "--"


def _format_observed(observed: Optional[float]) -> str:
    return f"{observed:0.4f}" if observed is not None else "--"


def evaluate_scenarios(scenarios: List[AlarmScenario]) -> None:
    """
    Compute and print WCRT values for a list of scenarios.

    For TRIAGE/4 baseline:
    - If b_P and r_P are set, compute the AAP bound.
    - Otherwise, compute the base TRIAGE/4 bound with sigma_A and rho_A.

    For STRICT baseline:
    - Use sigma_A/rho_A as the emergency flow parameters.
    - Use interferers (if provided) as higher-priority classes.
    - The sigma/b and rho/r columns display the totals (emergency + interferers)
      since the strict bound is applied to the aggregate.

    A table is printed with scenario name, baseline, parameters, WCRT, observed
    delay (e.g., maximum or 99.9th percentile from simulation; leave unset if
    no empirical measurement is available), and optional deadline for quick
    inspection.
    """
    header = (
        f"{'Scenario':<18} {'Baseline':<8} {'C [msg/s]':>10}"
        f"{'sigma/b':>12} {'rho/r':>12} {'WCRT [s]':>12}"
        f"{'Observed [s]':>14} {'Deadline [s]':>13}"
    )
    print(header)
    print("-" * len(header))

    for scenario in scenarios:
        baseline = scenario.baseline.strip().replace("_", "-").upper()
        baseline_display = baseline
        if baseline == "TRIAGE/4":
            if scenario.sigma_A is None or scenario.rho_A is None:
                raise ValueError(
                    "sigma_A and rho_A are required for base TRIAGE/4 scenarios."
                )
            burst = scenario.sigma_A
            rate = scenario.rho_A
            wcrt = compute_wcrt_alarm(scenario.C, sigma_A=burst, rho_A=rate)
        elif baseline == "TRIAGE/4-AAP":
            if scenario.b_P is None or scenario.r_P is None:
                raise ValueError("b_P and r_P are required for TRIAGE/4-AAP scenarios.")
            burst = scenario.b_P
            rate = scenario.r_P
            wcrt = compute_wcrt_alarm_aap(scenario.C, b_P=burst, r_P=rate)
        elif baseline == "STRICT":
            if scenario.sigma_A is None or scenario.rho_A is None:
                raise ValueError(
                    "sigma_A and rho_A are required for strict-priority scenarios."
                )
            interferers = scenario.interferers or []
            burst = scenario.sigma_A
            rate = scenario.rho_A
            wcrt = compute_wcrt_strict_emergency(
                scenario.C, sigma_E=burst, rho_E=rate, interferers=interferers
            )
            total_burst = burst + sum(pair[0] for pair in interferers)
            total_rate = rate + sum(pair[1] for pair in interferers)
            burst, rate = total_burst, total_rate
        else:
            raise ValueError("baseline must be one of: 'TRIAGE/4', 'TRIAGE/4-AAP', 'STRICT'.")

        print(
            f"{scenario.name:<18} {baseline_display:<8} {scenario.C:>10.2f}"
            f"{burst:>12.2f} {rate:>12.2f} {wcrt:>12.4f}"
            f"{_format_observed(scenario.observed_delay_s):>14}"
            f"{_format_deadline(scenario.deadline_s):>13}"
        )


def _resolve_results_path() -> pathlib.Path:
    candidates = [
        pathlib.Path("comprehensive_results.csv"),
        pathlib.Path("results/statistical/comprehensive_results.csv"),
        pathlib.Path("results/statistical_extended/comprehensive_results.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "comprehensive_results.csv not found in expected locations. "
        "Place it at one of the default paths or pass an explicit path."
    )


if __name__ == "__main__":
    results_path = _resolve_results_path()
    results_df = load_results_csv(str(results_path))

    # Evaluate three key empirical scenarios side by side:
    # - Alarm Under Burst
    # - Alarm Flood Attack
    # - Legitimate Extreme Emergency
    scenario_groups = [
        "Alarm Under Burst",
        "Alarm Flood Attack",
        "Legitimate Extreme Emergency",
    ]

    for csv_label in scenario_groups:
        print(f"\n=== Scenario: {csv_label} ===")

        # Bind local scenario names to the CSV scenario_name label.
        SCENARIO_NAME_MAP.clear()
        SCENARIO_NAME_MAP.update(
            {
                "Base_TRIAGE4": csv_label,
                "TRIAGE4_AAP": csv_label,
                "Strict_Emergency": csv_label,
            }
        )

        base_seps = AlarmScenario(
            name="Base_TRIAGE4",
            C=20.0,
            sigma_A=10.0,
            rho_A=5.0,
            deadline_s=0.5,
            baseline="TRIAGE/4",
        )
        base_seps.observed_delay_s = lookup_observed_delay(
            results_df, base_seps, metric="p95_mean"
        )

        seps_aap = AlarmScenario(
            name="TRIAGE4_AAP",
            C=20.0,
            b_P=45.0,
            r_P=15.0,
            deadline_s=3.0,
            baseline="TRIAGE/4-AAP",
        )
        seps_aap.observed_delay_s = lookup_observed_delay(
            results_df, seps_aap, metric="p95_mean"
        )

        strict_baseline = AlarmScenario(
            name="Strict_Emergency",
            C=20.0,
            sigma_A=10.0,
            rho_A=5.0,
            interferers=[(5.0, 4.0)],
            deadline_s=1.5,
            baseline="STRICT",
        )
        strict_baseline.observed_delay_s = lookup_observed_delay(
            results_df, strict_baseline, metric="p95_mean"
        )
        # Strict_Emergency totals: sigma_tot=15, rho_tot=9 => WCRT = 0.8225 s at C=20.

        evaluate_scenarios([base_seps, seps_aap, strict_baseline])

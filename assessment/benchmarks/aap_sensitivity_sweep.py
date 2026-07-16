"""
R2.3 — AAP Threshold Sensitivity Sweep (OFAT).

One-factor-at-a-time sweep over the Adaptive Alarm Protection parameters.

Band-global backstop:
  window_duration      (alarm_window_duration)
  lambda_up            (alarm_abnormal_threshold)
  lambda_down          (alarm_deactivation_threshold)
  b_P  (budget)        (alarm_limit_budget)
  burst_capacity       (alarm_burst_capacity)

Per-source limiter:
  source_lambda_up     (alarm_source_abnormal_threshold)
  source_lambda_down   (alarm_source_deactivation_threshold)
  source_budget        (alarm_source_limit_budget)
  source_burst         (alarm_source_burst_capacity)

Each parameter is swept across a range while the others are held at the nominal
default. Scenarios come from assessment.workloads.robustness so this sweep and
the statistical benchmark characterize the same workloads:
  alarm_flood_attack        — adversarial high-rate flood + real emergencies
  alarm_malfunction_surge   — eight malfunctioning sensors + real emergencies
  legit_extreme_emergency   — legitimate control (nothing may be shed)

Stability is judged on legitimate-alarm drops, not the aggregate drop rate. A
high aggregate drop is the intended outcome under flood and malfunction, so it
cannot discriminate a good setting from a bad one there; shedding a genuine
emergency can, and it is checkable in all three scenarios.

Outputs
  results/aap_sensitivity/aap_sensitivity_sweep.csv  — per-setting metrics
  results/aap_sensitivity/aap_sensitivity_sweep.json — structured summary

Usage
  .venv/bin/python -m assessment.benchmarks.aap_sensitivity_sweep
  .venv/bin/python -m assessment.benchmarks.aap_sensitivity_sweep --n-runs 20
  .venv/bin/python -m assessment.benchmarks.aap_sensitivity_sweep --output-dir results/my_sweep
"""

import argparse
import csv
import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from triage4 import TRIAGE4Config, TRIAGE4Scheduler
from assessment.workloads import ROBUSTNESS_SCENARIOS
from assessment.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Nominal (default) AAP parameter values — the sweep baseline
# ---------------------------------------------------------------------------

NOMINAL = {
    # Band-global backstop
    "alarm_window_duration": 10.0,
    "alarm_abnormal_threshold": 5.0,
    "alarm_deactivation_threshold": 4.0,
    "alarm_limit_budget": 15,
    "alarm_burst_capacity": 30,
    # Per-source limiter
    "alarm_source_abnormal_threshold": 0.5,
    "alarm_source_deactivation_threshold": 0.25,
    "alarm_source_limit_budget": 1,
    "alarm_source_burst_capacity": 2,
    # Shared by both layers: how many arrivals a source must produce before it
    # can be judged abnormal. Governs detection speed, and is what keeps
    # low-volume legitimate sources from ever being assessed.
    "alarm_min_observations": 3,
}

# OFAT sweep ranges (each entry: parameter name → list of values to test)
SWEEP_RANGES: Dict[str, List] = {
    # Band-global backstop
    "alarm_window_duration": [2.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
    "alarm_abnormal_threshold": [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
    "alarm_deactivation_threshold": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],  # must stay ≤ λ_up
    "alarm_limit_budget": [3, 5, 8, 10, 15, 20, 30],
    "alarm_burst_capacity": [5, 10, 15, 20, 30, 50, 80],
    # Per-source limiter. The threshold range spans the measured separation
    # between legitimate sources (0.05-0.2 alarms/s) and abnormal ones (2-5/s),
    # so the sweep locates the usable band rather than confirming the nominal.
    "alarm_source_abnormal_threshold": [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "alarm_source_deactivation_threshold": [0.05, 0.1, 0.25, 0.4, 0.5],
    "alarm_source_limit_budget": [1, 2, 3, 5, 8],
    "alarm_source_burst_capacity": [1, 2, 4, 8, 16],
    # Shared. Raising it delays detection, so an abnormal source stays unlimited
    # for longer; the range runs high enough to expose that boundary.
    "alarm_min_observations": [1, 2, 3, 5, 8, 12, 20],
}

# Scenarios: (builder_fn, name, service_rate_override). Shared with the
# statistical benchmark so both describe the same workloads.
SCENARIOS = ROBUSTNESS_SCENARIOS

# Stability thresholds
#
# Judged on legitimate-alarm drops, not the aggregate rate: under flood and
# malfunction a high aggregate drop is the intended outcome, so it cannot
# separate a good setting from a bad one. Shedding a genuine emergency can, and
# it is meaningful in every scenario.
LEGIT_MAX_DROP_RATE = 0.01  # legitimate_alarm_dropped_rate ceiling, all scenarios
LEGIT_MAX_CHURN = 5.0       # activation_churn (activations + deactivations) per run
ATTACK_MAX_CHURN = 20.0     # looser churn bound where protection is meant to engage


@dataclass
class SweepPoint:
    """One OFAT sweep data point."""

    parameter: str          # which AAP parameter was varied
    value: float            # value used for this run
    is_nominal: bool        # True when value == NOMINAL[parameter]
    scenario: str           # scenario key
    n_runs: int
    alarm_dropped_rate_mean: float
    alarm_dropped_rate_ci_lower: float
    alarm_dropped_rate_ci_upper: float
    # Drop attribution: the aggregate rate above cannot tell containment from
    # harm, these can.
    legitimate_alarm_dropped_rate_mean: float
    legitimate_alarm_dropped_rate_ci_lower: float
    legitimate_alarm_dropped_rate_ci_upper: float
    legitimate_alarm_n: float
    abnormal_alarm_dropped_rate_mean: float
    alarm_avg_latency_mean: float
    alarm_avg_latency_ci_lower: float
    alarm_avg_latency_ci_upper: float
    background_wait_mean_mean: float
    background_wait_mean_ci_lower: float
    background_wait_mean_ci_upper: float
    activation_churn_mean: float   # activations + deactivations per run
    activation_churn_ci_lower: float
    activation_churn_ci_upper: float
    stable: bool            # satisfies stability criteria
    expected_drop_floor: Optional[float] = None  # max(0, 1 - limit_rate/offered_rate)


def _config_with_overrides(**overrides) -> TRIAGE4Config:
    """Return a nominal AAP config with the given parameter overrides applied."""
    cfg = TRIAGE4Config(enable_alarm_protection=True)
    for k, v in {**NOMINAL, **overrides}.items():
        setattr(cfg, k, v)
    # Maintain hysteresis invariant: deactivation_threshold <= abnormal_threshold
    if cfg.alarm_deactivation_threshold > cfg.alarm_abnormal_threshold:
        cfg.alarm_deactivation_threshold = cfg.alarm_abnormal_threshold
    # Maintain bucket invariant: burst_capacity >= budget
    # When sweeping burst_capacity below the nominal budget, cap budget to match.
    if cfg.alarm_burst_capacity < cfg.alarm_limit_budget:
        cfg.alarm_limit_budget = cfg.alarm_burst_capacity
    # Same two invariants for the per-source layer.
    if cfg.alarm_source_deactivation_threshold > cfg.alarm_source_abnormal_threshold:
        cfg.alarm_source_deactivation_threshold = cfg.alarm_source_abnormal_threshold
    if cfg.alarm_source_burst_capacity < cfg.alarm_source_limit_budget:
        cfg.alarm_source_limit_budget = cfg.alarm_source_burst_capacity
    return cfg


def _run_scenario_once(
    generator,
    config: TRIAGE4Config,
    seed: int,
) -> Dict[str, float]:
    """Run one seed of a scenario and return the per-run metrics dict."""
    np.random.seed(seed)
    try:
        workload = generator(seed=seed)
    except TypeError:
        workload = generator()

    scheduler = TRIAGE4Scheduler(config, scheduler_seed=seed)
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
        zone_priorities=workload.zone_priorities,
        source_is_legitimate=workload.source_is_legitimate,
    )
    # Churn counts transitions in both layers: a setting that oscillates in
    # either one is unstable.
    activations = float(result.metadata.get("alarm_protection_activations", 0))
    deactivations = float(result.metadata.get("alarm_protection_deactivations", 0))
    activations += float(result.metadata.get("alarm_source_limit_activations", 0))
    deactivations += float(result.metadata.get("alarm_source_limit_deactivations", 0))
    metrics["activation_churn"] = activations + deactivations
    return metrics


def _ci95(values: List[float]) -> Tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) for a list of values."""
    from scipy import stats as scipy_stats

    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    sem = scipy_stats.sem(values)
    lo, hi = scipy_stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return mean, float(lo), float(hi)


def sweep_parameter(
    param_name: str,
    values: List,
    scenario_key: str,
    generator,
    service_rate_override: Optional[float],
    n_runs: int,
    base_seed: int,
    pbar=None,
) -> List[SweepPoint]:
    """OFAT sweep for one parameter × one scenario.

    Args:
        pbar: Optional tqdm bar, advanced once per run.
    """
    points: List[SweepPoint] = []

    for val in values:
        # Build config with this parameter set to val, others at nominal
        cfg = _config_with_overrides(**{param_name: val})
        if service_rate_override is not None:
            cfg.service_rate = service_rate_override

        run_metrics = {
            "alarm_dropped_rate": [],
            "legitimate_alarm_dropped_rate": [],
            "abnormal_alarm_dropped_rate": [],
            "legitimate_alarm_n": [],
            "alarm_avg_latency": [],
            "band_3_wait_mean": [],   # BACKGROUND band waiting time
            "activation_churn": [],
        }

        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            m = _run_scenario_once(generator, cfg, seed)
            run_metrics["alarm_dropped_rate"].append(m.get("alarm_dropped_rate", 0.0))
            run_metrics["legitimate_alarm_dropped_rate"].append(
                m.get("legitimate_alarm_dropped_rate", 0.0)
            )
            run_metrics["abnormal_alarm_dropped_rate"].append(
                m.get("abnormal_alarm_dropped_rate", 0.0)
            )
            run_metrics["legitimate_alarm_n"].append(m.get("legitimate_alarm_n", 0.0))
            run_metrics["alarm_avg_latency"].append(m.get("alarm_avg_latency", 0.0))
            run_metrics["band_3_wait_mean"].append(m.get("band_3_wait_mean", 0.0))
            run_metrics["activation_churn"].append(m.get("activation_churn", 0.0))
            if pbar is not None:
                pbar.update(1)

        dr_mean, dr_lo, dr_hi = _ci95(run_metrics["alarm_dropped_rate"])
        lg_mean, lg_lo, lg_hi = _ci95(run_metrics["legitimate_alarm_dropped_rate"])
        ab_mean, _, _ = _ci95(run_metrics["abnormal_alarm_dropped_rate"])
        lg_n = float(np.mean(run_metrics["legitimate_alarm_n"]))
        al_mean, al_lo, al_hi = _ci95(run_metrics["alarm_avg_latency"])
        bg_mean, bg_lo, bg_hi = _ci95(run_metrics["band_3_wait_mean"])
        ch_mean, ch_lo, ch_hi = _ci95(run_metrics["activation_churn"])

        is_nom = float(val) == float(NOMINAL.get(param_name, val))

        # Stability: a setting must never shed a genuine emergency, in any
        # scenario, and must not oscillate. The legitimate-drop bound is only
        # meaningful where legitimate sources exist (lg_n > 0); the churn bound
        # is looser where protection is meant to engage.
        churn_bound = (
            LEGIT_MAX_CHURN
            if scenario_key == "legit_extreme_emergency"
            else ATTACK_MAX_CHURN
        )
        safe = lg_n == 0 or lg_mean < LEGIT_MAX_DROP_RATE
        stable = safe and ch_mean <= churn_bound

        # Expected drop floor for attack/malfunction scenarios
        expected_floor: Optional[float] = None
        if scenario_key in ("alarm_flood_attack", "alarm_malfunction_surge"):
            limit_rate = cfg.alarm_limit_budget / cfg.alarm_limit_period
            # Offered alarm rate from workload structure. Flood: 20/s attacker
            # + 2/s legitimate. Malfunction: 5 + 7*2 = 19/s faulty + 1/s
            # legitimate. This floor describes the backstop only; per-source
            # limiting sheds ahead of it, so the observed rate exceeds it.
            offered_rate = 22.0 if scenario_key == "alarm_flood_attack" else 20.0
            expected_floor = max(0.0, 1.0 - limit_rate / offered_rate)

        points.append(
            SweepPoint(
                parameter=param_name,
                value=float(val),
                is_nominal=is_nom,
                scenario=scenario_key,
                n_runs=n_runs,
                alarm_dropped_rate_mean=dr_mean,
                alarm_dropped_rate_ci_lower=dr_lo,
                alarm_dropped_rate_ci_upper=dr_hi,
                legitimate_alarm_dropped_rate_mean=lg_mean,
                legitimate_alarm_dropped_rate_ci_lower=lg_lo,
                legitimate_alarm_dropped_rate_ci_upper=lg_hi,
                legitimate_alarm_n=lg_n,
                abnormal_alarm_dropped_rate_mean=ab_mean,
                alarm_avg_latency_mean=al_mean,
                alarm_avg_latency_ci_lower=al_lo,
                alarm_avg_latency_ci_upper=al_hi,
                background_wait_mean_mean=bg_mean,
                background_wait_mean_ci_lower=bg_lo,
                background_wait_mean_ci_upper=bg_hi,
                activation_churn_mean=ch_mean,
                activation_churn_ci_lower=ch_lo,
                activation_churn_ci_upper=ch_hi,
                stable=stable,
                expected_drop_floor=expected_floor,
            )
        )

    return points


def export_csv(points: List[SweepPoint], path: str) -> None:
    """Write sweep results to CSV.

    Columns are derived from SweepPoint rather than listed by hand: a
    hand-maintained list silently omits any field added to the dataclass, which
    is how the drop-attribution columns first went missing from this output.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    field_names = [f.name for f in dataclasses.fields(SweepPoint)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for p in points:
            row = dataclasses.asdict(p)
            row["is_nominal"] = int(row["is_nominal"])
            row["stable"] = int(row["stable"])
            if row["expected_drop_floor"] is None:
                row["expected_drop_floor"] = ""
            writer.writerow(row)
    print(f"✓ Saved CSV: {path}")


def export_json(points: List[SweepPoint], path: str) -> None:
    """Write sweep results to JSON.

    Fields are derived from SweepPoint, with ``*_mean`` / ``*_ci_lower`` /
    ``*_ci_upper`` triples folded back into nested blocks. Listing them by hand
    silently omits anything later added to the dataclass, which is how the
    drop-attribution fields first went missing from this export.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scalar_fields = {"parameter", "value", "is_nominal", "scenario", "n_runs",
                     "stable", "expected_drop_floor"}
    payload = []
    for p in points:
        row = dataclasses.asdict(p)
        entry: Dict[str, object] = {}
        for name, value in row.items():
            if name in scalar_fields:
                entry[name] = value
            elif name.endswith("_mean"):
                base = name[: -len("_mean")]
                block = {"mean": value}
                for suffix in ("ci_lower", "ci_upper"):
                    key = f"{base}_{suffix}"
                    if key in row:
                        block[suffix] = row[key]
                entry[base] = block
            elif not (name.endswith("_ci_lower") or name.endswith("_ci_upper")):
                entry[name] = value
        payload.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Saved JSON: {path}")


def print_stable_range_summary(all_points: List[SweepPoint]) -> None:
    """Print a table of stable operating ranges per parameter per scenario."""
    print("\n" + "=" * 90)
    print("STABLE OPERATING RANGE SUMMARY")
    print("=" * 90)

    by_param_scenario: Dict[Tuple[str, str], List[SweepPoint]] = {}
    for p in all_points:
        key = (p.parameter, p.scenario)
        by_param_scenario.setdefault(key, []).append(p)

    for param in SWEEP_RANGES:
        print(f"\nParameter: {param}  (nominal={NOMINAL[param]})")
        print(f"  {'Scenario':<35} {'Stable values':<40} {'Unstable values'}")
        print(f"  {'-'*35} {'-'*40} {'-'*30}")
        for scenario_key in SCENARIOS:
            points = by_param_scenario.get((param, scenario_key), [])
            stable_vals = sorted(p.value for p in points if p.stable)
            unstable_vals = sorted(p.value for p in points if not p.stable)
            sv = ", ".join(str(v) for v in stable_vals) or "—"
            uv = ", ".join(str(v) for v in unstable_vals) or "—"
            print(f"  {scenario_key:<35} {sv:<40} {uv}")


def main():
    parser = argparse.ArgumentParser(
        description="AAP Threshold Sensitivity Sweep (OFAT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-runs", type=int, default=20,
                        help="Runs per sweep point (default: 20)")
    parser.add_argument("--base-seed", type=int, default=999,
                        help="Base seed (default: 999)")
    parser.add_argument("--output-dir", type=str, default="results/aap_sensitivity",
                        help="Output directory (default: results/aap_sensitivity)")
    parser.add_argument("--scenario", type=str, choices=list(SCENARIOS.keys()),
                        help="Run only one scenario (default: all)")
    parser.add_argument("--parameter", type=str, choices=list(SWEEP_RANGES.keys()),
                        help="Sweep only one parameter (default: all)")
    args = parser.parse_args()

    selected_scenarios = (
        {args.scenario: SCENARIOS[args.scenario]} if args.scenario else SCENARIOS
    )
    selected_params = (
        {args.parameter: SWEEP_RANGES[args.parameter]} if args.parameter else SWEEP_RANGES
    )

    all_points: List[SweepPoint] = []

    for param_name, values in selected_params.items():
        print(f"\n{'='*90}")
        print(f"OFAT SWEEP: {param_name}  values={values}")
        print(f"{'='*90}")

        for scenario_key, (generator, scenario_name, sr_override) in selected_scenarios.items():
            print(f"\n  Scenario: {scenario_name}")
            total_runs = len(values) * args.n_runs
            with tqdm(
                total=total_runs,
                desc=f"  {param_name} × {scenario_key[:20]}",
                unit="run",
                ncols=90,
                ascii="░█",
            ) as pbar:
                points = sweep_parameter(
                    param_name=param_name,
                    values=values,
                    scenario_key=scenario_key,
                    generator=generator,
                    service_rate_override=sr_override,
                    n_runs=args.n_runs,
                    base_seed=args.base_seed,
                    pbar=pbar,
                )

                all_points.extend(points)

                # Print compact per-value table
                print(f"\n  {'Value':>10}  {'Drop%':>8}  {'Alarm Lat':>10}  {'BG Lat':>8}  {'Churn':>7}  {'Stable':>7}  {'ExpFloor':>9}")
                print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*9}")
                for pt in points:
                    ef = f"{pt.expected_drop_floor:.3f}" if pt.expected_drop_floor is not None else "   —"
                    nom_mark = "*" if pt.is_nominal else " "
                    print(
                        f"  {nom_mark}{pt.value:>9}  "
                        f"{pt.alarm_dropped_rate_mean*100:>7.2f}%  "
                        f"{pt.alarm_avg_latency_mean:>10.4f}  "
                        f"{pt.background_wait_mean_mean:>8.4f}  "
                        f"{pt.activation_churn_mean:>7.2f}  "
                        f"{'✓' if pt.stable else '✗':>7}  "
                        f"{ef:>9}"
                    )
                print("  (* = nominal value)")

    # Export results
    csv_path = os.path.join(args.output_dir, "aap_sensitivity_sweep.csv")
    json_path = os.path.join(args.output_dir, "aap_sensitivity_sweep.json")
    export_csv(all_points, csv_path)
    export_json(all_points, json_path)
    print_stable_range_summary(all_points)

    print("\n" + "=" * 90)
    print("AAP sensitivity sweep complete.")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()

"""
R2.2 — Detector-Error Robustness Sweep.

OFAT sweep over alarm misclassification rates to measure TRIAGE/4 behavior
under imperfect upstream alarm detection:
  FPR sweep (FNR=0): false positive rate 0.0 → 0.5, isolated from false negatives
  FNR sweep (FPR=0): false negative rate 0.0 → 0.5, isolated from false positives

Every error-rate point is compared against a zero-error baseline (FPR=0, FNR=0)
with the same workload seed.

Scenarios
  legit_extreme_emergency — FN impact dominates (true alarms mislabeled, demoted)
  alarm_flood_attack      — FP impact dominates (spurious alarms stress AAP)

Key metrics (per run)
  tp_latency              — latency of correctly detected true alarms
  fn_demotion_latency     — latency of true alarms mislabeled as routine (demoted)
  fp_alarm_latency        — latency of false alarms entering the ALARM band
  alarm_protection_activations — AAP activations triggered by FP injection

Outputs
  results/detector_error/detector_error_sweep.csv
  results/detector_error/detector_error_sweep.json

Usage
  .venv/bin/python -m assessment.benchmarks.detector_error_sweep
  .venv/bin/python -m assessment.benchmarks.detector_error_sweep --n-runs 20 --base-seed 999
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from triage4 import TRIAGE4Config, TRIAGE4Scheduler
from assessment.workloads import (
    generate_alarm_flood_attack,
    generate_detector_error_workload,
    generate_legit_extreme_emergency,
)
from assessment.metrics import compute_all_metrics, compute_detector_error_metrics


# FPR and FNR sweep values
FPR_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
FNR_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

# Scenarios: key → (generator_fn, display_name, service_rate_override, enable_aap)
SCENARIOS = {
    "legit_extreme_emergency": (
        generate_legit_extreme_emergency,
        "Legitimate Extreme Emergency",
        None,
        True,   # AAP enabled — check it does not fire on FP injections in legit case
    ),
    "alarm_flood_attack": (
        generate_alarm_flood_attack,
        "Alarm Flood Attack",
        20.0,
        True,   # AAP enabled — check it absorbs FP injection load
    ),
}


@dataclass
class ErrorSweepPoint:
    """One detector-error sweep data point."""

    sweep_type: str        # "fpr_sweep" or "fnr_sweep"
    fpr: float
    fnr: float
    scenario: str
    n_runs: int
    # TP latency (mean ± CI)
    tp_latency_mean: float
    tp_latency_ci_lower: float
    tp_latency_ci_upper: float
    # FN demotion latency (mean ± CI)
    fn_demotion_latency_mean: float
    fn_demotion_latency_ci_lower: float
    fn_demotion_latency_ci_upper: float
    # FP alarm latency (mean ± CI)
    fp_alarm_latency_mean: float
    fp_alarm_latency_ci_lower: float
    fp_alarm_latency_ci_upper: float
    # AAP activations (mean ± CI)
    aap_activations_mean: float
    aap_activations_ci_lower: float
    aap_activations_ci_upper: float
    # Counts (mean across runs)
    n_true_positives_mean: float
    n_false_negatives_mean: float
    n_false_positives_mean: float
    # Zero-error reference TP latency for relative comparison
    tp_latency_baseline: Optional[float] = None


def _ci95(values: List[float]) -> Tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) for a list of values."""
    from scipy import stats as scipy_stats

    n = len(values)
    mean = float(np.mean(values)) if values else 0.0
    if n < 2:
        return mean, mean, mean
    sem = scipy_stats.sem(values)
    if sem == 0.0:
        return mean, mean, mean
    lo, hi = scipy_stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return mean, float(lo), float(hi)


def _run_once(
    generator,
    fpr: float,
    fnr: float,
    config: TRIAGE4Config,
    workload_seed: int,
    scheduler_seed: int,
) -> Dict[str, float]:
    """Run one seed: inject detector errors, schedule, return metrics dict."""
    # Generate base workload with ground truth
    try:
        base = generator(seed=workload_seed)
    except TypeError:
        base = generator()

    # Inject detector errors (zero-error when fpr=fnr=0)
    noisy = generate_detector_error_workload(
        base,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        seed=workload_seed,  # same seed → deterministic noise injection
    )

    scheduler = TRIAGE4Scheduler(config, scheduler_seed=scheduler_seed)
    result = scheduler.schedule(
        arrival_times=noisy.arrival_times,
        device_ids=noisy.device_ids,
        zone_priorities=noisy.zone_priorities,
        is_alarm=noisy.is_alarm,
    )

    # Standard metrics (uses detected labels)
    metrics = compute_all_metrics(
        result=result,
        arrival_times=noisy.arrival_times,
        device_ids=noisy.device_ids,
        is_alarm=noisy.is_alarm,
        zone_priorities=noisy.zone_priorities,
        ground_truth_is_alarm=noisy.ground_truth_is_alarm,
    )

    # Detector-error metrics (TP / FN / FP latency)
    error_metrics = compute_detector_error_metrics(
        result=result,
        is_alarm=list(noisy.is_alarm),
        ground_truth_is_alarm=noisy.ground_truth_is_alarm,
    )
    metrics.update(error_metrics)
    return metrics


def sweep_error_rate(
    sweep_type: str,
    scenario_key: str,
    generator,
    service_rate_override: Optional[float],
    enable_aap: bool,
    n_runs: int,
    base_seed: int,
) -> List[ErrorSweepPoint]:
    """Run FPR or FNR sweep for one scenario."""
    config = TRIAGE4Config(enable_alarm_protection=enable_aap)
    if service_rate_override is not None:
        config.service_rate = service_rate_override

    rate_values = FPR_VALUES if sweep_type == "fpr_sweep" else FNR_VALUES
    points: List[ErrorSweepPoint] = []

    # Pre-compute zero-error baseline TP latency for relative reporting
    baseline_tp_latencies = []
    for run_idx in range(n_runs):
        m = _run_once(generator, 0.0, 0.0, config,
                      workload_seed=base_seed + run_idx,
                      scheduler_seed=base_seed + run_idx)
        baseline_tp_latencies.append(m.get("tp_latency", float("nan")))
    # Use nanmean to handle cases where there are no TPs in a run
    valid_baseline = [v for v in baseline_tp_latencies if not np.isnan(v)]
    baseline_tp_mean = float(np.mean(valid_baseline)) if valid_baseline else float("nan")

    for rate in rate_values:
        fpr = rate if sweep_type == "fpr_sweep" else 0.0
        fnr = rate if sweep_type == "fnr_sweep" else 0.0

        run_tp, run_fn, run_fp, run_aap, run_n_tp, run_n_fn, run_n_fp = [], [], [], [], [], [], []

        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            m = _run_once(generator, fpr, fnr, config,
                          workload_seed=seed, scheduler_seed=seed)

            run_tp.append(m.get("tp_latency", float("nan")))
            run_fn.append(m.get("fn_demotion_latency", float("nan")))
            run_fp.append(m.get("fp_alarm_latency", float("nan")))
            run_aap.append(m.get("alarm_protection_activations", 0.0))
            run_n_tp.append(m.get("n_true_positives", 0.0))
            run_n_fn.append(m.get("n_false_negatives", 0.0))
            run_n_fp.append(m.get("n_false_positives", 0.0))

        # Filter NaN before CI computation
        def _nanlist(lst):
            return [v for v in lst if not np.isnan(v)]

        tp_mean, tp_lo, tp_hi = _ci95(_nanlist(run_tp)) if _nanlist(run_tp) else (float("nan"),) * 3
        fn_mean, fn_lo, fn_hi = _ci95(_nanlist(run_fn)) if _nanlist(run_fn) else (float("nan"),) * 3
        fp_mean, fp_lo, fp_hi = _ci95(_nanlist(run_fp)) if _nanlist(run_fp) else (float("nan"),) * 3
        aap_mean, aap_lo, aap_hi = _ci95(run_aap)
        n_tp_mean = float(np.mean(run_n_tp))
        n_fn_mean = float(np.mean(run_n_fn))
        n_fp_mean = float(np.mean(run_n_fp))

        points.append(ErrorSweepPoint(
            sweep_type=sweep_type,
            fpr=fpr,
            fnr=fnr,
            scenario=scenario_key,
            n_runs=n_runs,
            tp_latency_mean=tp_mean,
            tp_latency_ci_lower=tp_lo,
            tp_latency_ci_upper=tp_hi,
            fn_demotion_latency_mean=fn_mean,
            fn_demotion_latency_ci_lower=fn_lo,
            fn_demotion_latency_ci_upper=fn_hi,
            fp_alarm_latency_mean=fp_mean,
            fp_alarm_latency_ci_lower=fp_lo,
            fp_alarm_latency_ci_upper=fp_hi,
            aap_activations_mean=aap_mean,
            aap_activations_ci_lower=aap_lo,
            aap_activations_ci_upper=aap_hi,
            n_true_positives_mean=n_tp_mean,
            n_false_negatives_mean=n_fn_mean,
            n_false_positives_mean=n_fp_mean,
            tp_latency_baseline=baseline_tp_mean,
        ))

    return points


def export_csv(points: List[ErrorSweepPoint], path: str) -> None:
    """Write sweep results to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "sweep_type", "fpr", "fnr", "scenario", "n_runs",
        "tp_latency_mean", "tp_latency_ci_lower", "tp_latency_ci_upper",
        "fn_demotion_latency_mean", "fn_demotion_latency_ci_lower", "fn_demotion_latency_ci_upper",
        "fp_alarm_latency_mean", "fp_alarm_latency_ci_lower", "fp_alarm_latency_ci_upper",
        "aap_activations_mean", "aap_activations_ci_lower", "aap_activations_ci_upper",
        "n_true_positives_mean", "n_false_negatives_mean", "n_false_positives_mean",
        "tp_latency_baseline",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for p in points:
            row = [
                p.sweep_type, p.fpr, p.fnr, p.scenario, p.n_runs,
                p.tp_latency_mean, p.tp_latency_ci_lower, p.tp_latency_ci_upper,
                p.fn_demotion_latency_mean, p.fn_demotion_latency_ci_lower, p.fn_demotion_latency_ci_upper,
                p.fp_alarm_latency_mean, p.fp_alarm_latency_ci_lower, p.fp_alarm_latency_ci_upper,
                p.aap_activations_mean, p.aap_activations_ci_lower, p.aap_activations_ci_upper,
                p.n_true_positives_mean, p.n_false_negatives_mean, p.n_false_positives_mean,
                p.tp_latency_baseline if p.tp_latency_baseline is not None else "",
            ]
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"✓ Saved CSV: {path}")


def export_json(points: List[ErrorSweepPoint], path: str) -> None:
    """Write sweep results to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _val(v):
        return None if (isinstance(v, float) and np.isnan(v)) else v

    payload = [
        {
            "sweep_type": p.sweep_type,
            "fpr": p.fpr,
            "fnr": p.fnr,
            "scenario": p.scenario,
            "n_runs": p.n_runs,
            "tp_latency": {"mean": _val(p.tp_latency_mean), "ci_lower": _val(p.tp_latency_ci_lower), "ci_upper": _val(p.tp_latency_ci_upper)},
            "fn_demotion_latency": {"mean": _val(p.fn_demotion_latency_mean), "ci_lower": _val(p.fn_demotion_latency_ci_lower), "ci_upper": _val(p.fn_demotion_latency_ci_upper)},
            "fp_alarm_latency": {"mean": _val(p.fp_alarm_latency_mean), "ci_lower": _val(p.fp_alarm_latency_ci_lower), "ci_upper": _val(p.fp_alarm_latency_ci_upper)},
            "aap_activations": {"mean": p.aap_activations_mean, "ci_lower": p.aap_activations_ci_lower, "ci_upper": p.aap_activations_ci_upper},
            "counts": {"tp": p.n_true_positives_mean, "fn": p.n_false_negatives_mean, "fp": p.n_false_positives_mean},
            "tp_latency_baseline": _val(p.tp_latency_baseline),
        }
        for p in points
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Saved JSON: {path}")


def print_summary(all_points: List[ErrorSweepPoint]) -> None:
    """Print compact tables for each sweep type × scenario."""
    for sweep_type in ("fpr_sweep", "fnr_sweep"):
        label = "FPR" if sweep_type == "fpr_sweep" else "FNR"
        print(f"\n{'='*100}")
        print(f"DETECTOR-ERROR SWEEP: {label}")
        print(f"{'='*100}")
        for scenario_key in SCENARIOS:
            pts = [p for p in all_points if p.sweep_type == sweep_type and p.scenario == scenario_key]
            if not pts:
                continue
            print(f"\n  Scenario: {scenario_key}")
            print(f"  {'Rate':>6}  {'TP lat':>9}  {'FN demot':>10}  {'FP lat':>9}  {'AAP acts':>9}  {'TP':>6}  {'FN':>6}  {'FP':>6}")
            print(f"  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*6}  {'-'*6}  {'-'*6}")
            for pt in pts:
                rate = pt.fpr if sweep_type == "fpr_sweep" else pt.fnr
                tp = f"{pt.tp_latency_mean:.4f}" if not np.isnan(pt.tp_latency_mean) else "   N/A"
                fn = f"{pt.fn_demotion_latency_mean:.4f}" if not np.isnan(pt.fn_demotion_latency_mean) else "     N/A"
                fp = f"{pt.fp_alarm_latency_mean:.4f}" if not np.isnan(pt.fp_alarm_latency_mean) else "   N/A"
                print(
                    f"  {rate:>6.2f}  {tp:>9}  {fn:>10}  {fp:>9}  "
                    f"{pt.aap_activations_mean:>9.2f}  "
                    f"{pt.n_true_positives_mean:>6.1f}  "
                    f"{pt.n_false_negatives_mean:>6.1f}  "
                    f"{pt.n_false_positives_mean:>6.1f}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Detector-Error Robustness Sweep (R2.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-runs", type=int, default=20,
                        help="Runs per sweep point (default: 20)")
    parser.add_argument("--base-seed", type=int, default=999,
                        help="Base seed (default: 999)")
    parser.add_argument("--output-dir", type=str, default="results/detector_error",
                        help="Output directory (default: results/detector_error)")
    args = parser.parse_args()

    all_points: List[ErrorSweepPoint] = []

    for sweep_type in ("fpr_sweep", "fnr_sweep"):
        label = "FPR" if sweep_type == "fpr_sweep" else "FNR"
        print(f"\n{'='*70}")
        print(f"Running {label} sweep  ({len(FPR_VALUES)} rates × {len(SCENARIOS)} scenarios × {args.n_runs} runs)")
        print(f"{'='*70}")

        for scenario_key, (generator, name, sr_override, enable_aap) in SCENARIOS.items():
            print(f"  [{label}] {name} ... ", end="", flush=True)
            pts = sweep_error_rate(
                sweep_type=sweep_type,
                scenario_key=scenario_key,
                generator=generator,
                service_rate_override=sr_override,
                enable_aap=enable_aap,
                n_runs=args.n_runs,
                base_seed=args.base_seed,
            )
            all_points.extend(pts)
            print("done")

    print_summary(all_points)

    csv_path = os.path.join(args.output_dir, "detector_error_sweep.csv")
    json_path = os.path.join(args.output_dir, "detector_error_sweep.json")
    export_csv(all_points, csv_path)
    export_json(all_points, json_path)

    print(f"\n{'='*70}")
    print("Detector-error sweep complete.")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

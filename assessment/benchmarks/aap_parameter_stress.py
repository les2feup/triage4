"""Deterministic stress analysis for Adaptive Alarm Protection parameters.

Each AAP parameter is varied one factor at a time over a constraint-compatible
candidate set derived from the AAP traffic and response assumptions. Component
probes measure exact
detector and token-bucket boundaries, then three deterministic alarm traces
exercise concentrated abnormal traffic, distributed abnormal traffic, and a
legitimate-only control. The output reports every tested value and compares
legitimate-drop findings with the delivered scenario-based sensitivity sweep.

Usage
  .venv/bin/python -m assessment.benchmarks.aap_parameter_stress
  .venv/bin/python -m assessment.benchmarks.aap_parameter_stress \
      --output-dir results/aap_parameter_stress \
      --current-sweep-json results/aap_sensitivity_delivered/aap_sensitivity_sweep.json
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from triage4 import (
    AdaptiveTokenBucket,
    AlarmRateMonitor,
    TRIAGE4Config,
    TRIAGE4Scheduler,
)

from assessment.benchmarks.aap_sensitivity_sweep import NOMINAL


# Candidate profiles preserve AAP invariants while bracketing operational
# boundaries and selected diagnostic values. The union contains 55 OFAT points;
# it is not a Cartesian product.
STRESS_RANGES: Dict[str, List[float]] = {
    "alarm_window_duration": [2.0, 5.0, 10.0, 15.0, 20.0],
    "alarm_abnormal_threshold": [4.0, 5.0, 6.0, 8.0, 12.0, 20.0],
    "alarm_deactivation_threshold": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    "alarm_limit_budget": [5, 8, 10, 12, 15, 20, 30],
    "alarm_burst_capacity": [15, 20, 30, 45, 60],
    "alarm_source_abnormal_threshold": [
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
    ],
    "alarm_source_deactivation_threshold": [0.05, 0.1, 0.25, 0.4, 0.5],
    "alarm_source_limit_budget": [1, 2],
    "alarm_source_burst_capacity": [1, 2, 4, 8],
    "alarm_min_observations": [1, 2, 3, 5, 8, 12, 20],
}

# These valid settings intentionally cross a practical protection boundary.
# They diagnose the resulting response but are not candidate operating settings.
DIAGNOSTIC_VALUES = {
    "alarm_limit_budget": {5.0, 8.0},
    "alarm_source_abnormal_threshold": {0.25, 3.0, 5.0},
    "alarm_source_limit_budget": {2.0},
    "alarm_min_observations": {20.0},
}


METHODOLOGY = (
    "Candidate values were selected from the observed workload envelope and "
    "the structural constraints of AAP. Activation thresholds bracketed the "
    "separation between legitimate and abnormal alarm rates, window settings "
    "covered response times up to approximately five seconds under the nominal "
    "source threshold, and selected controls crossed that target. Limiting "
    "budgets were centred on the residual traffic expected "
    "from concurrently limited sources. Each of the ten parameters was varied "
    "independently while the others remained nominal; every profile included its "
    "nominal value, values around the expected operating boundary, and, where "
    "informative, a deliberate diagnostic value. All 55 configurations satisfied "
    "the threshold and bucket-capacity invariants. Exact component probes and "
    "three deterministic alarm traces then measured the mechanism response, "
    "false activation of a recurrent 0.3 alarms/s legitimate source, and "
    "legitimate-alarm loss. Loss values were compared against the delivered "
    "stochastic R1--R3 sweep."
)


@dataclass(frozen=True)
class RateMonitorProfile:
    """Observed count boundaries of one alarm-rate monitor."""

    activation_count: int
    recovery_count: int


@dataclass(frozen=True)
class BucketProfile:
    """Observed admission capacities of one adaptive token bucket."""

    initial_allowance: int
    stored_burst: int


@dataclass(frozen=True)
class ParameterPoint:
    """Direct component responses for one OFAT parameter value."""

    parameter: str
    value: float
    is_nominal: bool
    design_role: str
    feasible: bool
    infeasible_reason: str = ""
    global_activation_count: Optional[int] = None
    global_recovery_count: Optional[int] = None
    source_activation_count: Optional[int] = None
    source_recovery_count: Optional[int] = None
    global_initial_allowance: Optional[int] = None
    global_stored_burst: Optional[int] = None
    source_initial_allowance: Optional[int] = None
    source_stored_burst: Optional[int] = None


@dataclass(frozen=True)
class StressTrace:
    """Minimal labelled alarm trace for end-to-end AAP stress."""

    name: str
    arrival_times: Tuple[float, ...]
    device_ids: Tuple[str, ...]
    source_is_legitimate: Tuple[bool, ...]


@dataclass(frozen=True)
class TraceOutcome:
    """Safety and containment response for one parameter value and trace."""

    parameter: str
    value: float
    trace: str
    legitimate_alarm_n: int
    legitimate_alarm_dropped: int
    abnormal_alarm_n: int
    abnormal_alarm_dropped: int
    abnormal_alarm_dropped_rate: float
    global_activations: int
    global_deactivations: int
    source_activations: int
    source_deactivations: int


@dataclass(frozen=True)
class ParameterSummary:
    """Concise comparison between deterministic and scenario-based findings."""

    parameter: str
    tested_values: Tuple[float, ...]
    diagnostic_values: Tuple[float, ...]
    infeasible_values: Tuple[float, ...]
    direct_response: str
    legitimate_source_activation_values: Tuple[float, ...]
    stress_drop_values: Tuple[float, ...]
    current_unstable_values: Tuple[float, ...]
    overlap_values: Tuple[float, ...]
    stress_only_values: Tuple[float, ...]
    current_only_values: Tuple[float, ...]


def measure_rate_monitor(
    window_duration: float,
    abnormal_threshold: float,
    deactivation_threshold: float,
    min_observations: int,
) -> RateMonitorProfile:
    """Measure activation and recovery count boundaries with simultaneous arrivals."""
    activation_monitor = AlarmRateMonitor(
        window_duration=window_duration,
        abnormal_threshold=abnormal_threshold,
        deactivation_threshold=deactivation_threshold,
        min_observations=min_observations,
    )
    search_limit = max(
        min_observations,
        math.ceil(window_duration * abnormal_threshold),
    ) + 2
    activation_count: Optional[int] = None
    for count in range(1, search_limit + 1):
        activation_monitor.record_arrival(0.0, "stress_source")
        if activation_monitor.is_abnormal(0.0):
            activation_count = count
            break
    if activation_count is None:
        raise RuntimeError("activation boundary was not reached")

    recovery_count = 0
    recovery_limit = math.ceil(window_duration * deactivation_threshold) + 2
    for count in range(1, recovery_limit + 1):
        recovery_monitor = AlarmRateMonitor(
            window_duration=window_duration,
            abnormal_threshold=abnormal_threshold,
            deactivation_threshold=deactivation_threshold,
            min_observations=min_observations,
        )
        for _ in range(count):
            recovery_monitor.record_arrival(0.0, "stress_source")
        if recovery_monitor.is_recovered(0.0):
            recovery_count = count

    return RateMonitorProfile(
        activation_count=activation_count,
        recovery_count=recovery_count,
    )


def _consume_until_denied(bucket: AdaptiveTokenBucket, current_time: float) -> int:
    """Return the number of unit requests admitted at one timestamp."""
    admitted = 0
    while bucket.consume(current_time):
        admitted += 1
    return admitted


def measure_bucket(
    budget: int,
    period: float,
    burst_capacity: int,
) -> BucketProfile:
    """Measure first-activation allowance and fully accumulated burst capacity."""
    initial_bucket = AdaptiveTokenBucket(
        budget=budget,
        period=period,
        burst_capacity=burst_capacity,
    )
    initial_bucket.activate(0.0)
    initial_allowance = _consume_until_denied(initial_bucket, 0.0)

    stored_bucket = AdaptiveTokenBucket(
        budget=budget,
        period=period,
        burst_capacity=burst_capacity,
    )
    stored_bucket.activate(0.0)
    idle_periods = math.ceil(burst_capacity / budget) + 1
    stored_burst = _consume_until_denied(stored_bucket, idle_periods * period)

    return BucketProfile(
        initial_allowance=initial_allowance,
        stored_burst=stored_burst,
    )


def _config_for_value(parameter: str, value: float) -> TRIAGE4Config:
    """Build one valid OFAT configuration without changing any paired parameter."""
    values = dict(NOMINAL)
    values[parameter] = value
    if parameter == "alarm_min_observations":
        values[parameter] = int(value)
    return TRIAGE4Config(
        enable_alarm_protection=True,
        service_rate=100.0,
        high_token_budget=100,
        standard_token_budget=100,
        background_token_budget=100,
        **values,
    )


def _point_from_config(
    parameter: str,
    value: float,
    config: TRIAGE4Config,
) -> ParameterPoint:
    """Measure all direct AAP responses for one feasible configuration."""
    global_monitor = measure_rate_monitor(
        window_duration=config.alarm_window_duration,
        abnormal_threshold=config.alarm_abnormal_threshold,
        deactivation_threshold=config.alarm_deactivation_threshold,
        min_observations=config.alarm_min_observations,
    )
    source_monitor = measure_rate_monitor(
        window_duration=config.alarm_window_duration,
        abnormal_threshold=config.alarm_source_abnormal_threshold,
        deactivation_threshold=config.alarm_source_deactivation_threshold,
        min_observations=config.alarm_min_observations,
    )
    global_bucket = measure_bucket(
        budget=config.alarm_limit_budget,
        period=config.alarm_limit_period,
        burst_capacity=config.alarm_burst_capacity,
    )
    source_bucket = measure_bucket(
        budget=config.alarm_source_limit_budget,
        period=config.alarm_source_limit_period,
        burst_capacity=config.alarm_source_burst_capacity,
    )
    return ParameterPoint(
        parameter=parameter,
        value=float(value),
        is_nominal=float(value) == float(NOMINAL[parameter]),
        design_role=(
            "diagnostic"
            if float(value) in DIAGNOSTIC_VALUES.get(parameter, set())
            else "operational"
        ),
        feasible=True,
        global_activation_count=global_monitor.activation_count,
        global_recovery_count=global_monitor.recovery_count,
        source_activation_count=source_monitor.activation_count,
        source_recovery_count=source_monitor.recovery_count,
        global_initial_allowance=global_bucket.initial_allowance,
        global_stored_burst=global_bucket.stored_burst,
        source_initial_allowance=source_bucket.initial_allowance,
        source_stored_burst=source_bucket.stored_burst,
    )


def build_parameter_points() -> List[ParameterPoint]:
    """Measure every value in the constraint-compatible OFAT design."""
    points: List[ParameterPoint] = []
    for parameter, values in STRESS_RANGES.items():
        for value in values:
            try:
                config = _config_for_value(parameter, value)
            except ValueError as exc:
                points.append(
                    ParameterPoint(
                        parameter=parameter,
                        value=float(value),
                        is_nominal=float(value) == float(NOMINAL[parameter]),
                        design_role="invalid",
                        feasible=False,
                        infeasible_reason=str(exc),
                    )
                )
                continue
            points.append(_point_from_config(parameter, value, config))
    return points


def _append_periodic(
    events: List[Tuple[float, str, bool]],
    source: str,
    rate: float,
    duration: float,
    legitimate: bool,
) -> None:
    """Append evenly spaced events centered within their inter-arrival slots."""
    count = int(rate * duration)
    if count <= 0:
        return
    for index in range(count):
        timestamp = (index + 0.5) * duration / count
        events.append((timestamp, source, legitimate))


def _append_legitimate_probes(
    events: List[Tuple[float, str, bool]],
    count: int,
    duration: float,
) -> None:
    """Append distinct legitimate sources evenly across the observation period."""
    for index in range(count):
        timestamp = (index + 1) * duration / (count + 1)
        events.append((timestamp, f"legitimate_{index}", True))


def _trace(name: str, events: Iterable[Tuple[float, str, bool]]) -> StressTrace:
    """Return a stable time-ordered trace from labelled events."""
    ordered = sorted(events, key=lambda event: (event[0], event[1]))
    return StressTrace(
        name=name,
        arrival_times=tuple(event[0] for event in ordered),
        device_ids=tuple(event[1] for event in ordered),
        source_is_legitimate=tuple(event[2] for event in ordered),
    )


def build_stress_traces() -> List[StressTrace]:
    """Build three deterministic traces that isolate AAP protection mechanisms."""
    concentrated: List[Tuple[float, str, bool]] = []
    _append_periodic(concentrated, "abnormal_single", 20.0, 10.0, False)
    _append_legitimate_probes(concentrated, count=20, duration=10.0)

    distributed: List[Tuple[float, str, bool]] = []
    _append_periodic(distributed, "abnormal_heavy", 5.0, 20.0, False)
    for index in range(7):
        _append_periodic(
            distributed,
            f"abnormal_light_{index}",
            2.0,
            20.0,
            False,
        )
    _append_legitimate_probes(distributed, count=20, duration=20.0)

    legitimate: List[Tuple[float, str, bool]] = []
    _append_periodic(
        legitimate,
        "legitimate_boundary",
        0.3,
        30.0,
        True,
    )
    for index in range(9):
        _append_periodic(
            legitimate,
            f"legitimate_zone_{index}",
            0.1,
            30.0,
            True,
        )

    return [
        _trace("single_source_overload", concentrated),
        _trace("multi_source_overload", distributed),
        _trace("legitimate_control", legitimate),
    ]


def run_stress_trace(
    parameter: str,
    value: float,
    config: TRIAGE4Config,
    trace: StressTrace,
    scheduler_seed: int,
) -> TraceOutcome:
    """Replay one deterministic trace through TRIAGE/4 and attribute its drops."""
    scheduler = TRIAGE4Scheduler(config, scheduler_seed=scheduler_seed)
    result = scheduler.schedule(
        arrival_times=list(trace.arrival_times),
        device_ids=list(trace.device_ids),
        zone_priorities=[5] * len(trace.arrival_times),
        is_alarm=[True] * len(trace.arrival_times),
    )
    dropped = [float(e2e) == 0.0 for e2e in result.e2e_times]
    legitimate_indices = [
        index
        for index, legitimate in enumerate(trace.source_is_legitimate)
        if legitimate
    ]
    abnormal_indices = [
        index
        for index, legitimate in enumerate(trace.source_is_legitimate)
        if not legitimate
    ]
    legitimate_dropped = sum(dropped[index] for index in legitimate_indices)
    abnormal_dropped = sum(dropped[index] for index in abnormal_indices)
    metadata = result.metadata
    return TraceOutcome(
        parameter=parameter,
        value=float(value),
        trace=trace.name,
        legitimate_alarm_n=len(legitimate_indices),
        legitimate_alarm_dropped=legitimate_dropped,
        abnormal_alarm_n=len(abnormal_indices),
        abnormal_alarm_dropped=abnormal_dropped,
        abnormal_alarm_dropped_rate=(
            abnormal_dropped / len(abnormal_indices) if abnormal_indices else 0.0
        ),
        global_activations=int(metadata.get("alarm_protection_activations", 0)),
        global_deactivations=int(metadata.get("alarm_protection_deactivations", 0)),
        source_activations=int(metadata.get("alarm_source_limit_activations", 0)),
        source_deactivations=int(metadata.get("alarm_source_limit_deactivations", 0)),
    )


def compare_current_sweep(path: Path) -> Dict[str, List[float]]:
    """Return unique unstable values from the scenario-based sweep."""
    with path.open(encoding="utf-8") as handle:
        points = json.load(handle)
    unstable: Dict[str, set[float]] = {}
    for point in points:
        if bool(point["stable"]):
            continue
        unstable.setdefault(point["parameter"], set()).add(float(point["value"]))
    return {
        parameter: sorted(values)
        for parameter, values in sorted(unstable.items())
    }


def _range(values: Sequence[int]) -> str:
    """Format a compact integer response range."""
    unique = sorted(set(values))
    if not unique:
        return "--"
    if len(unique) == 1:
        return str(unique[0])
    return f"{unique[0]}--{unique[-1]}"


def _direct_response(parameter: str, points: Sequence[ParameterPoint]) -> str:
    """Describe the direct mechanism affected by one parameter."""
    feasible = [point for point in points if point.feasible]
    if parameter in {"alarm_window_duration", "alarm_min_observations"}:
        global_counts = [point.global_activation_count for point in feasible]
        source_counts = [point.source_activation_count for point in feasible]
        return f"activation count P={_range(global_counts)}; S={_range(source_counts)}"
    if parameter == "alarm_abnormal_threshold":
        counts = [point.global_activation_count for point in feasible]
        return f"global activation count={_range(counts)}"
    if parameter == "alarm_deactivation_threshold":
        counts = [point.global_recovery_count for point in feasible]
        return f"global recovery count={_range(counts)}"
    if parameter == "alarm_limit_budget":
        values = [point.global_initial_allowance for point in feasible]
        return f"global first allowance={_range(values)}"
    if parameter == "alarm_burst_capacity":
        values = [point.global_stored_burst for point in feasible]
        return f"global stored burst={_range(values)}"
    if parameter == "alarm_source_abnormal_threshold":
        counts = [point.source_activation_count for point in feasible]
        return f"source activation count={_range(counts)}"
    if parameter == "alarm_source_deactivation_threshold":
        counts = [point.source_recovery_count for point in feasible]
        return f"source recovery count={_range(counts)}"
    if parameter == "alarm_source_limit_budget":
        values = [point.source_initial_allowance for point in feasible]
        return f"source first allowance={_range(values)}"
    if parameter == "alarm_source_burst_capacity":
        values = [point.source_stored_burst for point in feasible]
        return f"source stored burst={_range(values)}"
    raise KeyError(f"unknown AAP parameter: {parameter}")


def summarize_parameters(
    points: Sequence[ParameterPoint],
    outcomes: Sequence[TraceOutcome],
    current_unstable: Dict[str, List[float]],
) -> List[ParameterSummary]:
    """Build complete per-parameter summaries without interval assumptions."""
    summaries: List[ParameterSummary] = []
    for parameter in STRESS_RANGES:
        parameter_points = [point for point in points if point.parameter == parameter]
        stress_values = {
            outcome.value
            for outcome in outcomes
            if outcome.parameter == parameter and outcome.legitimate_alarm_dropped > 0
        }
        legitimate_activation_values = {
            outcome.value
            for outcome in outcomes
            if (
                outcome.parameter == parameter
                and outcome.trace == "legitimate_control"
                and outcome.source_activations > 0
            )
        }
        current_values = set(current_unstable.get(parameter, []))
        summaries.append(
            ParameterSummary(
                parameter=parameter,
                tested_values=tuple(point.value for point in parameter_points),
                diagnostic_values=tuple(
                    point.value
                    for point in parameter_points
                    if point.design_role == "diagnostic"
                ),
                infeasible_values=tuple(
                    point.value for point in parameter_points if not point.feasible
                ),
                direct_response=_direct_response(parameter, parameter_points),
                legitimate_source_activation_values=tuple(
                    sorted(legitimate_activation_values)
                ),
                stress_drop_values=tuple(sorted(stress_values)),
                current_unstable_values=tuple(sorted(current_values)),
                overlap_values=tuple(sorted(stress_values & current_values)),
                stress_only_values=tuple(sorted(stress_values - current_values)),
                current_only_values=tuple(sorted(current_values - stress_values)),
            )
        )
    return summaries


def run_analysis(
    current_sweep_path: Path,
    scheduler_seed: int,
) -> Tuple[List[ParameterPoint], List[TraceOutcome], List[ParameterSummary]]:
    """Run direct probes, deterministic traces, and scenario comparison."""
    points = build_parameter_points()
    traces = build_stress_traces()
    outcomes: List[TraceOutcome] = []
    for point in points:
        if not point.feasible:
            continue
        config = _config_for_value(point.parameter, point.value)
        for trace in traces:
            outcomes.append(
                run_stress_trace(
                    parameter=point.parameter,
                    value=point.value,
                    config=config,
                    trace=trace,
                    scheduler_seed=scheduler_seed,
                )
            )
    current_unstable = compare_current_sweep(current_sweep_path)
    summaries = summarize_parameters(points, outcomes, current_unstable)
    return points, outcomes, summaries


def _csv_value(value: object) -> object:
    """Serialize tuple fields compactly for CSV."""
    if isinstance(value, tuple):
        return ";".join(str(item) for item in value)
    return value


def _write_csv(records: Sequence[object], path: Path) -> None:
    """Write homogeneous dataclass records to CSV."""
    if not records:
        raise ValueError("cannot write an empty record collection")
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in dataclasses.fields(records[0])]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: _csv_value(value)
                    for key, value in dataclasses.asdict(record).items()
                }
            )


def write_outputs(
    output_dir: Path,
    points: Sequence[ParameterPoint],
    outcomes: Sequence[TraceOutcome],
    summaries: Sequence[ParameterSummary],
    current_sweep_path: Path,
    scheduler_seed: int,
) -> None:
    """Write complete machine-readable results and a concise summary table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(points, output_dir / "aap_parameter_stress_points.csv")
    _write_csv(outcomes, output_dir / "aap_parameter_stress_outcomes.csv")
    _write_csv(summaries, output_dir / "aap_parameter_stress_summary.csv")
    payload = {
        "methodology": METHODOLOGY,
        "scheduler_seed": scheduler_seed,
        "current_sweep_json": str(current_sweep_path),
        "points": [dataclasses.asdict(point) for point in points],
        "outcomes": [dataclasses.asdict(outcome) for outcome in outcomes],
        "summaries": [dataclasses.asdict(summary) for summary in summaries],
    }
    with (output_dir / "aap_parameter_stress.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)


def _values(values: Sequence[float]) -> str:
    """Format a concise parameter-value list."""
    return ",".join(f"{value:g}" for value in values) or "--"


def print_summary(summaries: Sequence[ParameterSummary]) -> None:
    """Print the complete comparison without per-event diagnostic noise."""
    print("\nAAP PARAMETER STRESS SUMMARY")
    print("=" * 139)
    print(
        f"{'Parameter':<37} {'Infeasible':<14} {'Legit src acts':<18} "
        f"{'Stress drops':<18} "
        f"{'Current unstable':<18} Direct response"
    )
    print("-" * 139)
    for summary in summaries:
        print(
            f"{summary.parameter:<37} "
            f"{_values(summary.infeasible_values):<14} "
            f"{_values(summary.legitimate_source_activation_values):<18} "
            f"{_values(summary.stress_drop_values):<18} "
            f"{_values(summary.current_unstable_values):<18} "
            f"{summary.direct_response}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aap_parameter_stress"),
    )
    parser.add_argument(
        "--current-sweep-json",
        type=Path,
        default=Path(
            "results/aap_sensitivity_delivered/aap_sensitivity_sweep.json"
        ),
    )
    parser.add_argument("--scheduler-seed", type=int, default=999)
    args = parser.parse_args()

    points, outcomes, summaries = run_analysis(
        current_sweep_path=args.current_sweep_json,
        scheduler_seed=args.scheduler_seed,
    )
    write_outputs(
        output_dir=args.output_dir,
        points=points,
        outcomes=outcomes,
        summaries=summaries,
        current_sweep_path=args.current_sweep_json,
        scheduler_seed=args.scheduler_seed,
    )
    print_summary(summaries)
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Tests for deterministic AAP parameter-stress analysis."""

import json

from assessment.benchmarks.aap_parameter_stress import (
    DIAGNOSTIC_VALUES,
    METHODOLOGY,
    STRESS_RANGES,
    build_parameter_points,
    build_stress_traces,
    compare_current_sweep,
    measure_bucket,
    measure_rate_monitor,
    print_summary,
    run_analysis,
    write_outputs,
)


def test_rate_monitor_profile_matches_nominal_global_boundaries():
    """The nominal global monitor activates and recovers at exact count boundaries."""
    profile = measure_rate_monitor(
        window_duration=10.0,
        abnormal_threshold=5.0,
        deactivation_threshold=4.0,
        min_observations=3,
    )

    assert profile.activation_count == 50
    assert profile.recovery_count == 40


def test_rate_monitor_profile_respects_minimum_observations():
    """Minimum evidence dominates when its count exceeds the rate boundary."""
    profile = measure_rate_monitor(
        window_duration=10.0,
        abnormal_threshold=0.5,
        deactivation_threshold=0.25,
        min_observations=8,
    )

    assert profile.activation_count == 8
    assert profile.recovery_count == 2


def test_bucket_profile_separates_activation_allowance_from_stored_burst():
    """Activation grants one budget while active idle refills reach capacity."""
    profile = measure_bucket(budget=15, period=1.0, burst_capacity=30)

    assert profile.initial_allowance == 15
    assert profile.stored_burst == 30


def test_parameter_points_cover_all_configured_ofat_values():
    """The deterministic analysis retains all 55 constraint-compatible values."""
    points = build_parameter_points()

    assert len(points) == 55
    assert {point.parameter for point in points} == {
        "alarm_window_duration",
        "alarm_abnormal_threshold",
        "alarm_deactivation_threshold",
        "alarm_limit_budget",
        "alarm_burst_capacity",
        "alarm_source_abnormal_threshold",
        "alarm_source_deactivation_threshold",
        "alarm_source_limit_budget",
        "alarm_source_burst_capacity",
        "alarm_min_observations",
    }


def test_parameter_points_are_feasible_without_repair():
    """Every candidate satisfies the paired AAP invariants as specified."""
    points = build_parameter_points()

    assert all(point.feasible for point in points)
    assert sum(len(values) for values in STRESS_RANGES.values()) == 55


def test_parameter_points_identify_diagnostic_values():
    """Diagnostic boundary checks remain distinct from operating candidates."""
    points = build_parameter_points()
    controls = {
        (point.parameter, point.value)
        for point in points
        if point.design_role == "diagnostic"
    }

    assert controls == {
        (parameter, value)
        for parameter, values in DIAGNOSTIC_VALUES.items()
        for value in values
    }


def test_compare_current_sweep_returns_unique_unstable_values(tmp_path):
    """Scenario comparison collapses scenario rows to parameter-value findings."""
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps(
            [
                {"parameter": "alarm_limit_budget", "value": 3, "stable": False},
                {"parameter": "alarm_limit_budget", "value": 3, "stable": False},
                {"parameter": "alarm_limit_budget", "value": 5, "stable": True},
            ]
        ),
        encoding="utf-8",
    )

    comparison = compare_current_sweep(sweep_path)

    assert comparison == {"alarm_limit_budget": [3.0]}


def test_overload_traces_spread_legitimate_probes_across_time():
    """Legitimate probes isolate collateral loss instead of creating a burst."""
    traces = {
        trace.name: trace
        for trace in build_stress_traces()
    }

    for name in ("single_source_overload", "multi_source_overload"):
        trace = traces[name]
        legitimate_times = [
            timestamp
            for timestamp, legitimate in zip(
                trace.arrival_times,
                trace.source_is_legitimate,
            )
            if legitimate
        ]
        assert len(legitimate_times) == 20
        assert len(set(legitimate_times)) == 20


def test_legitimate_control_includes_recurrent_boundary_source():
    """The control directly tests the 0.3 alarms/s legitimate-source boundary."""
    trace = next(
        trace
        for trace in build_stress_traces()
        if trace.name == "legitimate_control"
    )
    boundary_events = [
        timestamp
        for timestamp, device_id in zip(trace.arrival_times, trace.device_ids)
        if device_id == "legitimate_boundary"
    ]

    assert len(boundary_events) == 9
    assert len(boundary_events) / 30.0 == 0.3


def test_complete_analysis_writes_all_results(tmp_path, capsys):
    """The command workflow retains all points, outcomes, and summaries."""
    sweep_path = tmp_path / "current_sweep.json"
    sweep_path.write_text(
        json.dumps(
            [
                {
                    "parameter": "alarm_limit_budget",
                    "value": 3,
                    "stable": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    points, outcomes, summaries = run_analysis(
        current_sweep_path=sweep_path,
        scheduler_seed=999,
    )
    output_dir = tmp_path / "results"
    write_outputs(
        output_dir=output_dir,
        points=points,
        outcomes=outcomes,
        summaries=summaries,
        current_sweep_path=sweep_path,
        scheduler_seed=999,
    )
    print_summary(summaries)

    assert len(points) == 55
    assert len(outcomes) == 165
    assert len(summaries) == 10
    source_threshold = next(
        summary
        for summary in summaries
        if summary.parameter == "alarm_source_abnormal_threshold"
    )
    assert source_threshold.legitimate_source_activation_values == (0.25,)
    boundary_outcome = next(
        outcome
        for outcome in outcomes
        if (
            outcome.parameter == "alarm_source_abnormal_threshold"
            and outcome.value == 0.25
            and outcome.trace == "legitimate_control"
        )
    )
    assert boundary_outcome.source_activations > 0
    assert boundary_outcome.legitimate_alarm_dropped == 0
    assert (output_dir / "aap_parameter_stress_points.csv").is_file()
    assert (output_dir / "aap_parameter_stress_outcomes.csv").is_file()
    assert (output_dir / "aap_parameter_stress_summary.csv").is_file()
    payload = json.loads(
        (output_dir / "aap_parameter_stress.json").read_text(encoding="utf-8")
    )
    assert payload["methodology"] == METHODOLOGY
    assert payload["scheduler_seed"] == 999
    assert "AAP PARAMETER STRESS SUMMARY" in capsys.readouterr().out

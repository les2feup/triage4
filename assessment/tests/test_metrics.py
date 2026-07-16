"""
Tests for metrics computation module.

Verifies Jain fairness index, alarm metrics, bandwidth metrics,
and per-band fairness calculations.
"""

import pytest

from assessment.metrics import (
    compute_alarm_metrics,
    compute_all_metrics,
    compute_bandwidth_metrics,
    compute_detector_error_metrics,
    compute_device_fairness,
    compute_fairness_per_band,
    compute_high_priority_overhead,
    compute_per_source_fairness,
    jain_fairness_index,
)
from assessment.metrics.results import SchedulerResult


# === Jain Fairness Index Tests ===


def test_jain_fairness_perfect():
    """Perfect fairness: all values equal."""
    values = [1.0, 1.0, 1.0, 1.0]
    jfi = jain_fairness_index(values)
    assert jfi == pytest.approx(1.0, abs=1e-6)


def test_jain_fairness_unfair():
    """Maximum unfairness: one gets all."""
    values = [100.0, 0.0, 0.0, 0.0]
    jfi = jain_fairness_index(values)
    # JFI = (100)^2 / (4 * 10000) = 10000 / 40000 = 0.25
    assert jfi == pytest.approx(0.25, abs=1e-6)


def test_jain_fairness_partial():
    """Partial fairness: some variation."""
    values = [10.0, 8.0, 9.0]
    jfi = jain_fairness_index(values)
    # JFI = (27)^2 / (3 * (100 + 64 + 81)) = 729 / 735 ≈ 0.992
    assert 0.99 < jfi < 1.0


def test_jain_fairness_empty():
    """Empty list returns perfect fairness."""
    assert jain_fairness_index([]) == 1.0


def test_jain_fairness_single():
    """Single value returns perfect fairness."""
    assert jain_fairness_index([5.0]) == 1.0


def test_jain_fairness_zeros():
    """All zeros returns perfect fairness."""
    assert jain_fairness_index([0.0, 0.0, 0.0]) == 1.0


# === Alarm Metrics Tests ===


def test_alarm_metrics_basic():
    """Compute alarm metrics with alarms present."""
    result = SchedulerResult(
        waiting_times=[0.0, 0.5, 1.0, 0.2],
        e2e_times=[0.1, 0.6, 1.1, 0.3],
        priorities=[1, 0, 1, 0],  # Indices 1 and 3 are alarms
        metadata={},
    )
    arrival_times = [0.0, 0.1, 0.2, 0.3]
    is_alarm = [False, True, False, True]

    metrics = compute_alarm_metrics(result, arrival_times, is_alarm)

    # Alarm waiting times: [0.5, 0.2]
    assert metrics["alarm_avg_latency"] == pytest.approx(0.35, abs=1e-6)
    assert metrics["alarm_p95_latency"] == pytest.approx(0.485, abs=1e-2)
    assert metrics["alarm_count"] == 2


def test_alarm_latency_excludes_dropped_alarms():
    """A shed alarm must not enter the latency as a zero-wait delivery.

    The scheduler stores a dropped job with waiting_time 0.0, which is
    indistinguishable from a job served instantly. Without the delivered mask the
    mean below would read 0.2 (three alarms, two of them shed zeros) and a
    scheduler would look faster the more emergencies it discarded.
    """
    result = SchedulerResult(
        waiting_times=[0.6, 0.0, 0.0],
        e2e_times=[0.7, 0.0, 0.0],
        priorities=[0, 0, 0],
        delivered=[True, False, False],
        metadata={},
    )
    metrics = compute_alarm_metrics(result, [0.0, 0.1, 0.2], [True, True, True])

    assert metrics["alarm_avg_latency"] == pytest.approx(0.6, abs=1e-6)
    assert metrics["alarm_count"] == 3
    assert metrics["alarm_delivered_count"] == 1


def test_alarm_latency_counts_genuine_zero_wait_delivery():
    """A real zero-wait delivery still counts; only dropped jobs are excluded."""
    result = SchedulerResult(
        waiting_times=[0.0, 0.4],
        e2e_times=[0.1, 0.5],
        priorities=[0, 0],
        delivered=[True, True],
        metadata={},
    )
    metrics = compute_alarm_metrics(result, [0.0, 0.1], [True, True])

    assert metrics["alarm_avg_latency"] == pytest.approx(0.2, abs=1e-6)
    assert metrics["alarm_delivered_count"] == 2


def test_alarm_metrics_without_delivered_mask_serves_everything():
    """Baselines leave `delivered` as None, which must mean nothing was dropped."""
    result = SchedulerResult(
        waiting_times=[0.0, 0.4],
        e2e_times=[0.1, 0.5],
        priorities=[0, 0],
        metadata={},
    )
    metrics = compute_alarm_metrics(result, [0.0, 0.1], [True, True])

    assert metrics["alarm_avg_latency"] == pytest.approx(0.2, abs=1e-6)
    assert metrics["alarm_delivered_count"] == 2


def test_source_fairness_ignores_shed_sources():
    """A silenced source must not score as the best-served one.

    Zone 1 has both alarms shed. Counting those zeros would give it a mean
    latency of 0.0 against zone 0's 0.5 and report the shedding as unfairness
    that never happened; with the mask, only zone 0 was served, so the index is
    over a single source.
    """
    result = SchedulerResult(
        waiting_times=[0.5, 0.0, 0.0],
        e2e_times=[0.6, 0.0, 0.0],
        priorities=[0, 0, 0],
        delivered=[True, False, False],
        metadata={},
    )
    metrics = compute_per_source_fairness(result, [0, 1, 1], [True, True, True])

    assert metrics["alarm_source_count"] == 1
    assert metrics["alarm_source_fairness"] == pytest.approx(1.0, abs=1e-6)


def test_device_throughput_fairness_keeps_fully_shed_device():
    """A device whose traffic is entirely shed counts as zero, not as absent.

    Dropping it from the index would report a starved device as perfect fairness.
    """
    result = SchedulerResult(
        waiting_times=[0.1, 0.1, 0.0],
        e2e_times=[0.2, 0.2, 0.0],
        priorities=[0, 0, 0],
        delivered=[True, True, False],
        metadata={},
    )
    metrics = compute_device_fairness(result, ["a", "a", "b"])

    # Delivered counts are a=2, b=0 -> Jain = 4 / (2 * 4) = 0.5
    assert metrics["device_throughput_fairness"] == pytest.approx(0.5, abs=1e-6)


def test_alarm_metrics_no_alarms():
    """Compute alarm metrics with no alarms."""
    result = SchedulerResult(
        waiting_times=[0.1, 0.2],
        e2e_times=[0.2, 0.3],
        priorities=[1, 2],
        metadata={},
    )
    arrival_times = [0.0, 0.1]
    is_alarm = [False, False]

    metrics = compute_alarm_metrics(result, arrival_times, is_alarm)

    assert metrics["alarm_avg_latency"] == 0.0
    assert metrics["alarm_p95_latency"] == 0.0
    assert metrics["alarm_count"] == 0


# === Bandwidth Metrics Tests ===


def test_bandwidth_metrics_equal():
    """Devices with equal message rates."""
    result = SchedulerResult(
        waiting_times=[0.0, 0.1, 0.0, 0.1],
        e2e_times=[0.1, 0.2, 0.1, 0.2],
        priorities=[1, 1, 1, 1],
        metadata={},
    )
    arrival_times = [0.0, 0.5, 0.0, 0.5]  # Duration = 0.7s with service
    device_ids = ["A", "A", "B", "B"]  # Each device sends 2 messages

    metrics = compute_bandwidth_metrics(result, arrival_times, device_ids)

    # Each device: 2 msgs / 0.7s ≈ 2.857 msg/s (makespan-based)
    assert metrics["min_device_rate"] == pytest.approx(2.857, rel=1e-3)
    assert metrics["avg_device_rate"] == pytest.approx(2.857, rel=1e-3)
    assert metrics["max_device_rate"] == pytest.approx(2.857, rel=1e-3)


def test_bandwidth_metrics_unequal():
    """Devices with different message rates."""
    result = SchedulerResult(
        waiting_times=[0.0] * 5,
        e2e_times=[0.1] * 5,
        priorities=[1] * 5,
        metadata={},
    )
    arrival_times = [0.0, 0.1, 0.2, 0.3, 1.0]  # Makespan = 1.1s
    device_ids = ["A", "A", "A", "A", "B"]  # A sends 4, B sends 1

    metrics = compute_bandwidth_metrics(result, arrival_times, device_ids)

    # A: 4 msg / 1.1s ≈ 3.636 msg/s
    # B: 1 msg / 1.1s ≈ 0.909 msg/s
    assert metrics["min_device_rate"] == pytest.approx(0.909, rel=1e-3)
    assert metrics["avg_device_rate"] == pytest.approx(2.272, rel=1e-3)
    assert metrics["max_device_rate"] == pytest.approx(3.636, rel=1e-3)


def test_bandwidth_metrics_empty():
    """Empty workload returns zeros."""
    result = SchedulerResult(
        waiting_times=[],
        e2e_times=[],
        priorities=[],
        metadata={},
    )
    arrival_times = []
    device_ids = []

    metrics = compute_bandwidth_metrics(result, arrival_times, device_ids)

    assert metrics["min_device_rate"] == 0.0
    assert metrics["avg_device_rate"] == 0.0
    assert metrics["max_device_rate"] == 0.0


# === Per-Band Fairness Tests ===


def test_fairness_per_band_single_band():
    """Fairness in a single band with multiple devices."""
    result = SchedulerResult(
        waiting_times=[0.1, 0.1, 0.2, 0.2],
        e2e_times=[0.2, 0.2, 0.3, 0.3],
        priorities=[1, 1, 1, 1],  # All in HIGH band
        metadata={},
    )
    device_ids = ["A", "A", "B", "B"]

    metrics = compute_fairness_per_band(result, device_ids)

    # Device A avg: 0.1, Device B avg: 0.2
    # JFI([0.1, 0.2]) = (0.3)^2 / (2 * (0.01 + 0.04)) = 0.09 / 0.1 = 0.9
    assert metrics["band_1_fairness"] == pytest.approx(0.9, abs=1e-6)

    # Other bands empty → perfect fairness
    assert metrics["band_0_fairness"] == 1.0
    assert metrics["band_2_fairness"] == 1.0
    assert metrics["band_3_fairness"] == 1.0


def test_fairness_per_band_multiple_bands():
    """Fairness across multiple bands."""
    result = SchedulerResult(
        waiting_times=[0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
        e2e_times=[0.1] * 6,
        priorities=[0, 0, 1, 1, 2, 2],  # ALARM, HIGH, STANDARD
        metadata={},
    )
    device_ids = ["A", "A", "B", "B", "C", "C"]

    metrics = compute_fairness_per_band(result, device_ids)

    # Each band has perfect fairness (single device)
    assert metrics["band_0_fairness"] == 1.0  # Device A only
    assert metrics["band_1_fairness"] == 1.0  # Device B only
    assert metrics["band_2_fairness"] == 1.0  # Device C only


def test_fairness_per_band_empty_band():
    """Empty bands return perfect fairness."""
    result = SchedulerResult(
        waiting_times=[0.1, 0.2],
        e2e_times=[0.2, 0.3],
        priorities=[1, 1],  # Only HIGH band
        metadata={},
    )
    device_ids = ["A", "B"]

    metrics = compute_fairness_per_band(result, device_ids)

    # Only band 1 has messages
    assert "band_0_fairness" in metrics
    assert "band_1_fairness" in metrics
    assert "band_2_fairness" in metrics
    assert "band_3_fairness" in metrics

    # Empty bands → perfect fairness
    assert metrics["band_0_fairness"] == 1.0
    assert metrics["band_2_fairness"] == 1.0
    assert metrics["band_3_fairness"] == 1.0


# === High-Priority Overhead Tests ===


def test_high_priority_overhead_basic():
    """Compute overhead for HIGH band."""
    result = SchedulerResult(
        waiting_times=[0.1, 0.2, 0.5, 0.8],
        e2e_times=[0.2, 0.3, 0.6, 0.9],
        priorities=[1, 1, 2, 3],  # Two HIGH band messages
        metadata={},
    )

    metrics = compute_high_priority_overhead(result)

    # HIGH band (indices 0, 1): waiting times [0.1, 0.2]
    assert metrics["high_avg_latency"] == pytest.approx(0.15, abs=1e-6)
    # P95 of [0.1, 0.2] ≈ 0.195
    assert metrics["high_p95_latency"] == pytest.approx(0.195, abs=1e-2)


def test_high_priority_overhead_no_high():
    """No HIGH band messages."""
    result = SchedulerResult(
        waiting_times=[0.5, 0.8],
        e2e_times=[0.6, 0.9],
        priorities=[2, 3],  # STANDARD and BACKGROUND only
        metadata={},
    )

    metrics = compute_high_priority_overhead(result)

    assert metrics["high_avg_latency"] == 0.0
    assert metrics["high_p95_latency"] == 0.0


# === Integration Tests ===


def test_compute_all_metrics_integration():
    """Compute all metrics from a complete result."""
    result = SchedulerResult(
        waiting_times=[0.0, 0.5, 0.2, 0.3, 1.0],
        e2e_times=[0.1, 0.6, 0.3, 0.4, 1.1],
        priorities=[0, 1, 1, 2, 3],  # ALARM, HIGH, HIGH, STANDARD, BACKGROUND
        metadata={},
    )
    arrival_times = [0.0, 0.1, 0.2, 0.3, 0.4]
    device_ids = ["A", "B", "C", "D", "E"]
    is_alarm = [True, False, False, False, False]

    metrics = compute_all_metrics(result, arrival_times, device_ids, is_alarm)

    # Should contain all metric categories
    assert "alarm_avg_latency" in metrics
    assert "min_device_rate" in metrics
    assert "band_0_fairness" in metrics
    assert "high_avg_latency" in metrics
    assert "total_messages" in metrics
    assert "avg_waiting_time" in metrics

    # Check basic values
    assert metrics["total_messages"] == 5
    assert metrics["alarm_count"] == 1
    assert metrics["alarm_avg_latency"] == 0.0  # Alarm had 0.0 wait


def test_compute_all_metrics_with_seps():
    """Integration test with actual TRIAGE/4 scheduler."""
    from triage4 import TRIAGE4Config, TRIAGE4Scheduler

    scheduler = TRIAGE4Scheduler(TRIAGE4Config(), scheduler_seed=42)

    arrival_times = [0.0, 0.1, 0.2]
    device_ids = ["A", "B", "C"]
    zone_priorities = [0, 2, 5]
    is_alarm = [False, False, True]  # Last is alarm

    result = scheduler.schedule(arrival_times, device_ids, zone_priorities, is_alarm)

    metrics = compute_all_metrics(result, arrival_times, device_ids, is_alarm)

    # Should have all metric categories
    assert "alarm_avg_latency" in metrics
    assert "alarm_count" in metrics
    assert metrics["alarm_count"] == 1

    # Should have fairness metrics for all bands
    assert "band_0_fairness" in metrics
    assert "band_1_fairness" in metrics
    assert "band_2_fairness" in metrics
    assert "band_3_fairness" in metrics

    # Should have basic stats
    assert metrics["total_messages"] == 3
    assert metrics["avg_waiting_time"] >= 0
    assert metrics["min_device_rate"] > 0


# === Detector-Error Metrics Tests (R2.2) ===


def _make_result(waiting_times, priorities, metadata=None):
    n = len(waiting_times)
    return SchedulerResult(
        waiting_times=waiting_times,
        e2e_times=[w + 0.1 for w in waiting_times],
        priorities=priorities,
        metadata=metadata or {},
    )


def test_detector_error_metrics_no_ground_truth():
    """When ground_truth=None, every detected alarm is a TP."""
    result = _make_result([0.0, 0.5, 1.0], [0, 1, 0])
    is_alarm = [True, False, True]

    m = compute_detector_error_metrics(result, is_alarm, ground_truth_is_alarm=None)

    assert m["n_true_positives"] == 2
    assert m["n_false_negatives"] == 0
    assert m["n_false_positives"] == 0
    assert m["tp_latency"] == pytest.approx(0.5, abs=1e-6)  # mean of [0.0, 1.0]
    assert m["fn_demotion_latency"] == 0.0
    assert m["fp_alarm_latency"] == 0.0


def test_detector_error_metrics_with_ground_truth():
    """TP/FN/FP correctly classified against ground truth."""
    # Messages:  idx=0 GT=T det=T (TP), idx=1 GT=T det=F (FN), idx=2 GT=F det=T (FP), idx=3 GT=F det=F (TN)
    result = _make_result([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 2])
    is_alarm = [True, False, True, False]
    gt = [True, True, False, False]

    m = compute_detector_error_metrics(result, is_alarm, ground_truth_is_alarm=gt)

    assert m["n_true_positives"] == 1
    assert m["n_false_negatives"] == 1
    assert m["n_false_positives"] == 1
    assert m["tp_latency"] == pytest.approx(0.1)
    assert m["fn_demotion_latency"] == pytest.approx(0.2)
    assert m["fp_alarm_latency"] == pytest.approx(0.3)


def test_detector_error_metrics_zero_error():
    """Zero-error workload: detected == ground truth → no FN or FP."""
    result = _make_result([0.0, 0.5], [0, 1])
    is_alarm = [True, False]
    gt = [True, False]

    m = compute_detector_error_metrics(result, is_alarm, ground_truth_is_alarm=gt)

    assert m["n_false_negatives"] == 0
    assert m["n_false_positives"] == 0
    assert m["n_true_positives"] == 1


def test_compute_all_metrics_includes_detector_error():
    """compute_all_metrics forwards ground_truth and includes TP/FN/FP keys."""
    result = SchedulerResult(
        waiting_times=[0.0, 0.1, 0.2, 0.3],
        e2e_times=[0.1, 0.2, 0.3, 0.4],
        priorities=[0, 1, 0, 2],
        metadata={},
    )
    arrival_times = [0.0, 0.1, 0.2, 0.3]
    device_ids = ["A", "B", "C", "D"]
    is_alarm = [True, False, True, False]
    gt = [True, False, False, False]  # idx=2 is a FP

    m = compute_all_metrics(result, arrival_times, device_ids, is_alarm, ground_truth_is_alarm=gt)

    assert "tp_latency" in m
    assert "fn_demotion_latency" in m
    assert "fp_alarm_latency" in m
    assert m["n_true_positives"] == 1
    assert m["n_false_positives"] == 1
    assert m["n_false_negatives"] == 0


def test_compute_all_metrics_detector_error_defaults_when_no_gt():
    """Without ground_truth, detector-error keys still present (zero FN/FP)."""
    result = SchedulerResult(
        waiting_times=[0.0, 0.5],
        e2e_times=[0.1, 0.6],
        priorities=[0, 1],
        metadata={},
    )
    m = compute_all_metrics(result, [0.0, 0.1], ["A", "B"], [True, False])

    assert "tp_latency" in m
    assert m["n_false_negatives"] == 0
    assert m["n_false_positives"] == 0


# === AAP Activation/Deactivation Counter Tests ===


def test_aap_counters_in_metadata():
    """Scheduler metadata includes activation/deactivation counters."""
    from triage4 import TRIAGE4Config, TRIAGE4Scheduler
    from assessment.workloads import generate_alarm_flood_attack

    cfg = TRIAGE4Config(enable_alarm_protection=True)
    scheduler = TRIAGE4Scheduler(cfg, scheduler_seed=42)
    workload = generate_alarm_flood_attack(seed=42)

    result = scheduler.schedule(
        arrival_times=workload.arrival_times,
        device_ids=workload.device_ids,
        zone_priorities=workload.zone_priorities,
        is_alarm=workload.is_alarm,
    )

    assert "alarm_protection_activations" in result.metadata
    assert "alarm_protection_deactivations" in result.metadata
    assert isinstance(result.metadata["alarm_protection_activations"], int)
    assert isinstance(result.metadata["alarm_protection_deactivations"], int)
    assert result.metadata["alarm_protection_activations"] >= 0
    assert result.metadata["alarm_protection_deactivations"] >= 0


def test_aap_counters_zero_when_protection_disabled():
    """Without alarm protection, counters are zero."""
    from triage4 import TRIAGE4Config, TRIAGE4Scheduler
    from assessment.workloads import generate_alarm_flood_attack

    cfg = TRIAGE4Config(enable_alarm_protection=False)
    scheduler = TRIAGE4Scheduler(cfg, scheduler_seed=42)
    workload = generate_alarm_flood_attack(seed=42)

    result = scheduler.schedule(
        arrival_times=workload.arrival_times,
        device_ids=workload.device_ids,
        zone_priorities=workload.zone_priorities,
        is_alarm=workload.is_alarm,
    )

    assert result.metadata["alarm_protection_activations"] == 0
    assert result.metadata["alarm_protection_deactivations"] == 0


def test_aap_counters_propagate_to_compute_all_metrics():
    """compute_all_metrics exposes activation/deactivation counters as floats."""
    from triage4 import TRIAGE4Config, TRIAGE4Scheduler
    from assessment.workloads import generate_alarm_flood_attack

    cfg = TRIAGE4Config(enable_alarm_protection=True)
    scheduler = TRIAGE4Scheduler(cfg, scheduler_seed=42)
    workload = generate_alarm_flood_attack(seed=42)

    result = scheduler.schedule(
        arrival_times=workload.arrival_times,
        device_ids=workload.device_ids,
        zone_priorities=workload.zone_priorities,
        is_alarm=workload.is_alarm,
    )
    m = compute_all_metrics(
        result, workload.arrival_times, workload.device_ids, workload.is_alarm,
        zone_priorities=workload.zone_priorities,
    )

    assert "alarm_protection_activations" in m
    assert "alarm_protection_deactivations" in m
    assert m["alarm_protection_activations"] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

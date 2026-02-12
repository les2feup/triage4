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
    compute_fairness_per_band,
    compute_high_priority_overhead,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

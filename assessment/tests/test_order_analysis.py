"""
Tests for order-based analysis module.

Verifies service order reconstruction, position jump metrics,
and CSV export functionality.
"""

import csv
import os
import tempfile
from pathlib import Path

import pytest

from assessment.metrics import (
    compute_order_metrics,
    compute_service_order,
    export_input_order,
    export_output_order,
)
from assessment.metrics.order_analysis import compute_consecutive_serves, count_band_inversions
from assessment.metrics.results import SchedulerResult


# === Service Order Reconstruction Tests ===


def test_compute_service_order_fifo():
    """FIFO scheduler: service order = arrival order."""
    # FIFO: all jobs served in arrival order, no waiting
    arrival_times = [0.0, 0.1, 0.2, 0.3]
    waiting_times = [0.0, 0.0, 0.0, 0.0]  # No reordering

    result = SchedulerResult(
        waiting_times=waiting_times,
        e2e_times=[0.1, 0.1, 0.1, 0.1],
        priorities=[1, 1, 1, 1],
        metadata={}
    )

    service_order = compute_service_order(result, arrival_times)

    # FIFO: service order = [0, 1, 2, 3]
    assert service_order == [0, 1, 2, 3]


def test_compute_service_order_reordering():
    """Non-FIFO: service order differs from arrival."""
    arrival_times = [0.0, 0.1, 0.2]
    # Job 2 served first (0 wait), job 0 second (0.3 wait), job 1 last (0.4 wait)
    waiting_times = [0.3, 0.4, 0.0]

    result = SchedulerResult(
        waiting_times=waiting_times,
        e2e_times=[0.4, 0.5, 0.1],
        priorities=[1, 1, 0],  # Job 2 is ALARM
        metadata={}
    )

    service_order = compute_service_order(result, arrival_times)

    # Job 2 served at t=0.2+0.0=0.2 (first)
    # Job 0 served at t=0.0+0.3=0.3 (second)
    # Job 1 served at t=0.1+0.4=0.5 (third)
    assert service_order == [2, 0, 1]


def test_compute_service_order_simultaneous():
    """Handle simultaneous arrivals correctly."""
    arrival_times = [0.0, 0.0, 0.0]
    waiting_times = [0.0, 0.1, 0.2]

    result = SchedulerResult(
        waiting_times=waiting_times,
        e2e_times=[0.1, 0.2, 0.3],
        priorities=[0, 1, 2],
        metadata={}
    )

    service_order = compute_service_order(result, arrival_times)

    # Jobs served at t=0.0, 0.1, 0.2
    assert service_order == [0, 1, 2]


# === Position Jump Calculation Tests ===


def test_position_jump_calculation():
    """Position jump = service_rank - arrival_rank."""
    arrival_times = [0.0, 0.1, 0.2, 0.3]
    # Job 2 served first (at t=0.2), job 0 second (at t=0.3), job 1 third (at t=0.5), job 3 fourth (at t=0.6)
    waiting_times = [0.3, 0.4, 0.0, 0.3]

    result = SchedulerResult(
        waiting_times=waiting_times,
        e2e_times=[0.4, 0.5, 0.1, 0.4],
        priorities=[1, 1, 0, 2],
        metadata={}
    )

    service_order = compute_service_order(result, arrival_times)
    # Service times: job 0 at 0.3, job 1 at 0.5, job 2 at 0.2, job 3 at 0.6
    # Sorted: [2, 0, 1, 3]
    assert service_order == [2, 0, 1, 3]

    # Position jumps:
    # Job 0: arrival_rank=0, service_rank=1 → jump=+1 (pushed back)
    # Job 1: arrival_rank=1, service_rank=2 → jump=+1 (pushed back)
    # Job 2: arrival_rank=2, service_rank=0 → jump=-2 (jumped forward)
    # Job 3: arrival_rank=3, service_rank=3 → jump=0 (no change)

    metrics = compute_order_metrics(
        result, arrival_times, ["A", "B", "C", "D"], [False, False, True, False]
    )

    assert metrics["position_jump_mean"] == pytest.approx(0.0, abs=0.01)  # (-2+1+1+0)/4=0
    assert metrics["position_jump_min"] == -2  # Job 2 jumped forward most
    assert metrics["position_jump_max"] == 1  # Jobs 0,1 pushed back


# === Alarm Metrics Tests ===


def test_alarm_jump_metrics():
    """Alarm jump distance computed correctly."""
    arrival_times = [0.0, 0.1, 0.2, 0.3, 0.4]
    is_alarm = [False, False, True, False, False]
    # Alarm (job 2) served first, then others
    # Service times: job 0 at 0.3, job 1 at 0.5, job 2 at 0.2 (alarm), job 3 at 0.6, job 4 at 0.8
    waiting_times = [0.3, 0.4, 0.0, 0.3, 0.4]

    result = SchedulerResult(
        waiting_times=waiting_times,
        e2e_times=[0.4, 0.5, 0.1, 0.4, 0.5],
        priorities=[1, 1, 0, 1, 1],
        metadata={}
    )

    metrics = compute_order_metrics(result, arrival_times, ["A"] * 5, is_alarm)

    # Service order: [2, 0, 1, 3, 4]
    # Alarm (job 2): arrival_rank=2, service_rank=0 → jump=-2
    assert metrics["alarm_count"] == 1
    assert metrics["alarm_avg_jump"] == -2.0
    assert metrics["alarm_max_forward_jump"] == -2.0


def test_alarm_jump_no_alarms():
    """Handle scenarios with no alarms."""
    arrival_times = [0.0, 0.1, 0.2]
    is_alarm = [False, False, False]
    waiting_times = [0.0, 0.0, 0.0]

    result = SchedulerResult(
        waiting_times=waiting_times,
        e2e_times=[0.1, 0.1, 0.1],
        priorities=[1, 1, 1],
        metadata={}
    )

    metrics = compute_order_metrics(result, arrival_times, ["A", "B", "C"], is_alarm)

    assert metrics["alarm_count"] == 0
    assert metrics["alarm_avg_jump"] == 0.0
    assert metrics["alarm_max_forward_jump"] == 0.0


# === Consecutive Serves Tests ===


def test_consecutive_serves_perfect_interleaving():
    """Perfect round-robin: all counts = 1."""
    service_order = [0, 1, 2, 0, 1, 2]  # A, B, C, A, B, C
    device_ids = ["A", "B", "C", "A", "B", "C"]

    counts = compute_consecutive_serves(service_order, device_ids)

    # Pattern: A, B, C, A, B, C → [1, 1, 1, 1, 1, 1]
    assert counts == [1, 1, 1, 1, 1, 1]


def test_consecutive_serves_monopolization():
    """Device monopolization: some high counts."""
    # 6 jobs: job 0=device A, job 1=device B, jobs 2,3,4=device C, job 5=device D
    service_order = [0, 1, 2, 3, 4, 5]  # Jobs served in order
    device_ids = ["A", "B", "C", "C", "C", "D"]  # Indexed by job

    counts = compute_consecutive_serves(service_order, device_ids)

    # Service pattern: A, B, C, C, C, D → counts: [1, 1, 3, 1]
    assert counts == [1, 1, 3, 1]


def test_consecutive_serves_single_device():
    """Single device: one large count."""
    service_order = [0, 0, 0, 0]
    device_ids = ["A", "A", "A", "A"]

    counts = compute_consecutive_serves(service_order, device_ids)

    assert counts == [4]


def test_consecutive_serves_empty():
    """Empty service order."""
    counts = compute_consecutive_serves([], [])
    assert counts == []


# === Band Inversion Tests ===


def test_band_inversions_none():
    """Perfect priority order: no inversions."""
    # Service order: ALARM, HIGH, STANDARD, BACKGROUND
    service_order = [0, 1, 2, 3]
    band_assignments = [0, 1, 2, 3]

    inversions = count_band_inversions(service_order, band_assignments)

    assert inversions == 0


def test_band_inversions_some():
    """Some priority inversions."""
    # Service order: HIGH, STANDARD, HIGH, BACKGROUND
    #                 Job 0,    1,    2,       3
    # Bands:           1,      2,    1,       3
    # Inversions:
    #   - Job 0 (band 1) before Job 1 (band 2): OK
    #   - Job 1 (band 2) before Job 2 (band 1): INVERSION
    #   - Job 1 (band 2) before Job 3 (band 3): OK
    #   - Job 2 (band 1) before Job 3 (band 3): OK
    service_order = [0, 1, 2, 3]
    band_assignments = [1, 2, 1, 3]

    inversions = count_band_inversions(service_order, band_assignments)

    assert inversions == 1


def test_band_inversions_many():
    """Multiple inversions."""
    # Service order: BACKGROUND, STANDARD, HIGH, ALARM
    service_order = [0, 1, 2, 3]
    band_assignments = [3, 2, 1, 0]

    # Every pair is inverted except those in correct order
    # 3 > 2, 3 > 1, 3 > 0, 2 > 1, 2 > 0, 1 > 0 → 6 inversions
    inversions = count_band_inversions(service_order, band_assignments)

    assert inversions == 6


def test_band_inversions_no_bands():
    """Handle missing band information."""
    service_order = [0, 1, 2]
    inversions = count_band_inversions(service_order, None)
    assert inversions == 0


# === CSV Export Tests ===


def test_export_input_csv():
    """Input CSV exported with correct format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        arrival_times = [0.0, 0.1, 0.2]
        device_ids = ["A", "B", "C"]
        zone_priorities = [0, 2, 5]
        is_alarm = [False, False, True]

        csv_path = export_input_order(
            arrival_times, device_ids, zone_priorities, is_alarm,
            scenario_name="test_scenario",
            output_dir=tmpdir
        )

        # Verify file exists
        assert os.path.exists(csv_path)

        # Read and verify contents
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 3

            # Check first row
            assert rows[0]["arrival_rank"] == "0"
            assert rows[0]["device_id"] == "A"
            assert rows[0]["zone_priority"] == "0"
            assert rows[0]["is_alarm"] == "False"

            # Check alarm row
            assert rows[2]["arrival_rank"] == "2"
            assert rows[2]["is_alarm"] == "True"


def test_export_output_csv():
    """Output CSV exported with position jumps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        arrival_times = [0.0, 0.1, 0.2]
        device_ids = ["A", "B", "C"]
        zone_priorities = [0, 2, 5]
        is_alarm = [False, False, True]
        # Service order: [2, 0, 1] (alarm first)
        waiting_times = [0.3, 0.4, 0.0]

        result = SchedulerResult(
            waiting_times=waiting_times,
            e2e_times=[0.4, 0.5, 0.1],
            priorities=[1, 2, 0],
            metadata={}
        )

        csv_path = export_output_order(
            result, arrival_times, device_ids, zone_priorities, is_alarm,
            scheduler_name="TestScheduler",
            scenario_name="test_scenario",
            output_dir=tmpdir
        )

        # Verify file exists
        assert os.path.exists(csv_path)

        # Read and verify contents
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 3

            # First served: job 2 (alarm)
            assert rows[0]["service_rank"] == "0"
            assert rows[0]["arrival_rank"] == "2"
            assert rows[0]["position_jump"] == "-2"
            assert rows[0]["band"] == "0"

            # Second served: job 0
            assert rows[1]["service_rank"] == "1"
            assert rows[1]["arrival_rank"] == "0"
            assert rows[1]["position_jump"] == "1"

            # Third served: job 1
            assert rows[2]["service_rank"] == "2"
            assert rows[2]["arrival_rank"] == "1"
            assert rows[2]["position_jump"] == "1"


def test_csv_directory_creation():
    """CSV export creates directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        arrival_times = [0.0]
        device_ids = ["A"]
        zone_priorities = [0]
        is_alarm = [False]

        export_input_order(
            arrival_times, device_ids, zone_priorities, is_alarm,
            scenario_name="nested/test/scenario",
            output_dir=tmpdir
        )

        # Verify nested directory created
        expected_dir = Path(tmpdir) / "nested" / "test" / "scenario"
        assert expected_dir.exists()
        assert (expected_dir / "input.csv").exists()


# === Integration Tests ===


def test_compute_order_metrics_integration():
    """Full integration test with realistic data."""
    from triage4 import TRIAGE4Config, TRIAGE4Scheduler

    # Create small test scenario
    arrival_times = [0.0, 0.0, 0.0, 0.1]
    device_ids = ["A", "B", "C", "D"]
    zone_priorities = [0, 0, 5, 0]
    is_alarm = [False, False, True, False]

    scheduler = TRIAGE4Scheduler(TRIAGE4Config(), scheduler_seed=42)
    result = scheduler.schedule(arrival_times, device_ids, zone_priorities, is_alarm)

    metrics = compute_order_metrics(result, arrival_times, device_ids, is_alarm)

    # Basic sanity checks
    assert "position_jump_mean" in metrics
    assert "alarm_avg_jump" in metrics
    assert "max_consecutive_serves" in metrics
    assert "band_inversion_count" in metrics

    # Alarm should have jumped forward (negative jump)
    assert metrics["alarm_count"] == 1
    assert metrics["alarm_avg_jump"] < 0

    # Band inversions should be 0 for TRIAGE/4
    assert metrics["band_inversion_count"] == 0


def test_order_metrics_with_baselines():
    """Test order metrics with baseline schedulers."""
    from assessment.baselines import FIFOScheduler, StrictPriorityScheduler

    arrival_times = [0.0, 0.1, 0.2]
    device_ids = ["A", "B", "C"]
    zone_priorities = [0, 2, 5]
    is_alarm = [False, False, True]

    # FIFO
    fifo = FIFOScheduler(service_rate=20.0, scheduler_seed=42)
    fifo_result = fifo.schedule(arrival_times, device_ids, zone_priorities, is_alarm)
    fifo_metrics = compute_order_metrics(fifo_result, arrival_times, device_ids, is_alarm)

    # FIFO should have minimal reordering
    assert abs(fifo_metrics["position_jump_mean"]) < 1.0

    # Strict Priority
    strict = StrictPriorityScheduler(service_rate=20.0, scheduler_seed=42)
    strict_result = strict.schedule(arrival_times, device_ids, zone_priorities, is_alarm)
    strict_metrics = compute_order_metrics(strict_result, arrival_times, device_ids, is_alarm)

    # Both should compute without errors
    assert "position_jump_mean" in strict_metrics
    assert "alarm_avg_jump" in strict_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for baseline schedulers (Strict Priority and FIFO).

Verifies that baselines behave correctly and demonstrate the problems
that TRIAGE/4 solves.
"""

import pytest

from assessment.baselines import FIFOScheduler, StrictPriorityScheduler


# === Strict Priority Scheduler Tests ===


def test_strict_priority_basic():
    """Strict priority serves highest zone priority first."""
    scheduler = StrictPriorityScheduler(service_rate=10.0, scheduler_seed=42)

    # Three messages from different zones, all arrive together
    result = scheduler.schedule(
        arrival_times=[0.0, 0.0, 0.0],
        device_ids=["A", "B", "C"],
        zone_priorities=[2, 0, 1],  # Zone 0 highest, then 1, then 2
        is_alarm=[False, False, False],
    )

    # Zone 0 (idx=1) served first despite arriving same time
    # Should have zero waiting time
    assert result.waiting_times[1] == 0.0

    # Zone 2 (idx=0) served last
    # Should have longest waiting time
    assert result.waiting_times[0] > result.waiting_times[1]
    assert result.waiting_times[0] > result.waiting_times[2]


def test_strict_priority_ignores_alarm():
    """Strict priority IGNORES is_alarm flag - demonstrates the problem."""
    scheduler = StrictPriorityScheduler(service_rate=10.0, scheduler_seed=42)

    # Alarm from low-priority zone vs routine from high-priority zone
    result = scheduler.schedule(
        arrival_times=[0.0, 0.0],
        device_ids=["sensor_low", "sensor_high"],
        zone_priorities=[5, 0],  # Low zone, High zone
        is_alarm=[True, False],  # ALARM ignored, routine served first!
    )

    # Zone 0 (idx=1, routine) served BEFORE zone 5 (idx=0, alarm)
    # This is the PROBLEM that TRIAGE/4 solves!
    assert result.waiting_times[1] == 0.0  # High zone served first
    assert result.waiting_times[0] > 0.0  # Alarm delayed!


def test_strict_priority_fifo_within_priority():
    """Within same priority, strict priority uses FIFO."""
    scheduler = StrictPriorityScheduler(service_rate=10.0, scheduler_seed=42)

    result = scheduler.schedule(
        arrival_times=[0.0, 0.1, 0.2],
        device_ids=["A", "B", "C"],
        zone_priorities=[1, 1, 1],  # All same priority
        is_alarm=[False, False, False],
    )

    # Should be served in arrival order: A, B, C
    assert result.waiting_times[0] == 0.0  # A arrives first, no wait
    assert result.waiting_times[1] > 0.0  # B waits
    assert result.waiting_times[2] > result.waiting_times[1]  # C waits longest


def test_strict_priority_validation():
    """Strict priority validates inputs."""
    scheduler = StrictPriorityScheduler()

    # Mismatched lengths
    with pytest.raises(ValueError, match="device_ids length"):
        scheduler.schedule(
            arrival_times=[0.0, 1.0],
            device_ids=["A"],  # Wrong length
            zone_priorities=[0, 1],
            is_alarm=[False, False],
        )

    # Unsorted arrivals
    with pytest.raises(ValueError, match="arrival_times must be sorted"):
        scheduler.schedule(
            arrival_times=[1.0, 0.0],  # Unsorted
            device_ids=["A", "B"],
            zone_priorities=[0, 1],
            is_alarm=[False, False],
        )


# === FIFO Scheduler Tests ===


def test_fifo_basic():
    """FIFO serves messages in strict arrival order."""
    scheduler = FIFOScheduler(service_rate=10.0, scheduler_seed=42)

    # Different priorities, served in arrival order
    result = scheduler.schedule(
        arrival_times=[0.0, 0.1, 0.2],
        device_ids=["A", "B", "C"],
        zone_priorities=[5, 0, 2],  # All ignored
        is_alarm=[False, True, False],  # All ignored
    )

    # Served in order: A (0.0), B (0.1), C (0.2)
    assert result.waiting_times[0] == 0.0  # A first
    assert result.waiting_times[1] > 0.0  # B waits
    assert result.waiting_times[2] > result.waiting_times[1]  # C waits longest


def test_fifo_ignores_priorities():
    """FIFO ignores all priority information."""
    scheduler = FIFOScheduler(service_rate=10.0, scheduler_seed=42)

    # High priority arrives last
    result = scheduler.schedule(
        arrival_times=[0.0, 0.1, 0.2],
        device_ids=["A", "B", "C"],
        zone_priorities=[5, 5, 0],  # Zone 0 highest but arrives last
        is_alarm=[False, False, True],  # Alarm arrives last
    )

    # Still served in arrival order despite C being high-priority alarm
    assert result.waiting_times[0] == 0.0  # A first
    assert result.waiting_times[2] > result.waiting_times[0]  # C waits


def test_fifo_simultaneous_arrivals():
    """FIFO preserves order for simultaneous arrivals."""
    scheduler = FIFOScheduler(service_rate=10.0, scheduler_seed=42)

    # All arrive at same time
    result = scheduler.schedule(
        arrival_times=[0.0, 0.0, 0.0],
        device_ids=["A", "B", "C"],
        zone_priorities=[0, 1, 2],
        is_alarm=[True, False, False],
    )

    # First in array served first (A), then B, then C
    assert result.waiting_times[0] == 0.0
    assert result.waiting_times[1] > 0.0
    assert result.waiting_times[2] > result.waiting_times[1]


def test_fifo_validation():
    """FIFO validates inputs."""
    scheduler = FIFOScheduler()

    # Mismatched lengths
    with pytest.raises(ValueError, match="zone_priorities length"):
        scheduler.schedule(
            arrival_times=[0.0, 1.0],
            device_ids=["A", "B"],
            zone_priorities=[0],  # Wrong length
            is_alarm=[False, False],
        )


# === Comparison Tests ===


def test_baselines_vs_seps_alarm_handling():
    """
    Compare how baselines vs TRIAGE/4 handle alarms.

    Demonstrates the core problem TRIAGE/4 solves.
    """
    from triage4 import TRIAGE4Config, TRIAGE4Scheduler

    # Scenario: High-priority zone burst + low-priority alarm
    arrival_times = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01]
    device_ids = ["H1", "H2", "H3", "H4", "H5", "L_alarm"]
    zone_priorities = [0, 0, 0, 0, 0, 5]  # 5 high-priority, 1 low-priority
    is_alarm = [False, False, False, False, False, True]  # Last is ALARM

    seed = 42

    # Strict Priority: Ignores alarm, serves by zone
    strict = StrictPriorityScheduler(service_rate=10.0, scheduler_seed=seed)
    strict_result = strict.schedule(arrival_times, device_ids, zone_priorities, is_alarm)

    # FIFO: Serves in order (alarm last since it arrives last)
    fifo = FIFOScheduler(service_rate=10.0, scheduler_seed=seed)
    fifo_result = fifo.schedule(arrival_times, device_ids, zone_priorities, is_alarm)

    # TRIAGE/4: Alarm preempts (should have low waiting time)
    triage4_config = TRIAGE4Config(service_rate=10.0)
    seps = TRIAGE4Scheduler(triage4_config, scheduler_seed=seed)
    seps_result = seps.schedule(arrival_times, device_ids, zone_priorities, is_alarm)

    # Alarm is job index 5
    # TRIAGE/4 should serve alarm much faster than baselines
    seps_alarm_wait = seps_result.waiting_times[5]
    strict_alarm_wait = strict_result.waiting_times[5]
    fifo_alarm_wait = fifo_result.waiting_times[5]

    # TRIAGE/4 alarm waiting should be less than both baselines
    assert seps_alarm_wait < strict_alarm_wait
    assert seps_alarm_wait < fifo_alarm_wait

    print(f"\n📊 Alarm Waiting Times:")
    print(f"   TRIAGE/4:   {seps_alarm_wait:.4f}s")
    print(f"   Strict: {strict_alarm_wait:.4f}s  ({strict_alarm_wait/seps_alarm_wait:.1f}x worse)")
    print(f"   FIFO:   {fifo_alarm_wait:.4f}s  ({fifo_alarm_wait/seps_alarm_wait:.1f}x worse)")


def test_baselines_with_workload():
    """Baselines work with generated workloads."""
    from assessment.workloads import generate_alarm_under_burst

    workload = generate_alarm_under_burst(n_edu_devices=3, messages_per_edu=5)

    seed = 42
    schedulers = [
        StrictPriorityScheduler(service_rate=20.0, scheduler_seed=seed),
        FIFOScheduler(service_rate=20.0, scheduler_seed=seed),
    ]

    for scheduler in schedulers:
        result = scheduler.schedule(
            arrival_times=workload.arrival_times,
            device_ids=workload.device_ids,
            zone_priorities=workload.zone_priorities,
            is_alarm=workload.is_alarm,
        )

        # Basic sanity checks
        assert result.n_jobs == workload.n_messages
        assert all(w >= 0 for w in result.waiting_times)
        assert all(e >= 0 for e in result.e2e_times)


def test_all_schedulers_same_interface():
    """All schedulers (TRIAGE/4 + baselines) use same interface."""
    from triage4 import TRIAGE4Config, TRIAGE4Scheduler

    # Same test workload
    arrival_times = [0.0, 0.1, 0.2]
    device_ids = ["A", "B", "C"]
    zone_priorities = [0, 2, 1]
    is_alarm = [False, True, False]

    schedulers = [
        TRIAGE4Scheduler(TRIAGE4Config(service_rate=10.0), scheduler_seed=42),
        StrictPriorityScheduler(service_rate=10.0, scheduler_seed=42),
        FIFOScheduler(service_rate=10.0, scheduler_seed=42),
    ]

    for scheduler in schedulers:
        result = scheduler.schedule(arrival_times, device_ids, zone_priorities, is_alarm)

        # All return compatible SchedulerResult
        assert result.n_jobs == 3
        assert len(result.waiting_times) == 3
        assert len(result.e2e_times) == 3
        assert result.metadata is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

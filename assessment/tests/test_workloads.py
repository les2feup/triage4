"""
Tests for workload generators.

Verifies that generated workloads match specifications from REFACTORING_PLAN.md.
"""

import pytest

from assessment.workloads import (
    Workload,
    generate_alarm_under_burst,
    generate_device_monopolization,
    generate_multi_zone_emergency,
)


# === Workload Data Structure Tests ===


def test_workload_creation():
    """Workload can be created with valid data."""
    workload = Workload(
        arrival_times=[0.0, 0.1, 0.2],
        device_ids=["A", "B", "C"],
        zone_priorities=[0, 1, 2],
        is_alarm=[False, False, True],
        description="Test workload",
    )

    assert workload.n_messages == 3
    assert workload.n_devices == 3
    assert workload.n_alarms == 1
    assert workload.duration == 0.2


def test_workload_validation():
    """Workload validates input consistency."""
    # Mismatched lengths
    with pytest.raises(ValueError, match="device_ids length"):
        Workload(
            arrival_times=[0.0, 0.1],
            device_ids=["A"],  # Wrong length
            zone_priorities=[0, 1],
            is_alarm=[False, False],
        )

    # Unsorted arrival times
    with pytest.raises(ValueError, match="arrival_times must be sorted"):
        Workload(
            arrival_times=[1.0, 0.0],  # Unsorted
            device_ids=["A", "B"],
            zone_priorities=[0, 1],
            is_alarm=[False, False],
        )


# === Scenario 1: Alarm Under Burst Load ===


def test_alarm_under_burst_basic():
    """Generate alarm under burst scenario with default params."""
    workload = generate_alarm_under_burst()

    # 2 EDUs × 40 messages + 1 alarm = 81
    assert workload.n_messages == 81
    assert workload.n_alarms == 1
    assert workload.n_devices == 3  # 2 EDUs + 1 sensor

    # Alarm is from low-priority zone
    alarm_indices = [i for i, is_alarm in enumerate(workload.is_alarm) if is_alarm]
    assert len(alarm_indices) == 1
    alarm_idx = alarm_indices[0]
    assert workload.zone_priorities[alarm_idx] == 5  # BACKGROUND zone


def test_alarm_under_burst_custom_params():
    """Generate alarm scenario with custom parameters."""
    workload = generate_alarm_under_burst(
        n_edu_devices=5,
        messages_per_edu=50,
        burst_duration=2.0,
        alarm_injection_time=1.0,
    )

    assert workload.n_messages == 251  # 5 × 50 + 1
    assert workload.n_alarms == 1
    assert workload.duration >= 1.0  # Alarm at t=1.0


def test_alarm_under_burst_sorted():
    """Workload arrivals are sorted."""
    workload = generate_alarm_under_burst()
    assert workload.arrival_times == sorted(workload.arrival_times)


def test_alarm_under_burst_edu_zone():
    """EDU devices are in high-priority zone."""
    workload = generate_alarm_under_burst(n_edu_devices=3)

    # Find EDU messages (non-alarm)
    edu_priorities = [
        workload.zone_priorities[i]
        for i in range(workload.n_messages)
        if not workload.is_alarm[i]
    ]

    # All EDU messages should be zone 0 (HIGH band)
    assert all(p == 0 for p in edu_priorities)


# === Scenario 2: Device Monopolization ===


def test_device_monopolization_basic():
    """Generate device monopolization scenario."""
    workload = generate_device_monopolization()

    # Default: 480 from A + 48 from B = 528
    assert workload.n_messages == 528
    assert workload.n_devices == 2
    assert workload.n_alarms == 0  # No alarms in this scenario


def test_device_monopolization_rates():
    """Verify device message rates."""
    workload = generate_device_monopolization(
        high_rate_messages=100, low_rate_messages=10, duration=10.0
    )

    # Count messages per device
    device_a_count = sum(1 for d in workload.device_ids if d == "device_A")
    device_b_count = sum(1 for d in workload.device_ids if d == "device_B")

    assert device_a_count == 100
    assert device_b_count == 10

    # High-rate device sends 10x more
    assert device_a_count / device_b_count == 10.0


def test_device_monopolization_same_zone():
    """Both devices in same priority zone."""
    workload = generate_device_monopolization(zone_priority=2)

    # All messages should be from zone 2
    assert all(zp == 2 for zp in workload.zone_priorities)


def test_device_monopolization_sorted():
    """Workload arrivals are sorted."""
    workload = generate_device_monopolization()
    assert workload.arrival_times == sorted(workload.arrival_times)


# === Scenario 3: Multi-Zone Emergency ===


def test_multi_zone_emergency_basic():
    """Generate multi-zone emergency scenario."""
    workload = generate_multi_zone_emergency()

    # Default: 6 zones × 2 devices × 10 messages = 120
    # Plus alarms from 2 zones × 3 alarms = 6
    # Total = 126
    assert workload.n_messages == 126
    assert workload.n_alarms == 6
    assert workload.n_devices > 6  # At least 6 zones worth (12 total devices)


def test_multi_zone_emergency_zones():
    """Messages from all zones present."""
    workload = generate_multi_zone_emergency(n_zones=4, devices_per_zone=2)

    # Should have messages from zones 0, 1, 2, 3
    zones_present = set(workload.zone_priorities)
    assert zones_present == {0, 1, 2, 3}


def test_multi_zone_emergency_alarms():
    """Alarms from specified zones."""
    alarm_zones = [1, 3]
    workload = generate_multi_zone_emergency(
        n_zones=5, alarm_zones=alarm_zones, alarms_per_zone=2
    )

    # Find alarm zone priorities
    alarm_priorities = [
        workload.zone_priorities[i]
        for i in range(workload.n_messages)
        if workload.is_alarm[i]
    ]

    # Alarms should only come from specified zones
    assert set(alarm_priorities) == set(alarm_zones)
    assert len(alarm_priorities) == len(alarm_zones) * 2  # 2 alarms per zone


def test_multi_zone_emergency_sorted():
    """Workload arrivals are sorted."""
    workload = generate_multi_zone_emergency()
    assert workload.arrival_times == sorted(workload.arrival_times)


def test_multi_zone_emergency_custom():
    """Generate with custom parameters."""
    workload = generate_multi_zone_emergency(
        n_zones=3,
        devices_per_zone=2,
        messages_per_device=5,
        alarm_zones=[0, 2],
        alarms_per_zone=1,
        duration=5.0,
    )

    # 3 zones × 2 devices × 5 messages = 30
    # 2 alarm zones × 1 alarm = 2
    # Total = 32
    assert workload.n_messages == 32
    assert workload.n_alarms == 2


# === Integration Tests ===


def test_workload_with_seps_interface():
    """Workload can be used directly with TRIAGE/4 scheduler."""
    from triage4 import TRIAGE4Config, TRIAGE4Scheduler

    workload = generate_alarm_under_burst(n_edu_devices=2, messages_per_edu=10)

    config = TRIAGE4Config()
    scheduler = TRIAGE4Scheduler(config, scheduler_seed=42)

    # Should work without errors
    result = scheduler.schedule(
        arrival_times=workload.arrival_times,
        device_ids=workload.device_ids,
        zone_priorities=workload.zone_priorities,
        is_alarm=workload.is_alarm,
    )

    assert result.n_jobs == workload.n_messages


def test_all_scenarios_valid():
    """All three scenarios generate valid workloads."""
    scenarios = [
        generate_alarm_under_burst(),
        generate_device_monopolization(),
        generate_multi_zone_emergency(),
    ]

    for workload in scenarios:
        # Basic validation
        assert workload.n_messages > 0
        assert workload.n_devices > 0
        assert workload.duration > 0

        # Arrival times sorted
        assert workload.arrival_times == sorted(workload.arrival_times)

        # Consistent lengths
        n = workload.n_messages
        assert len(workload.device_ids) == n
        assert len(workload.zone_priorities) == n
        assert len(workload.is_alarm) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

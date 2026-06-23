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


# === WFQ Scheduler Tests ===


def test_wfq_weighted_ordering():
    """
    WFQ serves lower-zone messages first (they have smaller virtual finish times).

    Zone 0 has cost 1 and zone 5 has cost 6, so the zone-0 message always
    gets a smaller virtual finish time and is dispatched first.
    """
    from assessment.baselines import WFQScheduler

    scheduler = WFQScheduler(service_rate=10.0, scheduler_seed=42)
    result = scheduler.schedule(
        arrival_times=[0.0, 0.0],
        device_ids=["low_zone", "high_zone"],
        zone_priorities=[0, 5],
        is_alarm=[False, False],
    )
    # Zone 0 (idx 0) should have lower waiting time than zone 5 (idx 1)
    assert result.waiting_times[0] < result.waiting_times[1]


def test_wfq_no_alarm_priority():
    """
    Pure geographic WFQ ignores is_alarm: a zone-0 routine message is dispatched
    before a zone-5 alarm because cost(zone=0)=1 < cost(zone=5)=6.

    This demonstrates the priority-inversion failure mode that TRIAGE/4's
    semantic override (component A) is designed to correct.
    """
    from assessment.baselines import WFQScheduler

    scheduler = WFQScheduler(service_rate=10.0, scheduler_seed=42)
    result = scheduler.schedule(
        arrival_times=[0.0, 0.0],
        device_ids=["alarm_dev", "routine_dev"],
        zone_priorities=[5, 0],   # alarm from low-priority zone; routine from zone 0
        is_alarm=[True, False],
    )
    # Zone-0 routine (idx 1, cost=1) wins the virtual-clock race over zone-5 alarm (idx 0, cost=6)
    assert result.waiting_times[1] == 0.0
    assert result.waiting_times[0] > 0.0


def test_wfq_interface_compatible():
    """WFQScheduler returns a SchedulerResult with the standard interface."""
    from assessment.baselines import WFQScheduler

    scheduler = WFQScheduler(service_rate=10.0, scheduler_seed=42)
    result = scheduler.schedule(
        arrival_times=[0.0, 0.1, 0.2],
        device_ids=["A", "B", "C"],
        zone_priorities=[0, 2, 1],
        is_alarm=[False, False, True],
    )
    assert result.n_jobs == 3
    assert all(w >= 0 for w in result.waiting_times)


# === DRR Scheduler Tests ===


def test_drr_round_robin_progression():
    """DRR visits each device in round-robin order for simultaneous arrivals."""
    from assessment.baselines import DRRScheduler

    # Three devices all arrive at t=0; each has one message.
    # DRR should serve them in round-robin order: A → B → C.
    scheduler = DRRScheduler(service_rate=100.0, scheduler_seed=0)
    result = scheduler.schedule(
        arrival_times=[0.0, 0.0, 0.0],
        device_ids=["A", "B", "C"],
        zone_priorities=[0, 0, 0],
        is_alarm=[False, False, False],
    )
    # A (idx 0) is served first — no wait
    assert result.waiting_times[0] == 0.0
    # B and C wait (served after A's service time)
    assert result.waiting_times[1] > 0.0
    assert result.waiting_times[2] > result.waiting_times[1]


def test_drr_ignores_alarm_flag():
    """DRR ignores is_alarm — baseline for pure device-level fairness."""
    from assessment.baselines import DRRScheduler

    scheduler = DRRScheduler(service_rate=10.0, scheduler_seed=42)
    # Alarm from a device that arrives second still waits its turn
    result = scheduler.schedule(
        arrival_times=[0.0, 0.0],
        device_ids=["routine_dev", "alarm_dev"],
        zone_priorities=[0, 5],
        is_alarm=[False, True],
    )
    # Both arrive simultaneously; routine (idx 0) is served first by round-robin
    assert result.waiting_times[0] == 0.0
    assert result.waiting_times[1] > 0.0


def test_drr_interface_compatible():
    """DRRScheduler returns a SchedulerResult with the standard interface."""
    from assessment.baselines import DRRScheduler

    scheduler = DRRScheduler(service_rate=10.0, scheduler_seed=42)
    result = scheduler.schedule(
        arrival_times=[0.0, 0.1],
        device_ids=["A", "B"],
        zone_priorities=[0, 1],
        is_alarm=[False, False],
    )
    assert result.n_jobs == 2
    assert all(w >= 0 for w in result.waiting_times)


# === TBP Scheduler Tests ===


def test_tbp_no_alarm_priority():
    """
    TBP ignores is_alarm: a zone-0 routine message (HIGH band) is dispatched before
    a zone-5 alarm message (BACKGROUND band) because geographic routing places the
    alarm in the lowest-priority band.

    This demonstrates the priority-inversion failure mode that TRIAGE/4's semantic
    override (component A) is designed to correct.
    """
    from assessment.baselines import TokenBucketPriorityScheduler

    scheduler = TokenBucketPriorityScheduler(service_rate=10.0, scheduler_seed=42)
    result = scheduler.schedule(
        arrival_times=[0.0, 0.0],
        device_ids=["high_zone", "alarm_dev"],
        zone_priorities=[0, 5],
        is_alarm=[False, True],
    )
    # Zone-0 routine (idx 0) is routed to HIGH band and served first;
    # zone-5 alarm (idx 1) is routed to BACKGROUND band and waits
    assert result.waiting_times[0] == 0.0
    assert result.waiting_times[1] > 0.0


def test_tbp_work_conserving_under_flood():
    """TBP serves all messages without deadlock when a single band is flooded.

    TBP uses TRIAGE/4 default token parameters (fixed; not inheriting scenario
    overrides).  This test confirms work-conserving behavior: all HIGH-band
    messages are eventually served even after the initial token burst is exhausted
    and the scheduler must wait for token refills.
    """
    from assessment.baselines import TokenBucketPriorityScheduler

    scheduler = TokenBucketPriorityScheduler(service_rate=10.0, scheduler_seed=42)
    n_high = 30
    result = scheduler.schedule(
        arrival_times=[0.0] * n_high,
        device_ids=[f"h{i}" for i in range(n_high)],
        zone_priorities=[0] * n_high,
        is_alarm=[False] * n_high,
    )
    # All messages served — no deadlock during token-stall recovery
    assert result.n_jobs == n_high
    assert all(w >= 0 for w in result.waiting_times)


def test_tbp_fifo_within_band():
    """TBP serves messages within a band in FIFO order (no per-device RR)."""
    from assessment.baselines import TokenBucketPriorityScheduler

    # Two devices in HIGH band; A arrives first
    scheduler = TokenBucketPriorityScheduler(service_rate=100.0, scheduler_seed=42)
    result = scheduler.schedule(
        arrival_times=[0.0, 0.0, 0.1],
        device_ids=["A", "B", "A"],
        zone_priorities=[0, 0, 0],
        is_alarm=[False, False, False],
    )
    # A (idx 0) arrives first among simultaneous msgs — served first
    assert result.waiting_times[0] <= result.waiting_times[1]


def test_tbp_interface_compatible():
    """TokenBucketPriorityScheduler returns a SchedulerResult with the standard interface."""
    from assessment.baselines import TokenBucketPriorityScheduler

    scheduler = TokenBucketPriorityScheduler(service_rate=10.0, scheduler_seed=42)
    result = scheduler.schedule(
        arrival_times=[0.0, 0.1, 0.2],
        device_ids=["A", "B", "C"],
        zone_priorities=[0, 3, 5],
        is_alarm=[True, False, False],
    )
    assert result.n_jobs == 3
    assert all(w >= 0 for w in result.waiting_times)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for adaptive alarm protection extensions.
"""

import numpy as np
import pytest

from triage4 import (
    AlarmRateMonitor,
    AdaptiveTokenBucket,
    TRIAGE4Config,
    TRIAGE4Scheduler,
    SourceAwareQueue,
    SourceRateLimiter,
)
from assessment.workloads import (
    generate_alarm_flood_attack,
    generate_legit_extreme_emergency,
)


def test_alarm_rate_monitor_hysteresis():
    monitor = AlarmRateMonitor(
        window_duration=1.0,
        abnormal_threshold=2.0,
        deactivation_threshold=1.0,
        min_observations=1,
    )

    monitor.record_arrival(0.0, "z0")
    monitor.record_arrival(0.1, "z0")
    monitor.record_arrival(0.2, "z1")

    assert monitor.is_abnormal(0.2)  # 3 alarms in 1s

    # Move time forward so rate decays below deactivation threshold
    assert monitor.is_recovered(2.0)


def test_adaptive_token_bucket_activation_and_consumption():
    bucket = AdaptiveTokenBucket(budget=2, period=1.0, burst_capacity=2)
    assert not bucket.active
    assert bucket.consume(0.0)  # Inactive pass-through

    bucket.activate(0.0)
    assert bucket.active

    # Two tokens available, third should fail
    assert bucket.consume(0.0)
    assert bucket.consume(0.0)
    assert not bucket.consume(0.0)

    # After refill, tokens available again
    assert bucket.consume(1.1)


def test_source_aware_queue_zone_round_robin():
    queue = SourceAwareQueue()
    queue.enqueue(0, "dA", 0)
    queue.enqueue(1, "dB", 1)
    queue.enqueue(2, "dC", 0)

    # Expect zone 0 -> 1 -> 0 ordering
    assert queue.dequeue() == 0
    assert queue.dequeue() == 1
    assert queue.dequeue() == 2
    assert queue.dequeue() is None


def test_scheduler_drops_alarms_under_attack():
    config = TRIAGE4Config(
        enable_alarm_protection=True,
        alarm_window_duration=1.0,
        alarm_abnormal_threshold=1.0,
        alarm_deactivation_threshold=0.5,
        alarm_min_observations=1,
        alarm_limit_budget=2,
        alarm_limit_period=1.0,
        alarm_burst_capacity=2,
        high_token_budget=100,
        standard_token_budget=100,
        background_token_budget=100,
        service_rate=50.0,
    )
    scheduler = TRIAGE4Scheduler(config, scheduler_seed=0)

    # 10 alarms at t=0 exceed budget (2 allowed, 8 dropped)
    n = 10
    arrival_times = [0.0] * n
    device_ids = [f"attacker_{i}" for i in range(n)]
    zone_priorities = [5] * n
    is_alarm = [True] * n

    result = scheduler.schedule(arrival_times, device_ids, zone_priorities, is_alarm)

    assert result.metadata["alarm_dropped"] == 8
    # Only two jobs should have non-zero waiting/e2e (served alarms)
    served = np.count_nonzero(result.e2e_times)
    assert served == 2


def _source_limiter() -> SourceRateLimiter:
    """Limiter tuned so two arrivals inside a 1s window mark a source abnormal."""
    return SourceRateLimiter(
        window_duration=1.0,
        abnormal_threshold=2.0,
        deactivation_threshold=1.0,
        min_observations=2,
        limit_budget=1,
        limit_period=1.0,
        burst_capacity=1,
    )


def test_source_rate_limiter_spares_well_behaved_sources():
    """An abnormal source is limited without touching its quiet neighbours."""
    limiter = _source_limiter()

    chatty = [limiter.admit(0.0, "chatty") for _ in range(5)]
    assert limiter.is_limited("chatty")
    assert chatty.count(False) == 3  # limited, not silenced

    # One alarm every 2s never reaches the per-source threshold.
    quiet = [limiter.admit(t, "quiet") for t in (0.0, 2.0, 4.0)]
    assert all(quiet)
    assert not limiter.is_limited("quiet")
    assert limiter.limited_sources == 1


def test_first_alarm_from_a_source_is_always_admitted():
    """
    The alarm event itself always lands; only repeats can ever be shed.

    Activation grants a full budget, so even a source that is marked abnormal on
    its very first arrival still delivers that arrival. A sensor retransmitting
    until acknowledged therefore loses only duplicates of an alarm already
    delivered, never the alarm itself.
    """
    limiter = SourceRateLimiter(
        window_duration=1.0,
        abnormal_threshold=1.0,
        deactivation_threshold=0.5,
        min_observations=1,  # tightest case: the first arrival can trip the limit
        limit_budget=1,
        limit_period=1.0,
        burst_capacity=1,
    )

    assert limiter.admit(0.0, "retransmitter")
    assert limiter.is_limited("retransmitter")


def test_limited_source_keeps_a_residual_channel():
    """A failing device is throttled, not silenced: its signal must survive."""
    limiter = _source_limiter()

    for _ in range(5):
        limiter.admit(0.0, "failing")
    assert limiter.is_limited("failing")

    # Still flooding in the next period: the budget lands, the excess is shed.
    later = [limiter.admit(1.0 + i * 0.01, "failing") for i in range(5)]
    assert limiter.is_limited("failing")
    assert any(later), "a limited source must retain a residual channel"


def test_source_limiter_releases_on_recovery():
    """Hysteresis releases a source once its own rate subsides."""
    limiter = _source_limiter()

    for _ in range(5):
        limiter.admit(0.0, "recovering")
    assert limiter.is_limited("recovering")

    # A lone arrival a full window later leaves the source below deactivation.
    assert limiter.admit(5.0, "recovering")
    assert not limiter.is_limited("recovering")
    assert limiter.deactivations == 1


def test_legitimate_alarm_survives_a_flood_from_another_source():
    """
    Reviewer #1, Concern #3: a flood must not cost a legitimate source its alarm.

    The global backstop is configured out of reach so the per-source layer is
    what is under test.
    """
    config = TRIAGE4Config(
        enable_alarm_protection=True,
        alarm_window_duration=1.0,
        alarm_abnormal_threshold=1000.0,
        alarm_deactivation_threshold=999.0,
        alarm_min_observations=2,
        alarm_limit_budget=1000,
        alarm_limit_period=1.0,
        alarm_burst_capacity=1000,
        alarm_source_abnormal_threshold=2.0,
        alarm_source_deactivation_threshold=1.0,
        alarm_source_limit_budget=1,
        alarm_source_limit_period=1.0,
        alarm_source_burst_capacity=1,
        high_token_budget=100,
        standard_token_budget=100,
        background_token_budget=100,
        service_rate=50.0,
    )
    scheduler = TRIAGE4Scheduler(config, scheduler_seed=0)

    n_attack = 20
    arrival_times = [0.0] * (n_attack + 1)
    device_ids = ["attacker"] * n_attack + ["legit_zone2_sensor"]
    zone_priorities = [5] * n_attack + [2]
    is_alarm = [True] * (n_attack + 1)

    result = scheduler.schedule(arrival_times, device_ids, zone_priorities, is_alarm)

    legit_idx = n_attack
    assert result.e2e_times[legit_idx] > 0, "legitimate alarm was shed during the flood"
    # Attacker keeps its first arrival plus one budgeted token; the rest are shed.
    assert result.metadata["alarm_dropped"] == n_attack - 2


def test_false_positive_rate_formal_validation():
    """
    Criterion 2: False Positive Rate < 5%

    Formally validate that legitimate alarm traffic is not dropped
    under extreme but realistic multi-zone emergency conditions.
    """
    n_runs = 30
    total_alarms = 0
    total_drops = 0

    for seed in range(999, 999 + n_runs):
        workload = generate_legit_extreme_emergency(
            zones=10,
            alarms_per_zone=3,
            duration=30.0,
            background_load=0.3,
            service_rate=20.0,
        )

        config = TRIAGE4Config(
            enable_alarm_protection=True,
            alarm_window_duration=10.0,
            alarm_abnormal_threshold=5.0,
            alarm_limit_budget=15,
            alarm_limit_period=1.0,
            service_rate=20.0,
        )

        scheduler = TRIAGE4Scheduler(config, scheduler_seed=seed)
        result = scheduler.schedule(
            arrival_times=workload.arrival_times,
            device_ids=workload.device_ids,
            zone_priorities=workload.zone_priorities,
            is_alarm=workload.is_alarm,
        )

        n_alarms = np.sum(workload.is_alarm)
        n_dropped = result.metadata.get("alarm_dropped", 0)

        total_alarms += n_alarms
        total_drops += n_dropped

    fpr = (total_drops / total_alarms) * 100 if total_alarms > 0 else 0.0

    print(f"\n{'='*60}")
    print("FALSE POSITIVE RATE VALIDATION (Criterion 2)")
    print(f"{'='*60}")
    print(f"Total alarms across {n_runs} runs: {total_alarms}")
    print(f"Total dropped: {total_drops}")
    print(f"FPR: {fpr:.2f}%")
    print("Target: < 5%")
    print(f"Status: {'PASS' if fpr < 5.0 else 'FAIL'}")
    print(f"{'='*60}\n")

    assert fpr < 5.0, f"FPR {fpr:.2f}% exceeds 5% threshold"
    assert fpr == 0.0, "Expected zero drops for legitimate extreme emergency"

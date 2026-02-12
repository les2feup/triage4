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

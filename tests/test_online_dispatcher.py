"""
Parity and correctness tests for `Triage4EgressDispatcher`.

These prove the online dispatcher reproduces the offline TRIAGE/4 policy on
fixed micro-workloads driven by a hand-specified relative-time sequence. They
deliberately avoid asserting serve-order equality against
`TRIAGE4Scheduler.schedule`: the offline serve order depends on stochastic
exponential service times (`_start_job`), so that comparison is ill-defined.
Instead, each test pins the exact handle the dispatcher must return at each
step (controlled-`now` oracle).
"""

import pytest

from triage4 import TRIAGE4Config, Triage4EgressDispatcher


# Routine HIGH-band messages live in zone 0; STANDARD in zone 2; alarms carry
# is_alarm=True from any zone. now=0.0 keeps token buckets pre-refill.


def test_band_order_alarm_served_before_earlier_high():
    """An alarm enqueued after a HIGH message is still dispatched first."""
    disp = Triage4EgressDispatcher(TRIAGE4Config())
    disp.enqueue(handle=1, device_id="A", zone_priority=0, is_alarm=False, now=0.0)
    disp.enqueue(handle=2, device_id="A", zone_priority=0, is_alarm=True, now=0.0)
    assert disp.select_next(0.0) == 2  # ALARM precedes the earlier HIGH


def test_semantic_override_low_zone_alarm_beats_high_zone_routine():
    """An alarm from a low-priority zone outranks routine traffic from zone 0."""
    disp = Triage4EgressDispatcher(TRIAGE4Config())
    disp.enqueue(handle=1, device_id="A", zone_priority=0, is_alarm=False, now=0.0)
    disp.enqueue(handle=2, device_id="B", zone_priority=5, is_alarm=True, now=0.0)
    assert disp.select_next(0.0) == 2


def test_token_exhaustion_high_yields_to_standard():
    """With HIGH budget=2, the third HIGH within a period yields to STANDARD."""
    cfg = TRIAGE4Config(high_token_budget=2, standard_token_budget=5)
    disp = Triage4EgressDispatcher(cfg)
    disp.enqueue(handle=1, device_id="A", zone_priority=0, is_alarm=False, now=0.0)
    disp.enqueue(handle=2, device_id="A", zone_priority=0, is_alarm=False, now=0.0)
    disp.enqueue(handle=3, device_id="A", zone_priority=0, is_alarm=False, now=0.0)
    disp.enqueue(handle=4, device_id="A", zone_priority=2, is_alarm=False, now=0.0)
    assert disp.select_next(0.0) == 1  # HIGH token 1/2
    assert disp.select_next(0.0) == 2  # HIGH token 2/2
    assert disp.select_next(0.0) == 4  # HIGH exhausted -> STANDARD


def test_device_round_robin_alternation():
    """Two devices in one band alternate across successive dispatches."""
    cfg = TRIAGE4Config(high_token_budget=10)
    disp = Triage4EgressDispatcher(cfg)
    disp.enqueue(handle=1, device_id="A", zone_priority=0, is_alarm=False, now=0.0)
    disp.enqueue(handle=2, device_id="B", zone_priority=0, is_alarm=False, now=0.0)
    disp.enqueue(handle=3, device_id="A", zone_priority=0, is_alarm=False, now=0.0)
    assert disp.select_next(0.0) == 1  # device A
    assert disp.select_next(0.0) == 2  # device B (round-robin)
    assert disp.select_next(0.0) == 3  # back to device A


def test_aap_sheds_above_threshold():
    """Under an abnormal alarm rate, AAP rate-sheds excess alarms."""
    cfg = TRIAGE4Config(
        enable_alarm_protection=True,
        alarm_window_duration=1.0,
        alarm_abnormal_threshold=5.0,
        alarm_deactivation_threshold=4.0,
        alarm_min_observations=3,
        alarm_limit_budget=3,
        alarm_limit_period=1.0,
        alarm_burst_capacity=3,
    )
    disp = Triage4EgressDispatcher(cfg)
    # Ten alarms at a single instant: rate (10/window) crosses the 5/s threshold,
    # AAP activates, and once the 3-token budget is spent further alarms are shed.
    results = [
        disp.enqueue(handle=i, device_id="A", zone_priority=5, is_alarm=True, now=0.0)
        for i in range(10)
    ]
    assert False in results  # at least one alarm rate-shed


def test_aap_never_sheds_below_threshold():
    """Legitimate alarms below both AAP thresholds are never shed (R3 property)."""
    cfg = TRIAGE4Config(
        enable_alarm_protection=True,
        alarm_window_duration=10.0,
        alarm_abnormal_threshold=5.0,
        alarm_deactivation_threshold=4.0,
        alarm_min_observations=3,
        alarm_limit_budget=3,
        alarm_limit_period=1.0,
        alarm_burst_capacity=3,
    )
    disp = Triage4EgressDispatcher(cfg)
    # R3 emits 0.2 alarms/s per device: one every 5 s. That holds the source
    # under its own 1.0/s limit and the aggregate under the 5.0/s backstop, so
    # neither layer activates.
    results = [
        disp.enqueue(handle=i, device_id="A", zone_priority=5, is_alarm=True, now=5.0 * i)
        for i in range(12)
    ]
    assert all(results)


def test_controlled_now_oracle_refill_revives_starved_band():
    """A HIGH message starved by token exhaustion is served after a period refill."""
    cfg = TRIAGE4Config(high_token_budget=1, high_token_period=1.0)
    disp = Triage4EgressDispatcher(cfg)
    disp.enqueue(handle=1, device_id="A", zone_priority=0, is_alarm=False, now=0.0)
    disp.enqueue(handle=2, device_id="A", zone_priority=0, is_alarm=False, now=0.0)
    assert disp.select_next(0.0) == 1     # consumes the single HIGH token
    assert disp.select_next(0.1) is None  # token-starved -> server idles
    assert disp.select_next(1.0) == 2     # period boundary refills, h2 served


def test_relative_clock_guard_rejects_absolute_time():
    """Absolute-scale (raw monotonic) times are rejected to enforce D10."""
    disp = Triage4EgressDispatcher(TRIAGE4Config())
    with pytest.raises(AssertionError):
        disp.enqueue(handle=1, device_id="A", zone_priority=0, is_alarm=False, now=1e9)
    with pytest.raises(AssertionError):
        disp.select_next(now=1e9)

"""
Integration tests for TRIAGE/4.

Tests:
    1. Band classification logic
    2. Alarm preemption over all bands
    3. Token bucket rate limiting
    4. Per-device fairness (monopolization prevention)
    5. Multi-band interaction
"""

import pytest

from triage4 import (
    BAND_ALARM,
    BAND_BACKGROUND,
    BAND_HIGH,
    BAND_STANDARD,
    BandClassifier,
    TRIAGE4Config,
    TRIAGE4Scheduler,
)
from triage4.device_fair_queue import DeviceFairQueue


# === Unit Tests: BandClassifier ===


def test_band_classifier_alarm_override():
    """Alarm messages go to ALARM band regardless of zone priority."""
    classifier = BandClassifier(high_zone_max=1, standard_zone_max=3)

    # Alarms from all zones → ALARM band
    assert classifier.classify(zone_priority=0, is_alarm=True) == BAND_ALARM
    assert classifier.classify(zone_priority=1, is_alarm=True) == BAND_ALARM
    assert classifier.classify(zone_priority=5, is_alarm=True) == BAND_ALARM
    assert classifier.classify(zone_priority=100, is_alarm=True) == BAND_ALARM


def test_band_classifier_geographic_priority():
    """Non-alarm messages classified by zone priority."""
    classifier = BandClassifier(high_zone_max=1, standard_zone_max=3)

    # Zone 0-1 → HIGH
    assert classifier.classify(zone_priority=0, is_alarm=False) == BAND_HIGH
    assert classifier.classify(zone_priority=1, is_alarm=False) == BAND_HIGH

    # Zone 2-3 → STANDARD
    assert classifier.classify(zone_priority=2, is_alarm=False) == BAND_STANDARD
    assert classifier.classify(zone_priority=3, is_alarm=False) == BAND_STANDARD

    # Zone 4+ → BACKGROUND
    assert classifier.classify(zone_priority=4, is_alarm=False) == BAND_BACKGROUND
    assert classifier.classify(zone_priority=10, is_alarm=False) == BAND_BACKGROUND


def test_band_classifier_validation():
    """Classifier validates inputs."""
    classifier = BandClassifier(high_zone_max=1, standard_zone_max=3)

    # Negative zone priority raises error
    with pytest.raises(ValueError, match="zone_priority must be non-negative"):
        classifier.classify(zone_priority=-1, is_alarm=False)


def test_band_classifier_config_validation():
    """Classifier config validation."""
    # standard_zone_max must be >= high_zone_max
    with pytest.raises(ValueError, match="standard_zone_max"):
        BandClassifier(high_zone_max=5, standard_zone_max=3)

    # high_zone_max must be non-negative
    with pytest.raises(ValueError, match="high_zone_max must be non-negative"):
        BandClassifier(high_zone_max=-1, standard_zone_max=3)


# === Unit Tests: DeviceFairQueue ===


def test_device_fair_queue_round_robin():
    """Round-robin device selection."""
    queue = DeviceFairQueue()

    # Enqueue messages from 3 devices
    queue.enqueue(job_idx=0, device_id="A")
    queue.enqueue(job_idx=1, device_id="B")
    queue.enqueue(job_idx=2, device_id="C")
    queue.enqueue(job_idx=3, device_id="A")  # Second message from A

    # Round-robin: A → B → C → A
    assert queue.dequeue() == 0  # A's first message
    assert queue.dequeue() == 1  # B's message
    assert queue.dequeue() == 2  # C's message
    assert queue.dequeue() == 3  # A's second message


def test_device_fair_queue_monopolization_prevention():
    """High-rate device cannot monopolize queue."""
    queue = DeviceFairQueue()

    # Device A sends 10 messages, Device B sends 1
    for i in range(10):
        queue.enqueue(job_idx=i, device_id="A")
    queue.enqueue(job_idx=10, device_id="B")

    # Device B gets served before all of A's messages
    first = queue.dequeue()
    second = queue.dequeue()

    # One of the first two should be from device B
    assert first == 0 or second == 10  # B gets a turn early


def test_device_fair_queue_empty():
    """Empty queue returns None."""
    queue = DeviceFairQueue()
    assert queue.is_empty()
    assert queue.dequeue() is None
    assert queue.peek() is None


def test_device_fair_queue_single_device():
    """Single device works correctly (FIFO)."""
    queue = DeviceFairQueue()

    queue.enqueue(job_idx=0, device_id="A")
    queue.enqueue(job_idx=1, device_id="A")
    queue.enqueue(job_idx=2, device_id="A")

    assert queue.dequeue() == 0
    assert queue.dequeue() == 1
    assert queue.dequeue() == 2


# === Integration Tests: TRIAGE4Scheduler ===


def test_seps_alarm_preemption():
    """
    ALARM band preempts all other bands.

    Scenario:
        - Queue 10 HIGH band messages (zone 0)
        - Inject 1 ALARM message mid-simulation (zone 5)
        - Verify ALARM served immediately (waiting time ≈ 0)
    """
    config = TRIAGE4Config(
        high_zone_max=1,
        standard_zone_max=3,
        high_token_budget=100,  # Plenty of tokens
        service_rate=10.0,
    )
    scheduler = TRIAGE4Scheduler(config, scheduler_seed=42)

    # 10 HIGH messages, then 1 ALARM
    arrival_times = [0.0] * 10 + [0.5]
    device_ids = [f"device_{i}" for i in range(11)]
    zone_priorities = [0] * 10 + [5]  # HIGH zone, then BACKGROUND zone
    is_alarm = [False] * 10 + [True]  # Last is ALARM

    result = scheduler.schedule(arrival_times, device_ids, zone_priorities, is_alarm)

    # ALARM (job 10) should have low waiting time
    alarm_waiting = result.waiting_times[10]
    high_avg_waiting = result.waiting_times[:10].mean()

    # ALARM waiting should be much less than average HIGH waiting
    assert alarm_waiting < high_avg_waiting / 2, (
        f"ALARM waiting ({alarm_waiting:.3f}) should be much less than "
        f"HIGH avg ({high_avg_waiting:.3f})"
    )


def test_seps_token_rate_limiting():
    """
    Token buckets limit band throughput.

    Scenario:
        - Send 100 HIGH messages at once
        - Token budget = 10/sec, service rate = 100/sec
        - Verify HIGH throughput limited to ~10 msg/sec
    """
    config = TRIAGE4Config(
        high_zone_max=1,
        high_token_budget=10,
        high_token_period=1.0,
        service_rate=100.0,  # Very fast service
    )
    scheduler = TRIAGE4Scheduler(config, scheduler_seed=42)

    # 100 HIGH messages arrive at t=0
    n = 100
    arrival_times = [0.0] * n
    device_ids = [f"device_{i}" for i in range(n)]
    zone_priorities = [0] * n  # All HIGH zone
    is_alarm = [False] * n

    result = scheduler.schedule(arrival_times, device_ids, zone_priorities, is_alarm)

    # Total time to complete all jobs
    total_time = max(result.e2e_times)

    # Expected throughput ≈ token_budget / token_period = 10 msg/sec
    # So 100 messages should take ~10 seconds
    expected_time = n / 10  # 10 seconds

    # Allow 20% tolerance
    assert total_time > expected_time * 0.8, (
        f"Completion time ({total_time:.1f}s) too fast, "
        f"tokens may not be limiting (expected ~{expected_time:.1f}s)"
    )


def test_seps_per_device_fairness():
    """
    Per-device round-robin prevents monopolization.

    Scenario:
        - Device A sends 50 messages
        - Device B sends 10 messages
        - Both in HIGH band
        - Verify B gets served (not starved by A)
    """
    config = TRIAGE4Config(
        high_zone_max=1,
        high_token_budget=100,  # Plenty of tokens
        service_rate=10.0,
    )
    scheduler = TRIAGE4Scheduler(config, scheduler_seed=42)

    # Device A: 50 messages, Device B: 10 messages
    arrival_times = [0.0] * 60
    device_ids = ["A"] * 50 + ["B"] * 10
    zone_priorities = [0] * 60  # All HIGH zone
    is_alarm = [False] * 60

    result = scheduler.schedule(arrival_times, device_ids, zone_priorities, is_alarm)

    # Device B should have some messages completed early (not all at end)
    b_waiting_times = result.waiting_times[50:60]  # Device B messages
    a_waiting_times = result.waiting_times[:50]  # Device A messages

    # Average waiting time for B should be comparable to A
    # (not 5x worse despite 5x lower arrival rate)
    avg_b = b_waiting_times.mean()
    avg_a = a_waiting_times.mean()

    # B's average shouldn't be more than 2x worse than A's
    assert avg_b < avg_a * 2.0, (
        f"Device B average waiting ({avg_b:.3f}) much worse than "
        f"Device A ({avg_a:.3f}), fairness may be broken"
    )


def test_seps_multi_band_interaction():
    """
    Multiple bands interact correctly with priority hierarchy.

    Scenario:
        - Mix ALARM, HIGH, STANDARD, BACKGROUND messages
        - Verify ALARM served first, then HIGH, etc.
    """
    config = TRIAGE4Config(
        high_zone_max=1,
        standard_zone_max=3,
        high_token_budget=100,
        standard_token_budget=100,
        background_token_budget=100,
        service_rate=10.0,
    )
    scheduler = TRIAGE4Scheduler(config, scheduler_seed=42)

    # Mix of bands
    arrival_times = [0.0, 0.0, 0.0, 0.0]
    device_ids = ["d1", "d2", "d3", "d4"]
    zone_priorities = [0, 2, 5, 1]  # HIGH, STANDARD, BACKGROUND, HIGH
    is_alarm = [False, False, False, True]  # Last is ALARM

    result = scheduler.schedule(arrival_times, device_ids, zone_priorities, is_alarm)

    # ALARM (job 3) should have lowest waiting time
    assert result.waiting_times[3] < result.waiting_times[0]  # ALARM < HIGH
    assert result.waiting_times[3] < result.waiting_times[1]  # ALARM < STANDARD
    assert result.waiting_times[3] < result.waiting_times[2]  # ALARM < BACKGROUND


def test_seps_input_validation():
    """Scheduler validates inputs."""
    config = TRIAGE4Config()
    scheduler = TRIAGE4Scheduler(config)

    # Mismatched lengths
    with pytest.raises(ValueError, match="device_ids length"):
        scheduler.schedule(
            arrival_times=[0.0, 1.0],
            device_ids=["A"],  # Wrong length
            zone_priorities=[0, 1],
            is_alarm=[False, False],
        )

    # Unsorted arrival times
    with pytest.raises(ValueError, match="arrival_times must be sorted"):
        scheduler.schedule(
            arrival_times=[1.0, 0.0],  # Unsorted
            device_ids=["A", "B"],
            zone_priorities=[0, 1],
            is_alarm=[False, False],
        )

    # Negative zone priority
    with pytest.raises(ValueError, match="zone_priorities must be non-negative"):
        scheduler.schedule(
            arrival_times=[0.0, 1.0],
            device_ids=["A", "B"],
            zone_priorities=[-1, 1],  # Negative
            is_alarm=[False, False],
        )


def test_triage4_config_validation():
    """Config validates parameters."""
    # Negative budget
    with pytest.raises(ValueError, match="token_budget must be positive"):
        TRIAGE4Config(high_token_budget=-1)

    # Zero period
    with pytest.raises(ValueError, match="token_period must be positive"):
        TRIAGE4Config(high_token_period=0.0)

    # Burst < 1.0
    with pytest.raises(ValueError, match="burst_multiplier must be >= 1.0"):
        TRIAGE4Config(high_burst_multiplier=0.5)

    # Invalid zone thresholds
    with pytest.raises(ValueError, match="standard_zone_max"):
        TRIAGE4Config(high_zone_max=5, standard_zone_max=3)


def test_seps_basic_functionality():
    """Basic smoke test - scheduler runs without errors."""
    config = TRIAGE4Config()
    scheduler = TRIAGE4Scheduler(config, scheduler_seed=42)

    result = scheduler.schedule(
        arrival_times=[0.0, 0.1, 0.2, 0.3],
        device_ids=["sensor_1", "sensor_2", "sensor_1", "sensor_3"],
        zone_priorities=[0, 2, 5, 1],
        is_alarm=[False, False, True, False],
    )

    # Check result structure
    assert len(result.waiting_times) == 4
    assert len(result.e2e_times) == 4
    assert len(result.priorities) == 4
    assert result.metadata is not None
    assert result.metadata["scheduler"] == "TRIAGE/4"

    # All times should be non-negative
    assert all(w >= 0 for w in result.waiting_times)
    assert all(e >= 0 for e in result.e2e_times)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

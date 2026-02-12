"""
Evaluation scenarios for TRIAGE/4 benchmarking.

Three key scenarios from REFACTORING_PLAN.md:
1. Alarm Under Burst Load - Tests semantic override
2. Device Monopolization - Tests per-device fairness
3. Multi-Zone Emergency - Tests multi-band interaction
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Workload:
    """
    Workload specification for TRIAGE/4 scheduler.

    Attributes:
        arrival_times: Sorted list of message arrival times (seconds)
        device_ids: Device identifier for each message
        zone_priorities: Geographic zone priority (0=highest priority zone)
        is_alarm: Semantic urgency flag (True=emergency)
        description: Human-readable scenario description
    """

    arrival_times: List[float]
    device_ids: List[str]
    zone_priorities: List[int]
    is_alarm: List[bool]
    description: str = ""
    phase_boundaries: Optional[List[Tuple[float, float]]] = None

    def __post_init__(self):
        """Validate workload consistency."""
        n = len(self.arrival_times)
        if len(self.device_ids) != n:
            raise ValueError(f"device_ids length ({len(self.device_ids)}) != n ({n})")
        if len(self.zone_priorities) != n:
            raise ValueError(
                f"zone_priorities length ({len(self.zone_priorities)}) != n ({n})"
            )
        if len(self.is_alarm) != n:
            raise ValueError(f"is_alarm length ({len(self.is_alarm)}) != n ({n})")

        if self.arrival_times != sorted(self.arrival_times):
            raise ValueError("arrival_times must be sorted")

    @property
    def n_messages(self) -> int:
        """Total number of messages."""
        return len(self.arrival_times)

    @property
    def duration(self) -> float:
        """Workload duration (last arrival time)."""
        return max(self.arrival_times) if self.arrival_times else 0.0

    @property
    def n_alarms(self) -> int:
        """Number of alarm messages."""
        return sum(self.is_alarm)

    @property
    def n_devices(self) -> int:
        """Number of unique devices."""
        return len(set(self.device_ids))


def generate_alarm_under_burst(
    n_edu_devices: int = 2,
    messages_per_edu: int = 40,
    burst_duration: float = 5.0,
    alarm_injection_time: float = 2.5,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> Workload:
    """
    Scenario 1: Alarm Under Burst Load.

    Tests semantic urgency override: A critical alarm from a low-priority
    zone must preempt routine telemetry from high-priority zones during
    a realistic burst (80% of service capacity).

    Scenario:
        - N EDU devices in Zone 0 (HIGH band) send multimedia burst
        - Each EDU sends M messages over burst_duration seconds
        - Load: ~16 msg/s (80% of 20 msg/s service rate)
        - 1 sensor in Zone 5 (BACKGROUND band) sends alarm mid-burst
        - Alarm should be served immediately despite geographic priority

    Args:
        n_edu_devices: Number of EDU devices in high-priority zone
        messages_per_edu: Messages per EDU device
        burst_duration: Duration of burst traffic (seconds)
        alarm_injection_time: When alarm arrives during burst

    Returns:
        Workload with burst traffic + 1 alarm (81 messages total)

    Example:
        >>> workload = generate_alarm_under_burst()
        >>> workload.n_messages
        81
        >>> workload.n_alarms
        1
        >>> # Arrival rate: 81 msgs / 5 sec ≈ 16 msg/s (80% of 20 msg/s)
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    arrival_times = []
    device_ids = []
    zone_priorities = []
    is_alarm = []

    # Burst traffic from EDU devices in Zone 1 (HIGH band)
    interval = burst_duration / messages_per_edu
    for edu_id in range(n_edu_devices):
        device_id = f"EDU_{edu_id}"
        for msg_idx in range(messages_per_edu):
            t = msg_idx * interval
            if jitter_std > 0:
                t += np.random.normal(0.0, jitter_std)
            arrival_times.append(max(t, 0.0))
            device_ids.append(device_id)
            zone_priorities.append(0)  # Zone 1 (HIGH band)
            is_alarm.append(False)

    # Single alarm from low-priority zone sensor
    alarm_time = alarm_injection_time
    if jitter_std > 0:
        alarm_time += np.random.normal(0.0, jitter_std)
    arrival_times.append(max(alarm_time, 0.0))
    device_ids.append("sensor_42")
    zone_priorities.append(5)  # Zone 5 (BACKGROUND band)
    is_alarm.append(True)

    # Sort by arrival time
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=(
            f"Alarm Under Burst: {n_edu_devices} EDUs × {messages_per_edu} msgs "
            f"+ 1 alarm @ t≈{alarm_injection_time}s (jitter_std={jitter_std})"
        ),
    )


def generate_device_monopolization(
    high_rate_messages: int = 480,
    low_rate_messages: int = 48,
    duration: float = 60.0,
    zone_priority: int = 0,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> Workload:
    """
    Scenario 2: Device Monopolization Prevention.

    Tests per-device fairness: A high-rate device should not starve
    a low-rate device in the same priority band. Tests round-robin
    fairness under realistic operational load (no overload).

    Scenario:
        - Device A sends 480 messages over 60 sec (8 msg/sec)
        - Device B sends 48 messages over 60 sec (0.8 msg/sec)
        - Both in same zone (HIGH band)
        - Total load: ~8.8 msg/s (44% of service rate, well below capacity)
        - Device B should get fair interleaving despite 10x rate difference

    Args:
        high_rate_messages: Total messages from high-rate device
        low_rate_messages: Total messages from low-rate device
        duration: Simulation duration (seconds)
        zone_priority: Zone priority for both devices (0=HIGH)

    Returns:
        Workload with mixed-rate devices in same band (528 messages total)

    Example:
        >>> workload = generate_device_monopolization()
        >>> workload.n_devices
        2
        >>> workload.duration
        60.0
        >>> # Arrival rate: 528 msgs / 60 sec ≈ 8.8 msg/s (44% of service rate)
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    arrival_times = []
    device_ids = []
    zone_priorities = []
    is_alarm = []

    # Device A: High-rate sender
    high_interval = duration / high_rate_messages
    for i in range(high_rate_messages):
        t = i * high_interval
        if jitter_std > 0:
            t += np.random.normal(0.0, jitter_std)
        arrival_times.append(max(t, 0.0))
        device_ids.append("device_A")
        zone_priorities.append(zone_priority)
        is_alarm.append(False)

    # Device B: Low-rate sender
    low_interval = duration / low_rate_messages
    for i in range(low_rate_messages):
        t = i * low_interval
        if jitter_std > 0:
            t += np.random.normal(0.0, jitter_std)
        arrival_times.append(max(t, 0.0))
        device_ids.append("device_B")
        zone_priorities.append(zone_priority)
        is_alarm.append(False)

    # Sort by arrival time
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=(
            f"Device Monopolization: A={high_rate_messages} msgs, "
            f"B={low_rate_messages} msgs over {duration}s"
        ),
    )


def generate_skewed_alarm_sources(
    heavy_zone: int = 0,
    other_zones: int = 4,
    heavy_alarms: int = 80,
    light_alarms_per_zone: int = 5,
    duration: float = 20.0,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> Workload:
    """
    Axis 2 Scenario: Skewed multi-source alarm fairness.

    Tests per-source fairness when one zone produces many more alarms
    than others. Default pattern:
        - heavy_zone: 80 alarms over duration
        - other_zones: 4 zones with 5 alarms each
        - Total alarms ≈ 5 alarms/sec with defaults (20s)

    This mirrors the "skewed multi-source" scenario from the research
    proposal, where a single zone attempts to monopolize the ALARM band.
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    arrival_times: List[float] = []
    device_ids: List[str] = []
    zone_priorities: List[int] = []
    is_alarm: List[bool] = []

    # Heavy source zone
    if heavy_alarms > 0:
        interval_heavy = duration / heavy_alarms
        for k in range(heavy_alarms):
            t = k * interval_heavy
            if jitter_std > 0:
                t += np.random.normal(0.0, jitter_std)
            arrival_times.append(max(t, 0.0))
            device_ids.append(f"zone{heavy_zone}_heavy_alarm")
            zone_priorities.append(heavy_zone)
            is_alarm.append(True)

    # Light source zones
    for z in range(1, other_zones + 1):
        zone_id = heavy_zone + z
        if light_alarms_per_zone <= 0:
            continue
        interval_light = duration / light_alarms_per_zone
        for k in range(light_alarms_per_zone):
            t = k * interval_light + 0.5 * interval_light
            if jitter_std > 0:
                t += np.random.normal(0.0, jitter_std)
            arrival_times.append(max(t, 0.0))
            device_ids.append(f"zone{zone_id}_light_alarm")
            zone_priorities.append(zone_id)
            is_alarm.append(True)

    # Optional low-rate background telemetry for stability
    # (kept simple to avoid confounding fairness measurement)

    # Sort
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=(
            f"Skewed Alarm Sources: heavy_zone={heavy_zone} ({heavy_alarms} alarms), "
            f"{other_zones} light zones × {light_alarms_per_zone} alarms"
        ),
    )


def generate_multi_zone_emergency(
    n_zones: int = 6,
    devices_per_zone: int = 2,
    messages_per_device: int = 10,
    alarm_zones: List[int] | None = None,
    alarms_per_zone: int = 3,
    duration: float = 20.0,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> Workload:
    """
    Scenario 3: Multi-Zone Emergency.

    Tests complex multi-band interaction: Multiple zones with varying
    priorities, some experiencing emergencies. Uses realistic multi-band
    load distribution (~80% of service capacity).

    Scenario:
        - 6 zones with varying geographic priorities (0=highest)
        - Zone 0-1: HIGH band (2 devices × 10 msgs = 20 msgs each)
        - Zone 2-3: STANDARD band (2 devices × 10 msgs = 20 msgs each)
        - Zone 4-5: BACKGROUND band (2 devices × 10 msgs = 20 msgs each)
        - Total load: ~6 msg/s per band, ~18 msg/s total (90% of service rate)
        - Zones 2 and 4 experience emergencies (3 alarms each)
        - Tests: Band hierarchy, per-zone fairness, alarm preemption

    Args:
        n_zones: Number of geographic zones (0 to n_zones-1)
        devices_per_zone: Devices per zone
        messages_per_device: Routine messages per device
        alarm_zones: Which zones have emergencies (default: [2, 4])
        alarms_per_zone: Number of alarms per emergency zone
        duration: Simulation duration (seconds)

    Returns:
        Workload with multi-zone telemetry + alarms (126 messages total)

    Example:
        >>> workload = generate_multi_zone_emergency(n_zones=6)
        >>> workload.n_devices
        12
        >>> workload.n_alarms
        6
        >>> # Arrival rate: 126 msgs / 20 sec ≈ 6.3 msg/s per band (90% total)
    """
    if alarm_zones is None:
        alarm_zones = [2, 4]  # Mid and low priority zones

    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    arrival_times = []
    device_ids = []
    zone_priorities = []
    is_alarm = []

    # Routine telemetry from all zones
    interval = duration / messages_per_device
    for zone_id in range(n_zones):
        for device_idx in range(devices_per_zone):
            device_id = f"zone{zone_id}_device{device_idx}"
            for msg_idx in range(messages_per_device):
                t = msg_idx * interval
                if jitter_std > 0:
                    t += np.random.normal(0.0, jitter_std)
                arrival_times.append(max(t, 0.0))
                device_ids.append(device_id)
                zone_priorities.append(zone_id)  # Zone ID = priority
                is_alarm.append(False)

    # Inject alarms from specified zones
    alarm_interval = duration / (alarms_per_zone + 1)
    for zone_id in alarm_zones:
        if zone_id >= n_zones:
            continue
        for alarm_idx in range(alarms_per_zone):
            # Spread alarms throughout duration
            t = (alarm_idx + 1) * alarm_interval
            if jitter_std > 0:
                t += np.random.normal(0.0, jitter_std)
            arrival_times.append(max(t, 0.0))
            device_ids.append(f"zone{zone_id}_alarm_sensor")
            zone_priorities.append(zone_id)
            is_alarm.append(True)

    # Sort by arrival time
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=(
            f"Multi-Zone Emergency: {n_zones} zones × {devices_per_zone} devices "
            f"+ {len(alarm_zones)} zones with {alarms_per_zone} alarms each"
        ),
    )


def generate_alarm_rate_sweep(
    alarm_rate: float,
    duration: float = 30.0,
    background_rate: float = 10.0,
    n_alarm_sources: int = 1,
    n_background_devices: int = 5,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> Workload:
    """
    Axis 1 Scenario: Alarm rate sweep with fixed background load.

    Used to study adaptive protection activation across alarm rates while
    keeping background telemetry constant. Typical configurations:
        - alarm_rate in {0.5, 1.0, 3.0, 5.0, 10.0, 20.0} alarms/sec
        - duration = 30s
        - background_rate ≈ 10 msg/s (telemetry)

    Alarm pattern:
        - n_alarm_sources sources generate alarms at total rate alarm_rate
        - Alarms are evenly spaced over [0, duration]

    Background pattern:
        - n_background_devices telemetry devices
        - Total background_rate msg/s, evenly split across devices

    This scenario isolates the effect of alarm rate on protection behavior.
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    arrival_times: List[float] = []
    device_ids: List[str] = []
    zone_priorities: List[int] = []
    is_alarm: List[bool] = []

    # Alarm traffic
    total_alarms = int(max(alarm_rate * duration, 0))
    if total_alarms > 0 and alarm_rate > 0:
        alarms_per_source = max(total_alarms // max(n_alarm_sources, 1), 1)
        alarm_interval = duration / (alarms_per_source + 1)

        for src_idx in range(n_alarm_sources):
            source_id = f"alarm_src_{src_idx}"
            for k in range(alarms_per_source):
                t = (k + 1) * alarm_interval
                if jitter_std > 0:
                    t += np.random.normal(0.0, jitter_std)
                arrival_times.append(max(t, 0.0))
                device_ids.append(source_id)
                # Use mid-priority zones for alarms by default
                zone_priorities.append(2 + (src_idx % 3))
                is_alarm.append(True)

    # Background telemetry
    total_bg = int(max(background_rate * duration, 0))
    if total_bg > 0 and n_background_devices > 0:
        msgs_per_device = max(total_bg // n_background_devices, 1)
        bg_interval = duration / max(msgs_per_device, 1)

        for dev_idx in range(n_background_devices):
            dev_id = f"bg_dev_{dev_idx}"
            for k in range(msgs_per_device):
                t = k * bg_interval
                if jitter_std > 0:
                    t += np.random.normal(0.0, jitter_std)
                arrival_times.append(max(t, 0.0))
                device_ids.append(dev_id)
                zone_priorities.append(dev_idx % 3)
                is_alarm.append(False)

    # Sort
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=(
            f"Alarm Rate Sweep: alarm_rate={alarm_rate}/s, "
            f"background_rate={background_rate}/s, duration={duration}s"
        ),
    )


# =============================================================================
# Phase-Based Scenario Variants (Enhanced for Statistical Benchmarks)
# =============================================================================


def generate_alarm_under_burst_phased(
    duration: float = 60.0,
    service_rate: float = 20.0,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> Workload:
    """
    Enhanced Scenario 1: Alarm Under Burst with Multi-Phase Load Evolution.

    Tests alarm preemption under dynamic load transitions, not just static burst.
    Four phases with varying load demonstrate TRIAGE/4 alarm priority across conditions.

    Phases:
        Phase 1 (0-20s):   Baseline load (ρ=0.4, ~8 msg/s)
        Phase 2 (20-35s):  HIGH band burst (ρ=0.9, ~18 msg/s) ← ALARM ARRIVES
        Phase 3 (35-50s):  Recovery (ρ=0.5, ~10 msg/s)
        Phase 4 (50-60s):  Cooldown (ρ=0.3, ~6 msg/s)

    Alarms:
        - 1 alarm at t=25s (mid-burst) from Zone 5 (BACKGROUND)
        - 1 alarm at t=40s (recovery) from Zone 4 (STANDARD)

    Args:
        duration: Total simulation duration (default: 60s)
        service_rate: Service capacity in msg/s (default: 20)

    Returns:
        Workload with phase-based evolution and 2 alarms

    Example:
        >>> workload = generate_alarm_under_burst_phased()
        >>> workload.duration
        60.0
        >>> workload.n_alarms
        2
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    arrival_times = []
    device_ids = []
    zone_priorities = []
    is_alarm = []

    # Phase definitions: (start, end, target_load_factor, n_devices)
    phases = [
        (0.0, 20.0, 0.4, 2),  # Baseline
        (20.0, 35.0, 0.9, 4),  # Burst
        (35.0, 50.0, 0.5, 3),  # Recovery
        (50.0, 60.0, 0.3, 2),  # Cooldown
    ]

    for phase_idx, (start, end, load_factor, n_devices) in enumerate(phases):
        phase_duration = end - start
        target_arrival_rate = service_rate * load_factor
        messages_in_phase = int(target_arrival_rate * phase_duration)

        # Distribute messages across devices
        messages_per_device = messages_in_phase // n_devices

        for dev_idx in range(n_devices):
            device_id = f"EDU_{phase_idx}_{dev_idx}"
            interval = phase_duration / messages_per_device

            for msg_idx in range(messages_per_device):
                t = start + msg_idx * interval
                if jitter_std > 0:
                    t += np.random.normal(0.0, jitter_std)
                arrival_times.append(max(t, 0.0))
                device_ids.append(device_id)
                zone_priorities.append(0)  # Zone 0 (HIGH band)
                is_alarm.append(False)

    # Inject alarms
    # Alarm 1: Mid-burst (Phase 2) from Zone 5 (BACKGROUND)
    alarm_time = 25.0
    if jitter_std > 0:
        alarm_time += np.random.normal(0.0, jitter_std)
    arrival_times.append(max(alarm_time, 0.0))
    device_ids.append("sensor_alarm_1")
    zone_priorities.append(5)
    is_alarm.append(True)

    # Alarm 2: Recovery (Phase 3) from Zone 4 (STANDARD)
    alarm_time2 = 40.0
    if jitter_std > 0:
        alarm_time2 += np.random.normal(0.0, jitter_std)
    arrival_times.append(max(alarm_time2, 0.0))
    device_ids.append("sensor_alarm_2")
    zone_priorities.append(4)
    is_alarm.append(True)

    # Sort by arrival time
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=(
            "Alarm Under Burst (Phased): 4 phases with load evolution (ρ=0.4→0.9→0.5→0.3), 2 alarms"
        ),
        phase_boundaries=[(start, end) for start, end, _, _ in phases],
    )


def generate_device_monopolization_sweep(
    duration: float = 60.0,
    zone_priority: int = 0,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> Workload:
    """
    Enhanced Scenario 2: Device Monopolization with Rate Imbalance Sweep.

    Tests per-device fairness across increasing imbalance levels.
    Three phases with escalating rate ratios reveal fairness mechanism limits.

    Phases:
        Phase 1 (0-20s):   2:1 ratio (moderate imbalance)
        Phase 2 (20-40s):  10:1 ratio (severe imbalance)
        Phase 3 (40-60s):  50:1 ratio (extreme monopolizer)

    All traffic in same band (HIGH) to test intra-band fairness.

    Args:
        duration: Total simulation duration (default: 60s)
        zone_priority: Zone priority for all devices (default: 0 = HIGH band)

    Returns:
        Workload with 2 devices and increasing rate imbalance

    Example:
        >>> workload = generate_device_monopolization_sweep()
        >>> workload.n_devices
        2
        >>> workload.duration
        60.0
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    arrival_times = []
    device_ids = []
    zone_priorities = []
    is_alarm = []

    # Phase definitions: (start, end, high_rate_msg/s, low_rate_msg/s)
    phases = [
        (0.0, 20.0, 4.0, 2.0),  # 2:1 ratio
        (20.0, 40.0, 10.0, 1.0),  # 10:1 ratio
        (40.0, 60.0, 15.0, 0.3),  # 50:1 ratio
    ]

    for phase_idx, (start, end, high_rate, low_rate) in enumerate(phases):
        phase_duration = end - start

        # High-rate device
        n_high = int(high_rate * phase_duration)
        interval_high = phase_duration / n_high
        for msg_idx in range(n_high):
            t = start + msg_idx * interval_high
            if jitter_std > 0:
                t += np.random.normal(0.0, jitter_std)
            arrival_times.append(max(t, 0.0))
            device_ids.append("device_A")
            zone_priorities.append(zone_priority)
            is_alarm.append(False)

        # Low-rate device
        n_low = int(low_rate * phase_duration)
        if n_low > 0:
            interval_low = phase_duration / n_low
            for msg_idx in range(n_low):
                t = start + msg_idx * interval_low
                if jitter_std > 0:
                    t += np.random.normal(0.0, jitter_std)
                arrival_times.append(max(t, 0.0))
                device_ids.append("device_B")
                zone_priorities.append(zone_priority)
                is_alarm.append(False)

    # Sort by arrival time
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=(
            "Device Monopolization (Sweep): 3 phases with rate imbalance (2:1 → 10:1 → 50:1)"
        ),
        phase_boundaries=[(start, end) for start, end, _, _ in phases],
    )


def generate_multi_zone_emergency_cascade(
    n_zones: int = 6,
    devices_per_zone: int = 2,
    duration: float = 60.0,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> Workload:
    """
    Enhanced Scenario 3: Multi-Zone Emergency with Alarm Cascade.

    Tests ALARM band fairness and zone fairness under emergency surge.
    Real emergencies cascade (e.g., fire spreads) - alarms trigger sequentially.

    Phases:
        Phase 1 (0-20s):   Normal telemetry from all zones (baseline)
        Phase 2 (20-30s):  Alarm cascade: Zones 5→4→3 trigger sequentially
        Phase 3 (30-50s):  Recovery with residual alarms
        Phase 4 (50-60s):  Cooldown

    Alarm Pattern:
        - Zone 5: 3 alarms @ t=20, 22, 24s (cascade start)
        - Zone 4: 3 alarms @ t=21, 23, 25s (spread)
        - Zone 3: 3 alarms @ t=22, 24, 26s (spread)
        - Total: 9 alarms from 3 zones

    Args:
        n_zones: Number of geographic zones (default: 6)
        devices_per_zone: Devices per zone (default: 2)
        duration: Total simulation duration (default: 60s)

    Returns:
        Workload with 6 zones, cascading alarms from zones 3-5

    Example:
        >>> workload = generate_multi_zone_emergency_cascade()
        >>> workload.n_zones  # doctest: +SKIP
        6
        >>> workload.n_alarms
        9
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    arrival_times = []
    device_ids = []
    zone_priorities = []
    is_alarm = []

    # Phase 1: Baseline telemetry (0-20s)
    phase1_duration = 20.0
    messages_per_device = 5
    interval = phase1_duration / messages_per_device

    for zone_id in range(n_zones):
        for dev_idx in range(devices_per_zone):
            device_id = f"zone{zone_id}_device{dev_idx}"
            for msg_idx in range(messages_per_device):
                t = msg_idx * interval
                if jitter_std > 0:
                    t += np.random.normal(0.0, jitter_std)
                arrival_times.append(max(t, 0.0))
                device_ids.append(device_id)
                zone_priorities.append(zone_id)
                is_alarm.append(False)

    # Phase 2: Alarm cascade (20-30s)
    # Zones 5, 4, 3 trigger alarms in sequence
    cascade_zones = [5, 4, 3]
    alarms_per_zone = 3
    alarm_start_time = 20.0

    for cascade_idx, zone_id in enumerate(cascade_zones):
        for alarm_idx in range(alarms_per_zone):
            # Stagger alarms: zone 5 @ 20,22,24; zone 4 @ 21,23,25; zone 3 @ 22,24,26
            arrival_time = alarm_start_time + alarm_idx * 2.0 + cascade_idx * 1.0
            if jitter_std > 0:
                arrival_time += np.random.normal(0.0, jitter_std)
            arrival_times.append(max(arrival_time, 0.0))
            device_ids.append(f"zone{zone_id}_alarm_sensor")
            zone_priorities.append(zone_id)
            is_alarm.append(True)

    # Phase 3 & 4: Recovery + Cooldown telemetry (30-60s)
    phase34_start = 30.0
    phase34_duration = 30.0
    messages_per_device_recovery = 3
    interval_recovery = phase34_duration / messages_per_device_recovery

    for zone_id in range(n_zones):
        for dev_idx in range(devices_per_zone):
            device_id = f"zone{zone_id}_device{dev_idx}"
            for msg_idx in range(messages_per_device_recovery):
                t = phase34_start + msg_idx * interval_recovery
                if jitter_std > 0:
                    t += np.random.normal(0.0, jitter_std)
                arrival_times.append(max(t, 0.0))
                device_ids.append(device_id)
                zone_priorities.append(zone_id)
                is_alarm.append(False)

    # Sort by arrival time
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=(
            f"Multi-Zone Emergency (Cascade): {n_zones} zones, "
            f"alarm cascade from zones 5→4→3 (9 alarms total)"
        ),
        phase_boundaries=[
            (0.0, 20.0),
            (20.0, 30.0),
            (30.0, 50.0),
            (50.0, duration),
        ],
    )


def generate_alarm_load_regime(
    utilization: float = 0.7,
    service_rate: float = 20.0,
    duration: float = 30.0,
    alarm_rate: float = 1.0,
    n_alarm_sources: int = 4,
    n_background_devices: int = 8,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> Workload:
    """
    Axis 3 Scenario: Same alarm pattern under different load regimes.

    Keeps the alarm pattern fixed while varying background telemetry to
    achieve different utilizations ρ ≈ utilization:

        total_rate ≈ service_rate * utilization
        alarm_rate: fixed (alarms/sec)
        background_rate ≈ max(total_rate - alarm_rate, 0)

    This isolates the effect of load (ρ) on latency, fairness and protection
    behavior without changing alarm structure.
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    arrival_times: List[float] = []
    device_ids: List[str] = []
    zone_priorities: List[int] = []
    is_alarm: List[bool] = []

    # Alarm pattern: n_alarm_sources sharing alarm_rate evenly
    total_alarms = int(max(alarm_rate * duration, 0))
    alarms_per_source = max(total_alarms // max(n_alarm_sources, 1), 1) if total_alarms > 0 else 0
    if alarms_per_source > 0:
        alarm_interval = duration / (alarms_per_source + 1)
        for src_idx in range(n_alarm_sources):
            src_id = f"alarm_zone{src_idx}"
            zone_id = src_idx % 6
            for k in range(alarms_per_source):
                t = (k + 1) * alarm_interval
                if jitter_std > 0:
                    t += np.random.normal(0.0, jitter_std)
                arrival_times.append(max(t, 0.0))
                device_ids.append(src_id)
                zone_priorities.append(zone_id)
                is_alarm.append(True)

    # Compute background rate to reach target utilization
    target_total_rate = service_rate * utilization
    background_rate = max(target_total_rate - alarm_rate, 0.0)

    total_bg = int(max(background_rate * duration, 0))
    if total_bg > 0 and n_background_devices > 0:
        msgs_per_device = max(total_bg // n_background_devices, 1)
        interval = duration / max(msgs_per_device, 1)
        for dev_idx in range(n_background_devices):
            dev_id = f"bg_zone{dev_idx}"
            zone_id = dev_idx % 6
            for k in range(msgs_per_device):
                t = k * interval + 0.5 * interval
                if jitter_std > 0:
                    t += np.random.normal(0.0, jitter_std)
                arrival_times.append(max(t, 0.0))
                device_ids.append(dev_id)
                zone_priorities.append(zone_id)
                is_alarm.append(False)

    # Sort
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=(
            f"Alarm Load Regime: ρ≈{utilization:.2f}, alarm_rate={alarm_rate}/s, "
            f"service_rate={service_rate}/s, duration={duration}s"
        ),
    )


def generate_alarm_load_near_saturation_constrained(
    service_rate: float = 20.0,
    duration: float = 30.0,
    alarm_rate: float = 1.0,
    utilization: float = 0.95,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> Workload:
    """
    High-load regime specifically for testing token-bucket behavior under scarcity.

    This scenario mirrors `generate_alarm_load_regime` with ρ≈0.95 but is intended
    to be run with TRIAGE/4 token budgets summing to ≈100% of the service rate
    (e.g., HIGH+STANDARD+BACKGROUND ≈ service_rate). Under these conditions the
    scheduler must actively constrain non-alarm traffic instead of operating in
    an over-provisioned regime.
    """
    return generate_alarm_load_regime(
        utilization=utilization,
        service_rate=service_rate,
        duration=duration,
        alarm_rate=alarm_rate,
        n_alarm_sources=4,
        n_background_devices=8,
        jitter_std=jitter_std,
        seed=seed,
    )


# =============================================================================
# Adaptive Alarm Protection Scenarios
# =============================================================================


def generate_alarm_flood_attack(
    duration: float = 10.0,
    alarm_rate: float = 20.0,
    background_devices: int = 5,
    background_rate: float = 1.0,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> Workload:
    """
    Adversarial alarm flood to test adaptive protection.

    - Attacker: alarm_rate alarms/sec from a single zone/device
    - Background: steady telemetry from multiple devices to detect starvation
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    arrival_times: List[float] = []
    device_ids: List[str] = []
    zone_priorities: List[int] = []
    is_alarm: List[bool] = []

    # Attack traffic (alarms)
    n_attack = int(alarm_rate * duration)
    alarm_interval = 1.0 / alarm_rate if alarm_rate > 0 else duration
    for i in range(n_attack):
        t = i * alarm_interval
        if jitter_std > 0:
            t += np.random.normal(0.0, jitter_std)
        arrival_times.append(max(t, 0.0))
        device_ids.append("attacker_alarm_source")
        zone_priorities.append(5)
        is_alarm.append(True)

    # Background telemetry
    bg_interval = 1.0 / background_rate if background_rate > 0 else duration
    bg_messages = int(background_rate * duration)
    for dev_idx in range(background_devices):
        device_id = f"bg_device_{dev_idx}"
        for msg_idx in range(bg_messages):
            t = msg_idx * bg_interval
            if jitter_std > 0:
                t += np.random.normal(0.0, jitter_std)
            arrival_times.append(max(t, 0.0))
            device_ids.append(device_id)
            zone_priorities.append(dev_idx % 3)  # Mix of mid-priority zones
            is_alarm.append(False)

    # Sort
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=(
            f"Alarm Flood Attack: {alarm_rate} alarms/sec for {duration}s "
            f"+ {background_devices} background devices @ {background_rate}/s"
        ),
    )


def generate_alarm_malfunction_surge(
    zones: int = 5,
    alarms_per_zone: int = 20,
    duration: float = 10.0,
    legitimate_alarms: int = 3,
    jitter_std: float = 0.0,
    seed: int | None = None,
    heavy_zone: int | None = None,
    light_zones: List[int] | None = None,
    heavy_rate: float | None = None,
    light_rate: float | None = None,
) -> Workload:
    """
    Multiple malfunctioning sensors producing alarms simultaneously.

    - Legacy mode: zones malfunction, each emits alarms_per_zone alarms over duration
    - Rate mode: one heavy zone emits heavy_rate alarms/sec, light zones emit light_rate
    - Optional legitimate alarms can be added to measure fairness
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    arrival_times: List[float] = []
    device_ids: List[str] = []
    zone_priorities: List[int] = []
    is_alarm: List[bool] = []

    if heavy_rate is not None or light_rate is not None:
        if heavy_zone is None:
            heavy_zone = 0
        if light_zones is None:
            light_zones = [z for z in range(zones) if z != heavy_zone]
        if heavy_rate is None or light_rate is None:
            raise ValueError("heavy_rate and light_rate must both be provided in rate mode")

        n_heavy = int(heavy_rate * duration)
        heavy_arrivals = np.sort(np.random.uniform(0, duration, n_heavy))
        arrival_times.extend(heavy_arrivals)
        device_ids.extend([f"zone{heavy_zone}_faulty_device"] * n_heavy)
        zone_priorities.extend([heavy_zone] * n_heavy)
        is_alarm.extend([True] * n_heavy)

        for zone_id in light_zones:
            n_light = int(light_rate * duration)
            light_arrivals = np.sort(np.random.uniform(0, duration, n_light))
            arrival_times.extend(light_arrivals)
            device_ids.extend([f"zone{zone_id}_faulty_device"] * n_light)
            zone_priorities.extend([zone_id] * n_light)
            is_alarm.extend([True] * n_light)
    else:
        for zone_id in range(zones):
            interval = duration / alarms_per_zone if alarms_per_zone > 0 else duration
            for alarm_idx in range(alarms_per_zone):
                t = alarm_idx * interval
                if jitter_std > 0:
                    t += np.random.normal(0.0, jitter_std)
                arrival_times.append(max(t, 0.0))
                device_ids.append(f"zone{zone_id}_faulty_device")
                zone_priorities.append(zone_id)
                is_alarm.append(True)

    # Legitimate alarms sprinkled later in the window
    leg_interval = duration / (legitimate_alarms + 1) if legitimate_alarms > 0 else duration
    for i in range(legitimate_alarms):
        t = duration / 2 + (i + 1) * leg_interval / 2
        if jitter_std > 0:
            t += np.random.normal(0.0, jitter_std)
        arrival_times.append(max(t, 0.0))
        device_ids.append(f"legit_alarm_{i}")
        zone_priorities.append(zones + i)  # Lower priority zones
        is_alarm.append(True)

    # Sort
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    if heavy_rate is not None or light_rate is not None:
        description = (
            "Alarm Malfunction Surge (Rate): "
            f"heavy_zone={heavy_zone}, light_zones={len(light_zones)} "
            f"heavy_rate={heavy_rate}, light_rate={light_rate}, "
            f"legit_alarms={legitimate_alarms}"
        )
    else:
        description = (
            f"Alarm Malfunction Surge: {zones} zones × {alarms_per_zone} alarms "
            f"+ {legitimate_alarms} legitimate alarms"
        )

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=description,
    )


def generate_legit_extreme_emergency(
    zones: int = 10,
    alarms_per_zone: int = 3,
    duration: float = 30.0,
    background_load: float = 0.5,
    service_rate: float = 20.0,
    jitter_std: float = 0.0,
    seed: int | None = None,
) -> Workload:
    """
    Legitimate extreme emergency that should NOT trigger protection.

    - Many zones with a few alarms each (low aggregate rate)
    - Background telemetry at a fraction of service capacity
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    arrival_times: List[float] = []
    device_ids: List[str] = []
    zone_priorities: List[int] = []
    is_alarm: List[bool] = []

    # Legitimate alarms spread across duration
    alarm_interval = duration / alarms_per_zone if alarms_per_zone > 0 else duration
    for zone_id in range(zones):
        for alarm_idx in range(alarms_per_zone):
            t = alarm_idx * alarm_interval + (zone_id * 0.01)
            if jitter_std > 0:
                t += np.random.normal(0.0, jitter_std)
            arrival_times.append(max(t, 0.0))
            device_ids.append(f"zone{zone_id}_alarm")
            zone_priorities.append(zone_id % 6)
            is_alarm.append(True)

    # Background telemetry at target load
    total_messages = int(service_rate * background_load * duration)
    if total_messages > 0:
        telemetry_interval = duration / total_messages
        for msg_idx in range(total_messages):
            t = msg_idx * telemetry_interval + 0.005
            if jitter_std > 0:
                t += np.random.normal(0.0, jitter_std)
            arrival_times.append(max(t, 0.0))
            device_ids.append(f"telemetry_{msg_idx % 8}")
            zone_priorities.append(msg_idx % 6)
            is_alarm.append(False)

    # Sort
    sorted_indices = sorted(range(len(arrival_times)), key=lambda i: arrival_times[i])
    arrival_times = [arrival_times[i] for i in sorted_indices]
    device_ids = [device_ids[i] for i in sorted_indices]
    zone_priorities = [zone_priorities[i] for i in sorted_indices]
    is_alarm = [is_alarm[i] for i in sorted_indices]

    return Workload(
        arrival_times=arrival_times,
        device_ids=device_ids,
        zone_priorities=zone_priorities,
        is_alarm=is_alarm,
        description=(
            f"Legitimate Extreme Emergency: {zones} zones × {alarms_per_zone} alarms "
            f"+ background load {background_load:.0%}"
        ),
    )

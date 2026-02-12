"""
Tiered Resource Allocation for IoT Alarm and Geographic-priority Emergency (TRIAGE/4).

Four-band hierarchical scheduler that resolves priority inversion by
separating semantic urgency (alarms) from geographic priority (zones).

Main components:
    - TRIAGE4Scheduler: Main scheduler with discrete-event simulation
    - TRIAGE4Config: Configuration dataclass with band thresholds and token parameters
    - DeviceFairQueue: Per-device round-robin queue for fairness
    - BandClassifier: Message-to-band classification logic

Band hierarchy:
    0. ALARM: Emergency messages, strict priority
    1. HIGH: High-priority zones, token-constrained
    2. STANDARD: Standard zones, token-constrained
    3. BACKGROUND: Low-priority zones, token-constrained

Example:
    >>> from triage4 import TRIAGE4Scheduler, TRIAGE4Config
    >>> config = TRIAGE4Config(
    ...     high_zone_max=1,
    ...     standard_zone_max=3,
    ...     high_token_budget=10,
    ...     service_rate=20.0
    ... )
    >>> scheduler = TRIAGE4Scheduler(config)
    >>> result = scheduler.schedule(
    ...     arrival_times=[0.0, 0.1, 0.2],
    ...     device_ids=["sensor_1", "sensor_2", "sensor_1"],
    ...     zone_priorities=[0, 5, 2],
    ...     is_alarm=[False, True, False]
    ... )
"""

from .band_classifier import (
    BAND_ALARM,
    BAND_BACKGROUND,
    BAND_HIGH,
    BAND_STANDARD,
    BandClassifier,
)
from .adaptive_token_bucket import AdaptiveTokenBucket
from .alarm_rate_monitor import AlarmRateMonitor
from .device_fair_queue import DeviceFairQueue
from .triage4_config import TRIAGE4Config, create_triage4_custom, create_triage4_default
from .triage4_scheduler import TRIAGE4Scheduler
from .source_aware_queue import SourceAwareQueue

__all__ = [
    # Main scheduler
    "TRIAGE4Scheduler",
    # Configuration
    "TRIAGE4Config",
    "create_triage4_default",
    "create_triage4_custom",
    # Components
    "BandClassifier",
    "DeviceFairQueue",
    "AdaptiveTokenBucket",
    "AlarmRateMonitor",
    "SourceAwareQueue",
    # Band constants
    "BAND_ALARM",
    "BAND_HIGH",
    "BAND_STANDARD",
    "BAND_BACKGROUND",
]

__version__ = "1.0.0"

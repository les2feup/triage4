"""Compatibility exports for legacy ``src.schedulers.triage4`` imports."""

from triage4 import (
    BAND_ALARM,
    BAND_BACKGROUND,
    BAND_HIGH,
    BAND_STANDARD,
    AdaptiveTokenBucket,
    AlarmRateMonitor,
    BandClassifier,
    DeviceFairQueue,
    SourceAwareQueue,
    TRIAGE4Config,
    TRIAGE4Scheduler,
    create_triage4_custom,
    create_triage4_default,
)
from triage4 import __version__

__all__ = [
    "TRIAGE4Scheduler",
    "TRIAGE4Config",
    "create_triage4_default",
    "create_triage4_custom",
    "BandClassifier",
    "DeviceFairQueue",
    "AdaptiveTokenBucket",
    "AlarmRateMonitor",
    "SourceAwareQueue",
    "BAND_ALARM",
    "BAND_HIGH",
    "BAND_STANDARD",
    "BAND_BACKGROUND",
]

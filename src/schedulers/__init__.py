"""
Schedulers module for semantic-priority-scheduler.

Currently contains:
- TRIAGE/4: Four-band hierarchical scheduler
- Base scheduler interface
"""

from .base import Scheduler
from .triage4 import (
    BAND_ALARM,
    BAND_BACKGROUND,
    BAND_HIGH,
    BAND_STANDARD,
    BandClassifier,
    DeviceFairQueue,
    TRIAGE4Config,
    TRIAGE4Scheduler,
    create_triage4_custom,
    create_triage4_default,
)

__all__ = [
    # Base
    "Scheduler",
    # TRIAGE/4
    "TRIAGE4Scheduler",
    "TRIAGE4Config",
    "create_triage4_default",
    "create_triage4_custom",
    "BandClassifier",
    "DeviceFairQueue",
    # Band constants
    "BAND_ALARM",
    "BAND_HIGH",
    "BAND_STANDARD",
    "BAND_BACKGROUND",
]

"""Compatibility wrapper for legacy band classifier imports."""

from triage4.band_classifier import (
    BAND_ALARM,
    BAND_BACKGROUND,
    BAND_HIGH,
    BAND_STANDARD,
    BandClassifier,
)

__all__ = [
    "BAND_ALARM",
    "BAND_HIGH",
    "BAND_STANDARD",
    "BAND_BACKGROUND",
    "BandClassifier",
]

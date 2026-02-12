"""
Band classification logic for TRIAGE/4.

Maps messages to one of four bands based on semantic urgency (is_alarm)
and geographic priority (zone_priority).
"""

# Band constants
BAND_ALARM = 0  # Emergency messages, always served first
BAND_HIGH = 1  # High-priority zone telemetry, token-constrained
BAND_STANDARD = 2  # Standard zone telemetry, token-constrained
BAND_BACKGROUND = 3  # Low-priority zone data, best-effort


class BandClassifier:
    """
    Classifies messages into TRIAGE/4 bands.

    Classification hierarchy:
        1. Semantic urgency (is_alarm) overrides geographic priority
        2. Geographic priority (zone_priority) determines band for non-alarms

    Band assignment rules:
        - is_alarm=True → ALARM (0) - regardless of zone priority
        - zone_priority <= high_zone_max → HIGH (1)
        - zone_priority <= standard_zone_max → STANDARD (2)
        - zone_priority > standard_zone_max → BACKGROUND (3)

    This separation resolves priority inversion where routine telemetry
    from high-priority zones delays critical alarms from low-priority zones.
    """

    def __init__(self, high_zone_max: int, standard_zone_max: int):
        """
        Initialize band classifier with zone thresholds.

        Args:
            high_zone_max: Maximum zone priority for HIGH band (inclusive)
            standard_zone_max: Maximum zone priority for STANDARD band (inclusive)

        Example:
            >>> classifier = BandClassifier(high_zone_max=1, standard_zone_max=3)
            >>> classifier.classify(zone_priority=0, is_alarm=False)
            1  # HIGH band
            >>> classifier.classify(zone_priority=5, is_alarm=True)
            0  # ALARM band (semantic override)
        """
        if high_zone_max < 0:
            raise ValueError(f"high_zone_max must be non-negative, got {high_zone_max}")
        if standard_zone_max < high_zone_max:
            raise ValueError(
                f"standard_zone_max ({standard_zone_max}) must be >= "
                f"high_zone_max ({high_zone_max})"
            )

        self.high_zone_max = high_zone_max
        self.standard_zone_max = standard_zone_max

    def classify(self, zone_priority: int, is_alarm: bool) -> int:
        """
        Classify message into TRIAGE/4 band.

        Args:
            zone_priority: Geographic zone priority (0=highest priority zone)
            is_alarm: Semantic urgency flag (True for emergency messages)

        Returns:
            Band number: 0=ALARM, 1=HIGH, 2=STANDARD, 3=BACKGROUND

        Raises:
            ValueError: If zone_priority is negative
        """
        if zone_priority < 0:
            raise ValueError(f"zone_priority must be non-negative, got {zone_priority}")

        # Semantic urgency overrides geographic priority
        if is_alarm:
            return BAND_ALARM

        # Geographic priority determines band for non-alarms
        if zone_priority <= self.high_zone_max:
            return BAND_HIGH
        elif zone_priority <= self.standard_zone_max:
            return BAND_STANDARD
        else:
            return BAND_BACKGROUND

    def get_band_name(self, band: int) -> str:
        """
        Get human-readable band name.

        Args:
            band: Band number (0-3)

        Returns:
            Band name string
        """
        names = {
            BAND_ALARM: "ALARM",
            BAND_HIGH: "HIGH",
            BAND_STANDARD: "STANDARD",
            BAND_BACKGROUND: "BACKGROUND",
        }
        return names.get(band, f"UNKNOWN({band})")

    def __repr__(self) -> str:
        return (
            f"BandClassifier(high_zone_max={self.high_zone_max}, "
            f"standard_zone_max={self.standard_zone_max})"
        )

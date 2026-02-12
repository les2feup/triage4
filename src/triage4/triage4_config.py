"""
Configuration for TRIAGE/4.

Defines parameters for four-band hierarchical scheduling with token-bucket
resource reservation and per-device fair queuing.
"""

from dataclasses import dataclass


@dataclass
class TRIAGE4Config:
    """
    Configuration for TRIAGE/4 four-band scheduler.

    Bands:
        - ALARM (0): Emergency messages, strict priority, no token bucket
        - HIGH (1): High-priority zone telemetry, token-constrained (Q_H/P_H)
        - STANDARD (2): Standard zone telemetry, token-constrained (Q_S/P_S)
        - BACKGROUND (3): Low-priority zone data, token-constrained (Q_B/P_B)

    Classification:
        Messages are classified based on is_alarm flag and zone_priority:
        - is_alarm=True → ALARM band (overrides zone priority)
        - zone_priority <= high_zone_max → HIGH band
        - zone_priority <= standard_zone_max → STANDARD band
        - zone_priority > standard_zone_max → BACKGROUND band

    Token Buckets:
        Each non-alarm band has independent token bucket with:
        - Budget: tokens per period (Q)
        - Period: refill interval in seconds (P)
        - Burst multiplier: max accumulated tokens as multiple of budget
    """

    # === Band Classification Thresholds ===
    high_zone_max: int = 1  # Zone priorities 0-1 → HIGH band
    standard_zone_max: int = 3  # Zone priorities 2-3 → STANDARD band
    # Zone priorities 4+ → BACKGROUND band

    # === HIGH Band Token Bucket ===
    high_token_budget: int = 20  # Q_H: tokens per period (75% of service rate)
    high_token_period: float = 1.0  # P_H: refill period (seconds)
    high_burst_multiplier: float = 3.0  # Max burst capacity (multiple of budget)

    # === STANDARD Band Token Bucket ===
    standard_token_budget: int = 15  # Q_S: tokens per period (50% of service rate)
    standard_token_period: float = 1.0  # P_S: refill period (seconds)
    standard_burst_multiplier: float = 3.0  # Max burst capacity

    # === BACKGROUND Band Token Bucket ===
    background_token_budget: int = 5  # Q_B: tokens per period (25% of service rate)
    background_token_period: float = 1.0  # P_B: refill period (seconds)
    background_burst_multiplier: float = 3.0  # Max burst capacity

    # === Service Configuration ===
    service_rate: float = 20.0  # μ: mean service rate (messages/second)

    # === Adaptive Alarm Protection (optional) ===
    enable_alarm_protection: bool = False  # Opt-in to adaptive alarm limiting
    alarm_window_duration: float = 10.0  # Sliding window for rate detection (seconds)
    alarm_abnormal_threshold: float = 5.0  # Alarms/sec to activate protection
    alarm_deactivation_threshold: float = 4.0  # Alarms/sec to deactivate (hysteresis)
    alarm_min_observations: int = 3  # Minimum alarms before detection can trigger
    alarm_limit_budget: int = 15  # Tokens per period when protection active
    alarm_limit_period: float = 1.0  # Refill period for alarm token bucket
    alarm_burst_capacity: int = 30  # Burst capacity for alarm bucket

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.high_zone_max < 0:
            raise ValueError(
                f"high_zone_max must be non-negative, got {self.high_zone_max}"
            )

        if self.standard_zone_max < self.high_zone_max:
            raise ValueError(
                f"standard_zone_max ({self.standard_zone_max}) must be >= "
                f"high_zone_max ({self.high_zone_max})"
            )

        # Validate token bucket parameters
        for band, budget, period, burst in [
            (
                "HIGH",
                self.high_token_budget,
                self.high_token_period,
                self.high_burst_multiplier,
            ),
            (
                "STANDARD",
                self.standard_token_budget,
                self.standard_token_period,
                self.standard_burst_multiplier,
            ),
            (
                "BACKGROUND",
                self.background_token_budget,
                self.background_token_period,
                self.background_burst_multiplier,
            ),
        ]:
            if budget <= 0:
                raise ValueError(
                    f"{band} band token_budget must be positive, got {budget}"
                )
            if period <= 0:
                raise ValueError(
                    f"{band} band token_period must be positive, got {period}"
                )
            if burst < 1.0:
                raise ValueError(
                    f"{band} band burst_multiplier must be >= 1.0, got {burst}"
                )

        if self.service_rate <= 0:
            raise ValueError(f"service_rate must be positive, got {self.service_rate}")

        # Validate alarm protection parameters
        if self.enable_alarm_protection:
            if self.alarm_window_duration <= 0:
                raise ValueError("alarm_window_duration must be positive")
            if self.alarm_abnormal_threshold <= 0:
                raise ValueError("alarm_abnormal_threshold must be positive")
            if self.alarm_deactivation_threshold < 0:
                raise ValueError("alarm_deactivation_threshold must be non-negative")
            if self.alarm_deactivation_threshold > self.alarm_abnormal_threshold:
                raise ValueError(
                    "alarm_deactivation_threshold must be <= alarm_abnormal_threshold"
                )
            if self.alarm_min_observations <= 0:
                raise ValueError("alarm_min_observations must be positive")
            if self.alarm_limit_budget <= 0:
                raise ValueError("alarm_limit_budget must be positive")
            if self.alarm_limit_period <= 0:
                raise ValueError("alarm_limit_period must be positive")
            if self.alarm_burst_capacity < self.alarm_limit_budget:
                raise ValueError("alarm_burst_capacity must be >= alarm_limit_budget")


# Convenience factory functions


def create_triage4_default() -> TRIAGE4Config:
    """
    Create TRIAGE/4 with default configuration optimized for realistic operational loads.

    Default bands (proportions of 20 msg/s service rate):
        - ALARM: always served (no token constraint)
        - HIGH: 15 tokens/sec - 75% capacity (zones 0-1)
        - STANDARD: 10 tokens/sec - 50% capacity (zones 2-3)
        - BACKGROUND: 5 tokens/sec - 25% capacity (zones 4+)

    Total token budget: 30 tokens/sec (allows 1.5x service rate for burst handling)
    Burst capacity: 3x budget (allows short bursts up to 3x normal rate)

    Returns:
        TRIAGE4Config with default parameters
    """
    return TRIAGE4Config()


def create_triage4_custom(
    high_zone_max: int = 1,
    standard_zone_max: int = 3,
    high_budget: int = 10,
    standard_budget: int = 5,
    background_budget: int = 2,
    service_rate: float = 20.0,
) -> TRIAGE4Config:
    """
    Create TRIAGE/4 with custom configuration.

    Args:
        high_zone_max: Maximum zone priority for HIGH band
        standard_zone_max: Maximum zone priority for STANDARD band
        high_budget: Token budget for HIGH band (tokens/period)
        standard_budget: Token budget for STANDARD band
        background_budget: Token budget for BACKGROUND band
        service_rate: Mean service rate (messages/second)

    Returns:
        TRIAGE4Config with custom parameters
    """
    return TRIAGE4Config(
        high_zone_max=high_zone_max,
        standard_zone_max=standard_zone_max,
        high_token_budget=high_budget,
        standard_token_budget=standard_budget,
        background_token_budget=background_budget,
        service_rate=service_rate,
    )

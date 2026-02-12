"""
Adaptive token bucket that activates only when alarm protection is enabled.

Wraps the standard TokenBucket and exposes activation/deactivation hooks with
hysteresis controlled by the caller (e.g., AlarmRateMonitor).
"""

from .token_bucket import TIME_TOLERANCE, TokenBucket


class AdaptiveTokenBucket:
    """Token bucket that can be activated/deactivated at runtime."""

    def __init__(self, budget: int, period: float, burst_capacity: int | None = None):
        if budget <= 0:
            raise ValueError("budget must be positive")
        if period <= 0:
            raise ValueError("period must be positive")
        if burst_capacity is not None and burst_capacity < budget:
            raise ValueError("burst_capacity must be >= budget")

        self.bucket = TokenBucket(
            budget=budget, period=period, burst_capacity=burst_capacity
        )
        self.active = False

    def activate(self, current_time: float) -> None:
        """Enable rate limiting and realign refill schedule."""
        if not self.active:
            self.active = True
            # Align next refill to current_time to avoid retroactive refills.
            self.bucket.next_refill = current_time + self.bucket.period
            self.bucket.tokens = min(self.bucket.tokens, self.bucket.max_capacity)

    def deactivate(self) -> None:
        """Disable rate limiting (bucket becomes pass-through)."""
        self.active = False

    def consume(self, current_time: float, amount: int = 1) -> bool:
        """
        Consume tokens if active. Returns True if the request is allowed.
        When inactive, always returns True (no limiting).
        """
        if not self.active:
            return True

        self.bucket.refill(current_time)
        return self.bucket.consume(amount)

    def get_next_refill_time(self) -> float:
        """Expose next refill time for event scheduling."""
        if not self.active:
            # If inactive, never schedule on bucket events
            return float("inf")
        return self.bucket.get_next_refill_time()

    @property
    def tokens(self) -> int:
        """Current token count (0 when inactive implies unlimited)."""
        return self.bucket.tokens if self.active else -1


"""Token bucket for S-band reservation in Vanilla APS."""

# Floating-point tolerance for time comparisons
TIME_TOLERANCE = 1e-12
import math


def time_lte(t1: float, t2: float) -> bool:
    """Check if t1 <= t2 with floating-point tolerance."""
    return t1 <= t2 + TIME_TOLERANCE


class TokenBucket:
    """
    Token bucket for S-band rate limiting.

    Provides Q tokens every P seconds with burst capacity.
    Guarantees minimum S-band throughput.
    """

    def __init__(self, budget: int, period: float, burst_capacity: int = None):
        """
        Initialize token bucket.

        Args:
            budget: Tokens per period (Q)
            period: Refill period in seconds (P)
            burst_capacity: Max accumulated tokens (default: 2 * budget)
        """
        self.budget = budget
        self.period = period
        self.max_capacity = burst_capacity if burst_capacity is not None else (budget * 2)
        self.tokens = budget
        self.next_refill = period

    def consume(self, amount: int = 1) -> bool:
        """
        Attempt to consume tokens.

        Args:
            amount: Number of tokens to consume

        Returns:
            True if consumption succeeded, False if insufficient tokens
        """
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def refill(self, current_time: float) -> None:
        """Refill tokens if refill time has passed (O(1) jump)."""
        if not time_lte(self.next_refill, current_time):
            return

        # Number of full periods that elapsed since the scheduled refill (including the current one)
        elapsed = current_time - self.next_refill
        cycles = int(math.floor(elapsed / self.period)) + 1 if self.period > 0 else 0
        refill_tokens = cycles * self.budget

        self.tokens = min(self.tokens + refill_tokens, self.max_capacity)
        self.next_refill += cycles * self.period

    def has_tokens(self) -> bool:
        """Check if bucket has any tokens."""
        return self.tokens > 0

    def get_next_refill_time(self) -> float:
        """Get next scheduled refill time."""
        return self.next_refill

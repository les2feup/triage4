"""
Sliding-window alarm rate monitor for adaptive protection.

Detects abnormal alarm rates using a simple windowed counter with hysteresis.
"""

from collections import deque
from typing import Deque, Tuple

from .token_bucket import TIME_TOLERANCE


class AlarmRateMonitor:
    """Track alarm arrivals and detect abnormal rates."""

    def __init__(
        self,
        window_duration: float,
        abnormal_threshold: float,
        deactivation_threshold: float,
        min_observations: int = 1,
    ):
        """
        Args:
            window_duration: Sliding window size in seconds
            abnormal_threshold: Alarms/sec that triggers protection
            deactivation_threshold: Alarms/sec below which protection deactivates
            min_observations: Minimum alarms before detection can trigger
        """
        if window_duration <= 0:
            raise ValueError("window_duration must be positive")
        if abnormal_threshold <= 0:
            raise ValueError("abnormal_threshold must be positive")
        if deactivation_threshold < 0:
            raise ValueError("deactivation_threshold must be non-negative")
        if deactivation_threshold > abnormal_threshold:
            raise ValueError(
                "deactivation_threshold must be <= abnormal_threshold for hysteresis"
            )
        if min_observations <= 0:
            raise ValueError("min_observations must be positive")

        self.window_duration = float(window_duration)
        self.abnormal_threshold = float(abnormal_threshold)
        self.deactivation_threshold = float(deactivation_threshold)
        self.min_observations = int(min_observations)

        self._arrivals: Deque[Tuple[float, str]] = deque()

    def record_arrival(self, timestamp: float, alarm_source: str) -> None:
        """Add an alarm arrival to the window."""
        self._arrivals.append((timestamp, alarm_source))
        self._prune(timestamp)

    def get_rate(self, now: float | None = None) -> float:
        """Return alarms/sec over the current window."""
        if now is None and self._arrivals:
            now = self._arrivals[-1][0]
        elif now is None:
            return 0.0

        self._prune(now)
        if not self._arrivals:
            return 0.0

        window_span = max(self.window_duration, TIME_TOLERANCE)
        return len(self._arrivals) / window_span

    def is_abnormal(self, now: float | None = None) -> bool:
        """True if rate exceeds abnormal threshold with enough observations."""
        if now is None and self._arrivals:
            now = self._arrivals[-1][0]
        elif now is None:
            return False

        self._prune(now)
        if len(self._arrivals) < self.min_observations:
            return False
        return self.get_rate(now) >= self.abnormal_threshold - TIME_TOLERANCE

    def is_recovered(self, now: float | None = None) -> bool:
        """True if rate has fallen below deactivation threshold."""
        if now is None and self._arrivals:
            now = self._arrivals[-1][0]
        elif now is None:
            return True

        self._prune(now)
        return self.get_rate(now) <= self.deactivation_threshold + TIME_TOLERANCE

    def _prune(self, now: float) -> None:
        """Drop arrivals outside the sliding window."""
        cutoff = now - self.window_duration - TIME_TOLERANCE
        while self._arrivals and self._arrivals[0][0] < cutoff:
            self._arrivals.popleft()


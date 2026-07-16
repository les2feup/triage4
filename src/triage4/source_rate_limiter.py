"""
Per-source alarm rate limiting for adaptive protection.

Caps an individual misbehaving alarm source without silencing it. A source whose
own rate crosses the abnormal threshold is limited to a residual budget rather
than discarded, so a failing device still signals its condition while losing the
ability to monopolize the ALARM band. Sources below the threshold are never
limited, so a legitimate emergency spread across many devices passes untouched.

This layer sits ahead of the band-global backstop: shedding a flood here keeps
the flood from draining the global bucket that legitimate alarms depend on.
"""

from typing import Dict

from .adaptive_token_bucket import AdaptiveTokenBucket
from .alarm_rate_monitor import AlarmRateMonitor


class SourceRateLimiter:
    """Throttle individual alarm sources that exceed their own rate threshold."""

    def __init__(
        self,
        window_duration: float,
        abnormal_threshold: float,
        deactivation_threshold: float,
        min_observations: int,
        limit_budget: int,
        limit_period: float,
        burst_capacity: int,
    ):
        """
        Args:
            window_duration: Sliding window for per-source rate detection (seconds)
            abnormal_threshold: A source's own alarms/sec that triggers its limit
            deactivation_threshold: Per-source alarms/sec that releases the limit
            min_observations: Minimum arrivals from a source before it can trigger
            limit_budget: Tokens per period granted to a limited source
            limit_period: Refill period for a limited source's bucket
            burst_capacity: Burst capacity for a limited source's bucket
        """
        self.window_duration = float(window_duration)
        self.abnormal_threshold = float(abnormal_threshold)
        self.deactivation_threshold = float(deactivation_threshold)
        self.min_observations = int(min_observations)
        self.limit_budget = int(limit_budget)
        self.limit_period = float(limit_period)
        self.burst_capacity = int(burst_capacity)

        self._monitors: Dict[str, AlarmRateMonitor] = {}
        self._buckets: Dict[str, AdaptiveTokenBucket] = {}

        self.activations = 0
        self.deactivations = 0

    def _track(self, source: str) -> None:
        """Create the monitor/bucket pair for a source on first arrival."""
        if source in self._monitors:
            return
        self._monitors[source] = AlarmRateMonitor(
            window_duration=self.window_duration,
            abnormal_threshold=self.abnormal_threshold,
            deactivation_threshold=self.deactivation_threshold,
            min_observations=self.min_observations,
        )
        self._buckets[source] = AdaptiveTokenBucket(
            budget=self.limit_budget,
            period=self.limit_period,
            burst_capacity=self.burst_capacity,
        )

    def admit(self, current_time: float, source: str) -> bool:
        """
        Record an arrival from a source and decide whether to admit it.

        Returns True when the source is within its own rate, or when a limited
        source still has residual budget. False sheds the message.
        """
        self._track(source)
        monitor = self._monitors[source]
        bucket = self._buckets[source]

        monitor.record_arrival(current_time, source)
        if monitor.is_abnormal(current_time):
            if not bucket.active:
                bucket.activate(current_time)
                self.activations += 1
        elif bucket.active and monitor.is_recovered(current_time):
            bucket.deactivate()
            self.deactivations += 1

        return bucket.consume(current_time)

    def is_limited(self, source: str) -> bool:
        """True while a source is being rate-limited."""
        bucket = self._buckets.get(source)
        return bucket is not None and bucket.active

    @property
    def limited_sources(self) -> int:
        """Count of sources currently under limit."""
        return sum(1 for b in self._buckets.values() if b.active)

    @property
    def tracked_sources(self) -> int:
        """Count of sources seen so far."""
        return len(self._monitors)

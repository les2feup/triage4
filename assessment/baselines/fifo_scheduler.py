"""
FIFO Scheduler (Lower Bound Baseline for TRIAGE/4 evaluation).

Pure first-in-first-out queueing with no priority - demonstrates
worst-case scenario where alarms are treated like any other message.
"""

from typing import List, Optional

import numpy as np

from ..metrics.results import SchedulerResult


class FIFOScheduler:
    """
    First-In-First-Out (FIFO) / Best-Effort scheduler.

    Scheduling policy:
        - Serve messages in strict arrival order
        - Ignores zone_priorities
        - Ignores is_alarm flag
        - Completely fair but no urgency awareness

    Demonstrates worst case: No differentiation between critical
    alarms and routine telemetry.

    Example:
        >>> scheduler = FIFOScheduler(service_rate=20.0)
        >>> result = scheduler.schedule(
        ...     arrival_times=[0.0, 0.1, 0.2],
        ...     device_ids=["A", "B", "C"],
        ...     zone_priorities=[5, 0, 2],  # All ignored
        ...     is_alarm=[True, False, False]  # Alarm treated same as others
        ... )
        >>> # All served in arrival order: A, B, C
    """

    def __init__(
        self,
        service_rate: float = 20.0,
        scheduler_seed: Optional[int] = None,
    ):
        """
        Initialize FIFO scheduler.

        Args:
            service_rate: Mean service rate μ (messages/second)
            scheduler_seed: Random seed for reproducible service times
        """
        self.service_rate = service_rate

        # Independent RNG for service times
        if scheduler_seed is not None:
            self.service_rng = np.random.default_rng(seed=scheduler_seed)
        else:
            self.service_rng = np.random.default_rng()

    def schedule(
        self,
        arrival_times: List[float],
        device_ids: List[str],
        zone_priorities: List[int],
        is_alarm: List[bool],
    ) -> SchedulerResult:
        """
        Simulate FIFO queueing.

        Serves messages in pure arrival order, ignoring all priority
        and urgency information. This is the "no scheduling" baseline.

        Args:
            arrival_times: Sorted list of message arrival times
            device_ids: Device identifier for each message (tracked but not used)
            zone_priorities: Geographic zone priority (IGNORED)
            is_alarm: Semantic urgency flag (IGNORED)

        Returns:
            SchedulerResult with waiting times and band assignments
        """
        self._validate_inputs(arrival_times, device_ids, zone_priorities, is_alarm)

        n = len(arrival_times)
        waiting_times = np.zeros(n)
        e2e_times = np.zeros(n)

        # FIFO: Process in strict arrival order
        server_time = 0.0
        for i in range(n):
            # Server available at max(arrival_time, previous_completion)
            server_time = max(server_time, arrival_times[i])

            # Compute metrics
            waiting_times[i] = server_time - arrival_times[i]
            service_time = self.service_rng.exponential(1 / self.service_rate)
            e2e_times[i] = waiting_times[i] + service_time

            # Update server completion time
            server_time += service_time

        # Map to bands for compatibility with TRIAGE/4 analysis
        # (Note: This doesn't affect scheduling, only result classification)
        from triage4 import BandClassifier

        classifier = BandClassifier(high_zone_max=1, standard_zone_max=3)
        bands = [
            classifier.classify(zone_priorities[i], is_alarm[i]) for i in range(n)
        ]

        return SchedulerResult(
            waiting_times=waiting_times,
            e2e_times=e2e_times,
            priorities=bands,  # Band assignments for analysis
            metadata={
                "scheduler": "FIFO",
                "service_rate": self.service_rate,
                "note": "Pure arrival order (all priorities ignored)",
            },
        )

    def _validate_inputs(
        self,
        arrival_times: List[float],
        device_ids: List[str],
        zone_priorities: List[int],
        is_alarm: List[bool],
    ) -> None:
        """Validate scheduler inputs."""
        n = len(arrival_times)

        if len(device_ids) != n:
            raise ValueError(
                f"device_ids length ({len(device_ids)}) must match "
                f"arrival_times length ({n})"
            )

        if len(zone_priorities) != n:
            raise ValueError(
                f"zone_priorities length ({len(zone_priorities)}) must match "
                f"arrival_times length ({n})"
            )

        if len(is_alarm) != n:
            raise ValueError(
                f"is_alarm length ({len(is_alarm)}) must match "
                f"arrival_times length ({n})"
            )

        if n == 0:
            raise ValueError("Must have at least one job")

        if arrival_times != sorted(arrival_times):
            raise ValueError("arrival_times must be sorted")

    def __repr__(self) -> str:
        return f"FIFOScheduler(service_rate={self.service_rate})"

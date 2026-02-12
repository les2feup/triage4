"""
Strict Priority Scheduler (Baseline for TRIAGE/4 evaluation).

Geographic priority only - demonstrates the alarm delay problem
that TRIAGE/4 solves with semantic urgency override.
"""

import heapq
from typing import List, Optional

import numpy as np

from ..metrics.results import SchedulerResult


class StrictPriorityScheduler:
    """
    Strict priority queueing based on geographic zone priority.

    Scheduling policy:
        - Always serve highest geographic priority (zone 0 > zone 1 > ...)
        - Ignores semantic urgency (is_alarm flag)
        - Lower priority zones starve under sustained high-priority load

    Demonstrates the problem: Critical alarms from low-priority zones
    are delayed by routine telemetry from high-priority zones.

    Example:
        >>> scheduler = StrictPriorityScheduler(service_rate=20.0)
        >>> result = scheduler.schedule(
        ...     arrival_times=[0.0, 0.1],
        ...     device_ids=["sensor_1", "sensor_2"],
        ...     zone_priorities=[0, 5],  # High zone, low zone
        ...     is_alarm=[False, True]   # Routine, ALARM (ignored!)
        ... )
        >>> # Zone 0 served first despite zone 5 being an alarm
    """

    def __init__(
        self,
        service_rate: float = 20.0,
        scheduler_seed: Optional[int] = None,
    ):
        """
        Initialize strict priority scheduler.

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
        Simulate strict geographic priority queueing.

        Uses zone_priorities for scheduling order (ignores is_alarm).
        This demonstrates why TRIAGE/4 is needed: alarms from low-priority
        zones get delayed by routine traffic from high-priority zones.

        Args:
            arrival_times: Sorted list of message arrival times
            device_ids: Device identifier for each message (tracked but not used)
            zone_priorities: Geographic zone priority (0=highest priority zone)
            is_alarm: Semantic urgency flag (IGNORED in this baseline)

        Returns:
            SchedulerResult with waiting times and band assignments

        Note:
            is_alarm is ignored - this is the key limitation that TRIAGE/4 fixes
        """
        self._validate_inputs(arrival_times, device_ids, zone_priorities, is_alarm)

        n = len(arrival_times)
        server_time = 0.0
        queue = []  # min-heap: (zone_priority, arrival_seq, arrival_time, job_idx)
        waiting_times = np.zeros(n)
        e2e_times = np.zeros(n)

        i = 0  # Next arrival to process
        arrival_seq = 0  # FIFO tie-breaker for equal priorities
        processed = 0  # Jobs completed
        current_task = None  # (zone_priority, job_idx) or None
        current_end = float("inf")  # Completion time of current job

        while processed < n:
            # Determine next event time (arrival vs completion)
            next_arrival = arrival_times[i] if i < n else float("inf")
            server_time = next_arrival if next_arrival <= current_end else current_end

            # Process all arrivals at this time (with tolerance for floating-point)
            while i < n and abs(arrival_times[i] - server_time) < 1e-12:
                # Queue by zone_priority (ignoring is_alarm!)
                heapq.heappush(
                    queue,
                    (zone_priorities[i], arrival_seq, arrival_times[i], i),
                )
                arrival_seq += 1
                i += 1

            # Process completion if job finishes now
            if current_task and abs(current_end - server_time) < 1e-12:
                processed += 1
                current_task = None
                current_end = float("inf")

            # Schedule next job if server idle
            if current_task is None and queue:
                zone_prio, _, arr, idx = heapq.heappop(queue)

                # Compute metrics
                waiting_times[idx] = server_time - arr
                service_time = self.service_rng.exponential(1 / self.service_rate)
                e2e_times[idx] = waiting_times[idx] + service_time

                # Update server state
                current_task = (zone_prio, idx)
                current_end = server_time + service_time

        # Map zone_priorities to bands for compatibility with TRIAGE/4 analysis
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
                "scheduler": "StrictPriority",
                "service_rate": self.service_rate,
                "note": "is_alarm flag ignored (geographic priority only)",
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

        if any(zp < 0 for zp in zone_priorities):
            raise ValueError("All zone_priorities must be non-negative")

    def __repr__(self) -> str:
        return f"StrictPriorityScheduler(service_rate={self.service_rate})"

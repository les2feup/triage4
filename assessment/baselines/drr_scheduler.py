"""
Deficit Round Robin scheduler (DRR) baseline for TRIAGE/4 evaluation.

Policy: deficit round robin across all devices in a single flat queue (no
geographic band structure, no alarm preemption). Each device maintains a
deficit counter initialised to 0. On each scheduling round the server visits
the head device, adds quantum Q to its deficit, serves messages from that
device while deficit ≥ cost (cost = 1 per message), then moves to the next.

With Q = 1 and cost = 1 the deficit counter never accumulates — every device
gets exactly one message per round — so DRR degenerates mathematically to
plain round-robin. The full deficit-tracking code is retained because it
faithfully implements the cited algorithm and is forward-compatible with
non-uniform costs.

ALARM messages receive no preferential treatment (intentional). This isolates
pure device-level fairness and exposes the cost of ignoring semantic urgency.

Citation: Tabatabaee & Le Boudec, "Deficit Round-Robin: A Second Network
Calculus Analysis," IEEE/ACM ToN 2022, DOI: 10.1109/TNET.2022.3164772.
"""

from collections import deque
from typing import Dict, Deque, List, Optional

import numpy as np

from triage4 import BandClassifier
from ..metrics.results import SchedulerResult

QUANTUM: float = 1.0   # Q — added to deficit on each device visit
MSG_COST: float = 1.0  # uniform message cost (no packet-size model)


class DRRScheduler:
    """
    Deficit Round Robin across all devices with no geographic priority.

    Isolates per-device fairness without semantic preemption or band structure:
    - no A component (is_alarm ignored)
    - no B component (no geographic bands)
    - no C component (no token buckets)
    - no E component (no AAP)
    """

    def __init__(
        self,
        service_rate: float = 20.0,
        scheduler_seed: Optional[int] = None,
    ):
        self.service_rate = service_rate
        self.service_rng = (
            np.random.default_rng(seed=scheduler_seed)
            if scheduler_seed is not None
            else np.random.default_rng()
        )

    def schedule(
        self,
        arrival_times: List[float],
        device_ids: List[str],
        zone_priorities: List[int],
        is_alarm: List[bool],
    ) -> SchedulerResult:
        """
        Simulate DRR scheduling.

        Args:
            arrival_times: Sorted list of message arrival times
            device_ids: Device identifier for each message
            zone_priorities: Geographic zone priority (IGNORED by DRR)
            is_alarm: Semantic urgency flag (IGNORED by DRR)

        Returns:
            SchedulerResult with waiting times and band assignments
        """
        n = len(arrival_times)
        _validate_inputs(arrival_times, device_ids, zone_priorities, is_alarm)

        waiting_times = np.zeros(n)
        e2e_times = np.zeros(n)

        # Per-device message queues and deficit counters
        device_queues: Dict[str, Deque[int]] = {}   # device → deque of job indices
        device_deficit: Dict[str, float] = {}        # device → current deficit

        # Active device deque for round-robin traversal (devices with pending messages)
        active_devices: Deque[str] = deque()
        active_set: set = set()  # O(1) membership test

        i: int = 0             # next arrival index
        server_time: float = 0.0
        current_end: float = float("inf")
        current_job: Optional[int] = None
        processed: int = 0

        def _enqueue_arrival(job_idx: int) -> None:
            """Add a newly arrived message to its device queue."""
            dev = device_ids[job_idx]
            if dev not in device_queues:
                device_queues[dev] = deque()
                device_deficit[dev] = 0.0
            device_queues[dev].append(job_idx)
            if dev not in active_set:
                active_devices.append(dev)
                active_set.add(dev)

        def _dispatch_drr() -> Optional[int]:
            """
            Run one DRR pass over active_devices until a job is dispatched.

            Iterates up to len(active_devices) times (one full round) so that
            every device gets a chance to serve if it has deficit. Returns the
            dispatched job_idx or None if no messages are available.
            """
            rounds = len(active_devices)
            for _ in range(rounds):
                if not active_devices:
                    return None
                dev = active_devices[0]
                device_deficit[dev] += QUANTUM
                # Serve while deficit covers message cost
                if device_queues[dev] and device_deficit[dev] >= MSG_COST:
                    jidx = device_queues[dev].popleft()
                    device_deficit[dev] -= MSG_COST
                    # Remove device from active set if its queue is now empty
                    if not device_queues[dev]:
                        active_devices.popleft()
                        active_set.discard(dev)
                    else:
                        # Rotate device to back for next round
                        active_devices.rotate(-1)
                    return jidx
                else:
                    # No message to serve (or insufficient deficit) — rotate to next device
                    active_devices.rotate(-1)
            return None

        while processed < n:
            next_arrival = arrival_times[i] if i < n else float("inf")
            event_time = min(next_arrival, current_end)
            server_time = event_time

            # Enqueue all arrivals at this event time
            while i < n and abs(arrival_times[i] - server_time) < 1e-12:
                _enqueue_arrival(i)
                i += 1

            # Handle completion
            if current_job is not None and abs(current_end - server_time) < 1e-12:
                current_job = None
                current_end = float("inf")
                processed += 1

            # Dispatch next job via DRR
            if current_job is None:
                job_idx = _dispatch_drr()
                if job_idx is not None:
                    waiting_times[job_idx] = server_time - arrival_times[job_idx]
                    service_time = self.service_rng.exponential(1.0 / self.service_rate)
                    e2e_times[job_idx] = waiting_times[job_idx] + service_time
                    current_job = job_idx
                    current_end = server_time + service_time

        # Band assignment for metric comparability with TRIAGE/4
        classifier = BandClassifier(high_zone_max=1, standard_zone_max=3)
        bands = [classifier.classify(zone_priorities[j], is_alarm[j]) for j in range(n)]

        return SchedulerResult(
            waiting_times=waiting_times,
            e2e_times=e2e_times,
            priorities=bands,
            metadata={
                "scheduler": "DRR",
                "service_rate": self.service_rate,
                "quantum": QUANTUM,
                "note": "Global per-device DRR; is_alarm and zone_priority ignored",
            },
        )

    def __repr__(self) -> str:
        return f"DRRScheduler(service_rate={self.service_rate})"


def _validate_inputs(arrival_times, device_ids, zone_priorities, is_alarm) -> None:
    n = len(arrival_times)
    if len(device_ids) != n or len(zone_priorities) != n or len(is_alarm) != n:
        raise ValueError("All input lists must have the same length")
    if n == 0:
        raise ValueError("Must have at least one job")
    if arrival_times != sorted(arrival_times):
        raise ValueError("arrival_times must be sorted")

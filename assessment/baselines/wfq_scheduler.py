"""
Weighted Fair Queueing scheduler (WFQ) baseline for TRIAGE/4 evaluation.

Policy: pure geographic per-device virtual-clock WFQ. Each device is a distinct
WFQ flow; its weight is φ(z) = 1/(z+1), so zone 0 (highest geographic importance)
gets weight 1.0, zone 1 gets 0.5, and so on. The is_alarm flag is ignored in the
scheduling decision — this is a classical WFQ over geographic metadata only.

The virtual finish time for message i arriving from device d at zone z is:
    F_i = max(virtual_clock, F_d_prev) + (z + 1)

where (z+1) is the normalised cost (inverse weight) and virtual_clock is frozen
when the server is idle (P-GPS approximation).

A zone-5 alarm message (cost=6) is therefore scheduled behind a zone-0 routine
message (cost=1). This is the priority-inversion failure mode that TRIAGE/4's
semantic override (component A) is designed to fix; WFQ here provides the
controlled absence of that component.

Band assignment in SchedulerResult uses the same zone thresholds as TRIAGE/4 so
metrics are comparable across schedulers.

Citation: Chen & Wen, "Design of Edge-IoMT Network Architecture with
Weight-Based Scheduling," Sensors (MDPI) 2023, DOI: 10.3390/s23208553.
"""

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from triage4 import BandClassifier
from ..metrics.results import SchedulerResult


class WFQScheduler:
    """
    Pure geographic per-device Weighted Fair Queueing.

    Isolates continuous per-device weighted fairness without any of TRIAGE/4's
    additional mechanisms:
    - no semantic override (no A component): is_alarm ignored in dispatch
    - no discrete band structure (no B component)
    - no per-band token-bucket guarantees (no C component)
    - no AAP (no E component)

    Scheduling policy: min-heap ordered by virtual finish time F_i, broken by
    arrival sequence. All messages compete through the virtual-clock mechanism
    regardless of is_alarm.
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
        Simulate pure geographic WFQ with virtual-clock approximation.

        is_alarm is accepted for interface compatibility and band-assignment
        metrics but plays no role in the scheduling decision.
        """
        n = len(arrival_times)
        _validate_inputs(arrival_times, device_ids, zone_priorities, is_alarm)

        waiting_times = np.zeros(n)
        e2e_times = np.zeros(n)

        virtual_clock: float = 0.0
        device_vft: Dict[str, float] = {}
        # min-heap of (virtual_finish_time, arrival_seq, job_idx)
        wfq_heap: List[Tuple[float, int, int]] = []

        server_time: float = 0.0
        arrival_seq: int = 0
        i: int = 0
        processed: int = 0
        current_end: float = float("inf")
        current_job: Optional[int] = None

        while processed < n:
            next_arrival = arrival_times[i] if i < n else float("inf")
            server_time = min(next_arrival, current_end)

            # Enqueue all messages arriving at this event time (is_alarm ignored)
            while i < n and abs(arrival_times[i] - server_time) < 1e-12:
                z = zone_priorities[i]
                cost = z + 1  # inverse weight; zone 0 → cost 1, zone 5 → cost 6
                d = device_ids[i]
                vft = max(virtual_clock, device_vft.get(d, 0.0)) + cost
                heapq.heappush(wfq_heap, (vft, arrival_seq, i))
                arrival_seq += 1
                i += 1

            # Handle completion; advance virtual clock (P-GPS: vc tracks real time while busy)
            if current_job is not None and abs(current_end - server_time) < 1e-12:
                virtual_clock = max(virtual_clock, server_time)
                current_job = None
                current_end = float("inf")
                processed += 1

            # Dispatch next job
            if current_job is None and wfq_heap:
                vft, _, job_idx = heapq.heappop(wfq_heap)
                device_vft[device_ids[job_idx]] = vft
                waiting_times[job_idx] = server_time - arrival_times[job_idx]
                service_time = self.service_rng.exponential(1.0 / self.service_rate)
                e2e_times[job_idx] = waiting_times[job_idx] + service_time
                current_job = job_idx
                current_end = server_time + service_time

        # Band assignment uses same zone thresholds as TRIAGE/4 for metric comparability
        classifier = BandClassifier(high_zone_max=1, standard_zone_max=3)
        bands = [classifier.classify(zone_priorities[j], is_alarm[j]) for j in range(n)]

        return SchedulerResult(
            waiting_times=waiting_times,
            e2e_times=e2e_times,
            priorities=bands,
            metadata={
                "scheduler": "WFQ",
                "service_rate": self.service_rate,
                "note": "Pure geographic per-device WFQ; is_alarm ignored in dispatch",
            },
        )

    def __repr__(self) -> str:
        return f"WFQScheduler(service_rate={self.service_rate})"


def _validate_inputs(arrival_times, device_ids, zone_priorities, is_alarm) -> None:
    n = len(arrival_times)
    if len(device_ids) != n or len(zone_priorities) != n or len(is_alarm) != n:
        raise ValueError("All input lists must have the same length")
    if n == 0:
        raise ValueError("Must have at least one job")
    if arrival_times != sorted(arrival_times):
        raise ValueError("arrival_times must be sorted")

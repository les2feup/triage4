"""
Token Bucket Priority scheduler (TBP) baseline for TRIAGE/4 evaluation.

Policy: three-band strict geographic priority (HIGH → STANDARD → BACKGROUND) with
per-band token buckets and FIFO within each band. The is_alarm flag is ignored in
the scheduling decision — band assignment uses only zone_priority, so a zone-5 alarm
message is placed in the BACKGROUND band and waits behind all HIGH and STANDARD traffic.

This isolates geographic band structure (B) combined with per-band token-bucket rate
control (C), without any of TRIAGE/4's additional mechanisms:
- no semantic override (no A component): is_alarm ignored in dispatch
- no per-device round-robin (no D component)
- no AAP (no E component)

A zone-5 alarm therefore waits behind zone-0 HIGH traffic; this is the same
priority-inversion failure mode as WFQ and Strict Priority. B and C together are
insufficient without A — demonstrating the motivation for TRIAGE/4's semantic override.

Band assignment in SchedulerResult uses the actual is_alarm flag (matching TRIAGE/4
classification) so that alarm-latency metrics are comparable across schedulers.

Citation: Fu, Sun & He, "A Survey of Traffic Shaping Technology in Internet of
Things," IEEE Access 2023, DOI: 10.1109/ACCESS.2022.3233394.
"""

from collections import deque
from typing import Dict, Deque, List, Optional

import numpy as np

from triage4 import BAND_BACKGROUND, BAND_HIGH, BAND_STANDARD, BandClassifier
from triage4.token_bucket import TokenBucket
from triage4.triage4_config import TRIAGE4Config
from ..metrics.results import SchedulerResult

# Shared time-comparison tolerance (matches TRIAGE/4 internals)
_TIME_TOL = 1e-12


class TokenBucketPriorityScheduler:
    """
    Three-band strict geographic priority with per-band token buckets, FIFO within band.

    Isolates B+C: geographic band structure and per-band token-bucket rate control.
    is_alarm is ignored in dispatch — alarms compete through their geographic band only.
    Token parameters match TRIAGE/4 defaults for a fair comparison.
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

        # Default token budgets match TRIAGE/4Config defaults
        cfg = TRIAGE4Config(service_rate=service_rate)
        self.high_bucket = TokenBucket(
            budget=cfg.high_token_budget,
            period=cfg.high_token_period,
            burst_capacity=int(cfg.high_token_budget * cfg.high_burst_multiplier),
        )
        self.standard_bucket = TokenBucket(
            budget=cfg.standard_token_budget,
            period=cfg.standard_token_period,
            burst_capacity=int(cfg.standard_token_budget * cfg.standard_burst_multiplier),
        )
        self.background_bucket = TokenBucket(
            budget=cfg.background_token_budget,
            period=cfg.background_token_period,
            burst_capacity=int(cfg.background_token_budget * cfg.background_burst_multiplier),
        )
        self._classifier = BandClassifier(
            high_zone_max=cfg.high_zone_max,
            standard_zone_max=cfg.standard_zone_max,
        )

    def schedule(
        self,
        arrival_times: List[float],
        device_ids: List[str],
        zone_priorities: List[int],
        is_alarm: List[bool],
    ) -> SchedulerResult:
        """
        Simulate TBP: three-band geographic priority, FIFO within band, token-constrained.

        is_alarm is accepted for interface compatibility and metric reporting but plays
        no role in the scheduling decision — all messages are routed by zone_priority only.
        """
        n = len(arrival_times)
        _validate_inputs(arrival_times, device_ids, zone_priorities, is_alarm)

        # Routing: ignore is_alarm so alarms compete through their geographic band
        routing_bands = [
            self._classifier.classify(zone_priorities[j], False) for j in range(n)
        ]
        # Reporting: use actual is_alarm for cross-scheduler metric comparability
        report_bands = [
            self._classifier.classify(zone_priorities[j], is_alarm[j]) for j in range(n)
        ]

        waiting_times = np.zeros(n)
        e2e_times = np.zeros(n)

        # FIFO queues per geographic band (no ALARM band — is_alarm ignored in routing)
        band_queues: Dict[int, Deque[int]] = {
            BAND_HIGH: deque(),
            BAND_STANDARD: deque(),
            BAND_BACKGROUND: deque(),
        }

        i: int = 0
        server_time: float = 0.0
        current_end: float = float("inf")
        current_job: Optional[int] = None
        processed: int = 0

        def _next_event_time() -> float:
            """
            Next event time: earliest of next arrival, current completion, or token refill.

            Including token refill times prevents an infinite loop when all messages have
            arrived, the server is idle, and all token buckets are simultaneously depleted.
            Without this, event_time = min(inf, inf) = inf and the refill while-loop hangs.
            """
            t = arrival_times[i] if i < n else float("inf")
            t = min(t, current_end)
            # Add refill candidates only for non-empty token-gated bands
            if band_queues[BAND_HIGH]:
                t = min(t, self.high_bucket.next_refill)
            if band_queues[BAND_STANDARD]:
                t = min(t, self.standard_bucket.next_refill)
            if band_queues[BAND_BACKGROUND]:
                t = min(t, self.background_bucket.next_refill)
            return t

        def _refill(t: float) -> None:
            """Refill token buckets using the O(1) jump method (avoids iterating periods)."""
            for bucket in (self.high_bucket, self.standard_bucket, self.background_bucket):
                bucket.refill(t)

        def _dispatch() -> Optional[int]:
            """Select next job using band-priority with token gating, FIFO within band."""
            # HIGH: token-constrained
            if band_queues[BAND_HIGH] and self.high_bucket.consume():
                return band_queues[BAND_HIGH].popleft()
            # STANDARD: token-constrained
            if band_queues[BAND_STANDARD] and self.standard_bucket.consume():
                return band_queues[BAND_STANDARD].popleft()
            # BACKGROUND: token-constrained
            if band_queues[BAND_BACKGROUND] and self.background_bucket.consume():
                return band_queues[BAND_BACKGROUND].popleft()
            return None

        def _all_empty() -> bool:
            return all(len(q) == 0 for q in band_queues.values())

        while processed < n:
            server_time = _next_event_time()

            _refill(server_time)

            # Enqueue all arrivals at this event time (routed by geographic band only)
            while i < n and abs(arrival_times[i] - server_time) < _TIME_TOL:
                band_queues[routing_bands[i]].append(i)
                i += 1

            # Handle completion
            if current_job is not None and abs(current_end - server_time) < _TIME_TOL:
                current_job = None
                current_end = float("inf")
                processed += 1

            # Dispatch
            if current_job is None:
                job_idx = _dispatch()
                if job_idx is not None:
                    waiting_times[job_idx] = server_time - arrival_times[job_idx]
                    service_time = self.service_rng.exponential(1.0 / self.service_rate)
                    e2e_times[job_idx] = waiting_times[job_idx] + service_time
                    current_job = job_idx
                    current_end = server_time + service_time

            # Fast exit when all arrived and queues empty
            if i >= n and current_job is None and _all_empty():
                break

        return SchedulerResult(
            waiting_times=waiting_times,
            e2e_times=e2e_times,
            priorities=report_bands,
            metadata={
                "scheduler": "TBP",
                "service_rate": self.service_rate,
                "note": "Three-band geographic token-bucket priority; is_alarm ignored in dispatch (no A component)",
            },
        )

    def __repr__(self) -> str:
        return f"TokenBucketPriorityScheduler(service_rate={self.service_rate})"


def _validate_inputs(arrival_times, device_ids, zone_priorities, is_alarm) -> None:
    n = len(arrival_times)
    if len(device_ids) != n or len(zone_priorities) != n or len(is_alarm) != n:
        raise ValueError("All input lists must have the same length")
    if n == 0:
        raise ValueError("Must have at least one job")
    if arrival_times != sorted(arrival_times):
        raise ValueError("arrival_times must be sorted")

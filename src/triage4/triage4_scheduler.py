"""
TRIAGE/4 scheduler.

Four-band hierarchical scheduler that resolves priority inversion in
geo-distributed IoT monitoring by separating semantic urgency (alarms)
from geographic priority (zones).
"""

from typing import Any, Dict, List

import numpy as np

from .results import SchedulerResult
from .token_bucket import TIME_TOLERANCE, TokenBucket
from .band_classifier import BAND_ALARM, BAND_HIGH, BAND_STANDARD, BAND_BACKGROUND, BandClassifier
from .adaptive_token_bucket import AdaptiveTokenBucket
from .alarm_rate_monitor import AlarmRateMonitor
from .device_fair_queue import DeviceFairQueue
from .triage4_config import TRIAGE4Config
from .source_aware_queue import SourceAwareQueue


def time_equal(t1: float, t2: float) -> bool:
    """Check if two times are equal within tolerance."""
    return abs(t1 - t2) < TIME_TOLERANCE


class TRIAGE4Scheduler:
    """
    TRIAGE/4 scheduler.

    Four-band hierarchical scheduler with token-bucket resource reservation
    and per-device fair queuing within each band.

    Architecture:
        - Band 0 (ALARM): Emergency messages, strict priority, no token bucket
        - Band 1 (HIGH): High-priority zones, token-constrained (Q_H/P_H)
        - Band 2 (STANDARD): Standard zones, token-constrained (Q_S/P_S)
        - Band 3 (BACKGROUND): Low-priority zones, token-constrained (Q_B/P_B)

    Scheduling policy:
        1. ALARM band always served first (if non-empty)
        2. HIGH band served if non-empty AND tokens available
        3. STANDARD band served if non-empty AND tokens available
        4. BACKGROUND band served if non-empty AND tokens available
        5. Server idles if all queues empty or token-exhausted

    Key features:
        - Semantic override: Alarms bypass geographic priority
        - Per-device fairness: Round-robin prevents monopolization
        - Bandwidth guarantees: Token buckets ensure minimum service rates
        - Zero starvation: All bands eventually served

    Example:
        >>> config = TRIAGE4Config(
        ...     high_zone_max=1,
        ...     standard_zone_max=3,
        ...     high_token_budget=10,
        ...     service_rate=20.0
        ... )
        >>> scheduler = TRIAGE4Scheduler(config)
        >>> result = scheduler.schedule(
        ...     arrival_times=[0.0, 0.1, 0.2],
        ...     device_ids=["sensor_1", "sensor_2", "sensor_1"],
        ...     zone_priorities=[0, 5, 2],
        ...     is_alarm=[False, True, False]
        ... )
        >>> result.waiting_times  # Per-job waiting times
        array([...])
    """

    def __init__(self, config: TRIAGE4Config, scheduler_seed: int | None = None):
        """
        Initialize TRIAGE/4 scheduler.

        Args:
            config: Scheduler configuration
            scheduler_seed: Random seed for reproducible service times
                          (None for non-deterministic)
        """
        self.cfg = config
        self.classifier = BandClassifier(
            high_zone_max=config.high_zone_max,
            standard_zone_max=config.standard_zone_max,
        )

        # Alarm protection (optional adaptive limiter)
        self.alarm_protection_enabled = config.enable_alarm_protection
        if self.alarm_protection_enabled:
            self.alarm_queue = SourceAwareQueue()
            self.alarm_monitor = AlarmRateMonitor(
                window_duration=config.alarm_window_duration,
                abnormal_threshold=config.alarm_abnormal_threshold,
                deactivation_threshold=config.alarm_deactivation_threshold,
                min_observations=config.alarm_min_observations,
            )
            self.alarm_bucket = AdaptiveTokenBucket(
                budget=config.alarm_limit_budget,
                period=config.alarm_limit_period,
                burst_capacity=config.alarm_burst_capacity,
            )
        else:
            self.alarm_queue = DeviceFairQueue()
            self.alarm_monitor = None
            self.alarm_bucket = None

        # Three per-device fair queues (one per non-alarm band)
        self.high_queue = DeviceFairQueue()
        self.standard_queue = DeviceFairQueue()
        self.background_queue = DeviceFairQueue()

        # Three token buckets (ALARM has none - always served)
        self.high_bucket = TokenBucket(
            budget=config.high_token_budget,
            period=config.high_token_period,
            burst_capacity=int(config.high_token_budget * config.high_burst_multiplier),
        )
        self.standard_bucket = TokenBucket(
            budget=config.standard_token_budget,
            period=config.standard_token_period,
            burst_capacity=int(config.standard_token_budget * config.standard_burst_multiplier),
        )
        self.background_bucket = TokenBucket(
            budget=config.background_token_budget,
            period=config.background_token_period,
            burst_capacity=int(config.background_token_budget * config.background_burst_multiplier),
        )

        # Service time RNG (independent stream for reproducibility)
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
        Run discrete-event simulation with four-band scheduling.

        Args:
            arrival_times: Job arrival times in seconds (must be sorted)
            device_ids: Device identifier for each job (e.g., "sensor_42")
            zone_priorities: Geographic zone priority for each job (0=highest)
            is_alarm: Semantic urgency flag for each job (True=emergency)

        Returns:
            SchedulerResult with:
                - waiting_times: Per-job waiting times (arrival to service start)
                - e2e_times: Per-job end-to-end times (arrival to completion)
                - priorities: Band assignments [0=ALARM, 1=HIGH, 2=STANDARD, 3=BACKGROUND]
                - metadata: Scheduler configuration and statistics

        Raises:
            ValueError: If input lists have mismatched lengths or invalid values
        """
        n = len(arrival_times)
        self._validate_inputs(arrival_times, device_ids, zone_priorities, is_alarm)

        # Classify all messages into bands
        bands = [
            self.classifier.classify(zone_priorities[i], is_alarm[i])
            for i in range(n)
        ]

        # Initialize simulation state
        state = self._initialize_state(n, arrival_times)

        # Discrete-event simulation loop
        while state["completed"] + state["dropped"] < n:
            # 1. Advance to next event time
            state["current_time"] = self._next_event_time(arrival_times, state)

            # 2. Refill token buckets based on current time
            self._refill_tokens(state["current_time"])

            # 3. Process all arrivals at current time
            self._handle_arrivals(arrival_times, bands, device_ids, zone_priorities, state)

            # 4. Handle job completion if current job finishes
            if self._handle_completion(state):
                state["completed"] += 1

            # 5. Dispatch next job if server idle
            if state["current_job"] is None:
                self._dispatch_next(state)

            # Fast exit: all arrived, server idle, queues empty
            if (
                state["arrival_idx"] >= n
                and state["current_job"] is None
                and self._all_queues_empty()
            ):
                break

        return self._build_result(state, bands)

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

    def _initialize_state(self, n: int, arrival_times: List[float]) -> Dict[str, Any]:
        """Initialize simulation state."""
        return {
            "n": n,
            "arrival_times": arrival_times,  # Store for waiting time calculation
            "current_time": 0.0,
            "arrival_idx": 0,  # Next arrival to process
            "completed": 0,  # Jobs completed
            "dropped": 0,  # Dropped alarms (adaptive protection)
            "current_job": None,  # (job_idx, band) or None
            "current_end": float("inf"),  # Completion time of current job
            "waiting_times": [0.0] * n,
            "e2e_times": [0.0] * n,
        }

    def _next_event_time(
        self,
        arrival_times: List[float],
        state: Dict[str, Any],
    ) -> float:
        """
        Determine next event time (arrival, completion, or token refill).

        Returns:
            Next event time in simulation
        """
        next_arrival = float("inf")
        if state["arrival_idx"] < state["n"]:
            next_arrival = arrival_times[state["arrival_idx"]]

        next_completion = state["current_end"]

        # Consider token refill times (important when tokens exhausted)
        refill_times = [
            self.high_bucket.get_next_refill_time(),
            self.standard_bucket.get_next_refill_time(),
            self.background_bucket.get_next_refill_time(),
        ]
        if self.alarm_protection_enabled and self.alarm_bucket is not None:
            refill_times.append(self.alarm_bucket.get_next_refill_time())

        next_refill = min(refill_times)

        # Next event is whichever comes first
        return min(next_arrival, next_completion, next_refill)

    def _refill_tokens(self, current_time: float) -> None:
        """Refill all token buckets based on current time."""
        # Don't refill at infinity (happens when no more events)
        if current_time == float("inf"):
            return

        self.high_bucket.refill(current_time)
        self.standard_bucket.refill(current_time)
        self.background_bucket.refill(current_time)
        if self.alarm_protection_enabled and self.alarm_bucket is not None:
            # Adaptive bucket refills only when active
            if self.alarm_bucket.active:
                self.alarm_bucket.bucket.refill(current_time)

    def _handle_arrivals(
        self,
        arrival_times: List[float],
        bands: List[int],
        device_ids: List[str],
        zone_priorities: List[int],
        state: Dict[str, Any],
    ) -> None:
        """
        Process all arrivals at current time.

        Enqueues jobs to appropriate band based on classification.
        """
        current_time = state["current_time"]
        arrival_idx = state["arrival_idx"]
        n = state["n"]

        # Process all arrivals at current time (with floating-point tolerance)
        while arrival_idx < n and time_equal(arrival_times[arrival_idx], current_time):
            job_idx = arrival_idx
            band = bands[job_idx]
            device_id = device_ids[job_idx]
            zone_priority = zone_priorities[job_idx]

            # Enqueue to appropriate band queue
            if band == BAND_ALARM:
                if not self.alarm_protection_enabled:
                    self.alarm_queue.enqueue(job_idx, device_id)
                else:
                    # Record and update protection state
                    self.alarm_monitor.record_arrival(current_time, str(zone_priority))
                    if self.alarm_monitor.is_abnormal(current_time):
                        self.alarm_bucket.activate(current_time)
                    elif self.alarm_bucket.active and self.alarm_monitor.is_recovered(
                        current_time
                    ):
                        self.alarm_bucket.deactivate()

                    # Apply adaptive token bucket if active
                    allowed = self.alarm_bucket.consume(current_time)
                    if allowed:
                        self.alarm_queue.enqueue(job_idx, device_id, zone_priority)
                    else:
                        state["dropped"] += 1
                        state["waiting_times"][job_idx] = 0.0
                        state["e2e_times"][job_idx] = 0.0
            elif band == BAND_HIGH:
                self.high_queue.enqueue(job_idx, device_id)
            elif band == BAND_STANDARD:
                self.standard_queue.enqueue(job_idx, device_id)
            elif band == BAND_BACKGROUND:
                self.background_queue.enqueue(job_idx, device_id)

            arrival_idx += 1

        state["arrival_idx"] = arrival_idx

    def _handle_completion(self, state: Dict[str, Any]) -> bool:
        """
        Handle job completion if current job finishes.

        Returns:
            True if a job completed, False otherwise
        """
        current_time = state["current_time"]

        if state["current_job"] is not None and time_equal(
            state["current_end"], current_time
        ):
            # Current job completed
            state["current_job"] = None
            state["current_end"] = float("inf")
            return True

        return False

    def _dispatch_next(self, state: Dict[str, Any]) -> None:
        """
        Select and dispatch next message using four-band priority hierarchy.

        Priority order:
            1. ALARM (if non-empty) - always served
            2. HIGH (if non-empty AND tokens available)
            3. STANDARD (if non-empty AND tokens available)
            4. BACKGROUND (if non-empty AND tokens available)

        If no job can be dispatched (queues empty or token-exhausted),
        server remains idle.
        """
        current_time = state["current_time"]

        # Priority 1: ALARM band (always served, no token constraint)
        if not self.alarm_queue.is_empty():
            job_idx = self.alarm_queue.dequeue()
            self._start_job(job_idx, BAND_ALARM, current_time, state)
            return

        # Priority 2: HIGH band (token-constrained)
        if not self.high_queue.is_empty():
            if self.high_bucket.consume():
                job_idx = self.high_queue.dequeue()
                self._start_job(job_idx, BAND_HIGH, current_time, state)
                return
            # Token exhausted, try next band

        # Priority 3: STANDARD band (token-constrained)
        if not self.standard_queue.is_empty():
            if self.standard_bucket.consume():
                job_idx = self.standard_queue.dequeue()
                self._start_job(job_idx, BAND_STANDARD, current_time, state)
                return
            # Token exhausted, try next band

        # Priority 4: BACKGROUND band (token-constrained)
        if not self.background_queue.is_empty():
            if self.background_bucket.consume():
                job_idx = self.background_queue.dequeue()
                self._start_job(job_idx, BAND_BACKGROUND, current_time, state)
                return
            # Token exhausted, server idles

        # All queues empty or token-exhausted - server idles

    def _start_job(
        self,
        job_idx: int,
        band: int,
        current_time: float,
        state: Dict[str, Any],
    ) -> None:
        """
        Start processing a job.

        Updates waiting time, generates service time, and schedules completion.
        """
        # Compute waiting time (arrival to service start)
        arrival_time = state["arrival_times"][job_idx]
        state["waiting_times"][job_idx] = current_time - arrival_time

        # Generate service time (exponential distribution)
        service_time = self.service_rng.exponential(1 / self.cfg.service_rate)

        # Compute end-to-end time
        state["e2e_times"][job_idx] = state["waiting_times"][job_idx] + service_time

        # Update server state
        state["current_job"] = (job_idx, band)
        state["current_end"] = current_time + service_time

    def _all_queues_empty(self) -> bool:
        """Check if all band queues are empty."""
        return (
            self.alarm_queue.is_empty()
            and self.high_queue.is_empty()
            and self.standard_queue.is_empty()
            and self.background_queue.is_empty()
        )

    def _build_result(
        self,
        state: Dict[str, Any],
        bands: List[int],
    ) -> SchedulerResult:
        """
        Build SchedulerResult from simulation state.

        Returns:
            SchedulerResult with metrics and metadata
        """
        return SchedulerResult(
            waiting_times=np.array(state["waiting_times"]),
            e2e_times=np.array(state["e2e_times"]),
            priorities=bands,
            metadata={
                "scheduler": "TRIAGE/4",
                "alarm_protection_enabled": self.alarm_protection_enabled,
                "alarm_dropped": state.get("dropped", 0),
                "high_zone_max": self.cfg.high_zone_max,
                "standard_zone_max": self.cfg.standard_zone_max,
                "high_token_budget": self.cfg.high_token_budget,
                "high_token_period": self.cfg.high_token_period,
                "standard_token_budget": self.cfg.standard_token_budget,
                "standard_token_period": self.cfg.standard_token_period,
                "background_token_budget": self.cfg.background_token_budget,
                "background_token_period": self.cfg.background_token_period,
                "service_rate": self.cfg.service_rate,
                "final_high_tokens": self.high_bucket.tokens,
                "final_standard_tokens": self.standard_bucket.tokens,
                "final_background_tokens": self.background_bucket.tokens,
                "final_alarm_tokens": (
                    self.alarm_bucket.tokens
                    if self.alarm_bucket is not None
                    else None
                ),
            },
        )

    def __repr__(self) -> str:
        return (
            f"TRIAGE4Scheduler("
            f"high_zone≤{self.cfg.high_zone_max}, "
            f"standard_zone≤{self.cfg.standard_zone_max}, "
            f"service_rate={self.cfg.service_rate})"
        )

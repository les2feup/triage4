"""
Online egress dispatcher for a live TRIAGE/4 broker.

`Triage4EgressDispatcher` is the push/pop counterpart of `TRIAGE4Scheduler`:
where the offline scheduler consumes whole arrival arrays inside a discrete-
event loop, this dispatcher accepts one message at a time as it arrives and
hands back the next message to transmit when the egress server is free. It
carries no simulation loop and no service-time RNG — the real transmitter
supplies timing.

Parity contract:
    - `enqueue` is the body of `TRIAGE4Scheduler._handle_arrivals` minus array
      indexing (it stores an opaque integer handle instead of a job index).
    - `select_next` is `TRIAGE4Scheduler._dispatch_next` minus `_start_job`'s
      stochastic service time.
    These methods are siblings: if the offline enqueue/AAP or band-selection
    logic changes, mirror the change here. The `within_band_fifo` ablation is
    intentionally omitted — the hardware set is FIFO + Strict + TRIAGE/4, none
    of which need in-band FIFO on the deployable dispatcher.

Relative-clock requirement (mandatory):
    Callers must pass a *relative* time `now`, seconds since broker startup
    (`now = monotonic() - t0`, starting near 0). `TokenBucket.next_refill`
    initializes to `period` relative to t=0, so a raw `monotonic()` value (a
    large absolute epoch) would make the first `refill` jump thousands of
    periods and clamp every bucket to its burst capacity, silently defeating
    rate limiting. The guard below rejects absolute-scale inputs in debug
    builds (disabled under `python -O`).
"""

from typing import Optional

from .triage4_config import TRIAGE4Config
from .band_classifier import (
    BandClassifier,
    BAND_ALARM,
    BAND_HIGH,
    BAND_STANDARD,
    BAND_BACKGROUND,
)
from .device_fair_queue import DeviceFairQueue
from .token_bucket import TokenBucket
from .adaptive_token_bucket import AdaptiveTokenBucket
from .alarm_rate_monitor import AlarmRateMonitor
from .source_aware_queue import SourceAwareQueue


class Triage4EgressDispatcher:
    """Online form of `TRIAGE4Scheduler` for a live broker egress queue.

    Stores opaque integer message handles; the caller maps handle -> payload.
    All state mirrors `TRIAGE4Scheduler.__init__` so the two stay in lockstep.
    """

    def __init__(self, config: TRIAGE4Config):
        self.cfg = config
        self.classifier = BandClassifier(
            high_zone_max=config.high_zone_max,
            standard_zone_max=config.standard_zone_max,
            disable_semantic_override=config.disable_semantic_override,
        )
        self.aap = config.enable_alarm_protection

        # ALARM band: source-aware fair queue + adaptive limiter when AAP is on,
        # otherwise a plain per-device fair queue with no rate shedding.
        if self.aap:
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

        # Non-alarm bands: per-device fair queues with independent token buckets.
        self.high_queue = DeviceFairQueue()
        self.standard_queue = DeviceFairQueue()
        self.background_queue = DeviceFairQueue()
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

    def enqueue(
        self,
        handle: int,
        device_id: str,
        zone_priority: int,
        is_alarm: bool,
        now: float,
    ) -> bool:
        """Admit one message at relative time `now`.

        Returns False iff Adaptive Alarm Protection rate-shed the message
        (dropped); True when it was enqueued. Mirrors `_handle_arrivals`.
        """
        assert now < 1e6, (
            "Triage4EgressDispatcher.enqueue received an absolute-scale time; "
            "pass a relative clock (monotonic() - t0)."
        )
        band = self.classifier.classify(zone_priority, is_alarm)

        if band == BAND_ALARM:
            if not self.aap:
                self.alarm_queue.enqueue(handle, device_id)
                return True
            # Update AAP detection state on transitions only, then rate-shed.
            self.alarm_monitor.record_arrival(now, str(zone_priority))
            if self.alarm_monitor.is_abnormal(now):
                if not self.alarm_bucket.active:
                    self.alarm_bucket.activate(now)
            elif self.alarm_bucket.active and self.alarm_monitor.is_recovered(now):
                self.alarm_bucket.deactivate()
            if self.alarm_bucket.consume(now):
                self.alarm_queue.enqueue(handle, device_id, zone_priority)
                return True
            return False

        target = {
            BAND_HIGH: self.high_queue,
            BAND_STANDARD: self.standard_queue,
            BAND_BACKGROUND: self.background_queue,
        }[band]
        target.enqueue(handle, device_id)
        return True

    def select_next(self, now: float) -> Optional[int]:
        """Return the next handle to transmit, or None if the server should idle.

        Refills the non-alarm buckets at relative time `now`, then applies the
        four-band priority hierarchy: ALARM (unconstrained) first, then HIGH,
        STANDARD, BACKGROUND, each gated on token availability. Mirrors
        `_dispatch_next` minus the service-time RNG.
        """
        assert now < 1e6, (
            "Triage4EgressDispatcher.select_next received an absolute-scale time; "
            "pass a relative clock (monotonic() - t0)."
        )
        self.high_bucket.refill(now)
        self.standard_bucket.refill(now)
        self.background_bucket.refill(now)

        if not self.alarm_queue.is_empty():
            return self.alarm_queue.dequeue()

        for queue, bucket in (
            (self.high_queue, self.high_bucket),
            (self.standard_queue, self.standard_bucket),
            (self.background_queue, self.background_bucket),
        ):
            if not queue.is_empty() and (self.cfg.disable_token_buckets or bucket.consume()):
                return queue.dequeue()
        return None

    def is_empty(self) -> bool:
        """True when no band holds a pending message."""
        return (
            self.alarm_queue.is_empty()
            and self.high_queue.is_empty()
            and self.standard_queue.is_empty()
            and self.background_queue.is_empty()
        )

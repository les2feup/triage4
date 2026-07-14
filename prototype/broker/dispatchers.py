"""
Research-baseline online egress dispatchers for the broker.

These are the prototype-only comparators that frame TRIAGE/4's on-hardware
behaviour. They implement the same online contract as
``triage4.Triage4EgressDispatcher`` so the broker can drive all three
uniformly without special-casing:

    enqueue(handle, device_id, zone_priority, is_alarm, now) -> bool
        Admit one message. Returns False only if the dispatcher sheds it; these
        baselines never shed, so they always return True.
    select_next(now) -> Optional[int]
        Return the next handle to transmit, or None when idle.
    is_empty() -> bool

The baselines keep no alarm state: they exist to show what TRIAGE/4's semantic
override and per-band token shaping add over plain queueing disciplines. FIFO and
Strict mirror the simulation's weakest comparators; WFQ mirrors its *strongest*
(the one that ties TRIAGE/4 in C1 and comes within 4% in S2), so the hardware
result is not left resting on the two easiest opponents.
"""

import heapq
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from triage4 import BAND_BACKGROUND, BAND_HIGH, BAND_STANDARD, BandClassifier, TRIAGE4Config
from triage4.token_bucket import TokenBucket

QUANTUM: float = 1.0   # DRR: added to a device's deficit on each visit
MSG_COST: float = 1.0  # DRR: uniform message cost (no packet-size model)


class FifoEgressDispatcher:
    """First-in-first-out egress: a single queue, no bands, no priority.

    The simplest possible discipline — every message shares one FIFO queue and
    is served in arrival order regardless of zone or alarm flag.
    """

    def __init__(self) -> None:
        self._queue: Deque[int] = deque()

    def enqueue(self, handle: int, device_id: str, zone_priority: int,
                is_alarm: bool, now: float) -> bool:
        self._queue.append(handle)
        return True

    def select_next(self, now: float) -> Optional[int]:
        return self._queue.popleft() if self._queue else None

    def is_empty(self) -> bool:
        return not self._queue


class StrictEgressDispatcher:
    """Strict geographic-priority egress — the priority-inversion baseline.

    One FIFO queue per geographic zone priority; ``select_next`` always serves
    the lowest zone number (highest geographic priority) first. The alarm flag
    is ignored, so a critical alarm from a low-priority zone waits behind
    routine traffic from any higher-priority zone — exactly the inversion
    TRIAGE/4's semantic override is designed to resolve.
    """

    def __init__(self) -> None:
        # zone_priority -> FIFO of handles; served in ascending zone order.
        self._zones: Dict[int, Deque[int]] = {}

    def enqueue(self, handle: int, device_id: str, zone_priority: int,
                is_alarm: bool, now: float) -> bool:
        self._zones.setdefault(zone_priority, deque()).append(handle)
        return True

    def select_next(self, now: float) -> Optional[int]:
        for zone in sorted(self._zones):
            queue = self._zones[zone]
            if queue:
                handle = queue.popleft()
                if not queue:
                    del self._zones[zone]
                return handle
        return None

    def is_empty(self) -> bool:
        return all(not q for q in self._zones.values())


class WfqEgressDispatcher:
    """Pure geographic per-device Weighted Fair Queueing — the strongest baseline.

    The online counterpart of ``assessment.baselines.wfq_scheduler.WFQScheduler``:
    each device is a WFQ flow with weight φ(z) = 1/(z+1), so a message's virtual
    finish time is

        F = max(virtual_clock, F_device_prev) + (z + 1)

    and messages are served in ascending F. The virtual clock tracks real time
    while the server is busy and is frozen while it is idle (P-GPS approximation)
    — here it advances only on a successful dispatch, which is the same rule.

    ``is_alarm`` is ignored in the scheduling decision. That is the point: WFQ
    gives per-device fairness weighted by geography but has no semantic override,
    so a zone-5 alarm (cost 6) is still served behind a zone-0 routine message
    (cost 1). It is the controlled absence of TRIAGE/4's component A, and unlike
    FIFO and Strict it is a genuinely competitive scheduler — which is why it
    belongs in the hardware set rather than only in simulation.
    """

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, int, str]] = []
        self._device_vft: Dict[str, float] = {}
        self._virtual_clock: float = 0.0
        self._seq: int = 0

    def enqueue(self, handle: int, device_id: str, zone_priority: int,
                is_alarm: bool, now: float) -> bool:
        cost = zone_priority + 1  # inverse weight: zone 0 -> 1, zone 5 -> 6
        vft = max(self._virtual_clock, self._device_vft.get(device_id, 0.0)) + cost
        # The arrival sequence breaks virtual-finish-time ties deterministically.
        heapq.heappush(self._heap, (vft, self._seq, handle, device_id))
        self._seq += 1
        return True

    def select_next(self, now: float) -> Optional[int]:
        if not self._heap:
            return None  # idle: the virtual clock stays frozen
        vft, _seq, handle, device_id = heapq.heappop(self._heap)
        self._device_vft[device_id] = vft
        self._virtual_clock = max(self._virtual_clock, now)
        return handle

    def is_empty(self) -> bool:
        return not self._heap


class DrrEgressDispatcher:
    """Deficit Round Robin across devices — pure per-device fairness.

    The online counterpart of ``assessment.baselines.drr_scheduler``. One queue per
    device, visited round-robin; each visit adds QUANTUM to the device's deficit and
    serves a message while the deficit covers MSG_COST. Both zone_priority and
    is_alarm are ignored.

    With Q = cost = 1 the deficit never accumulates, so DRR degenerates to plain
    round-robin. The deficit is kept anyway because it is the cited algorithm and the
    degeneracy is a property of this workload, not of the code.

    In simulation this collapses onto FIFO in both hardware scenarios (C3 and S1,
    identical to the decimal). Running it here therefore tests a falsifiable
    prediction: if the hardware reproduces that collapse, the testbed is behaving as
    the model says it should; if it does not, the divergence is worth knowing about.
    """

    def __init__(self) -> None:
        self._queues: Dict[str, Deque[int]] = {}
        self._deficit: Dict[str, float] = {}
        self._active: Deque[str] = deque()
        self._active_set: Set[str] = set()

    def enqueue(self, handle: int, device_id: str, zone_priority: int,
                is_alarm: bool, now: float) -> bool:
        self._queues.setdefault(device_id, deque()).append(handle)
        self._deficit.setdefault(device_id, 0.0)
        if device_id not in self._active_set:
            self._active.append(device_id)
            self._active_set.add(device_id)
        return True

    def select_next(self, now: float) -> Optional[int]:
        # One full round: every active device gets a chance to accumulate deficit.
        for _ in range(len(self._active)):
            if not self._active:
                return None
            device = self._active[0]
            self._deficit[device] += QUANTUM
            if self._queues[device] and self._deficit[device] >= MSG_COST:
                handle = self._queues[device].popleft()
                self._deficit[device] -= MSG_COST
                if not self._queues[device]:
                    self._active.popleft()
                    self._active_set.discard(device)
                else:
                    self._active.rotate(-1)  # back of the queue for the next round
                return handle
            self._active.rotate(-1)
        return None

    def is_empty(self) -> bool:
        return all(not q for q in self._queues.values())


class TbpEgressDispatcher:
    """Token Bucket Priority — geographic bands + token buckets, no semantic override.

    The online counterpart of ``assessment.baselines.tbp_scheduler``. Three bands by
    zone (HIGH → STANDARD → BACKGROUND), FIFO within a band, each band gated by its
    own token bucket; a band that cannot pay falls through to the next.

    The crucial line is that a message is banded by ``classify(zone, False)`` — the
    alarm flag is ignored when queueing — so a zone-5 alarm is placed in BACKGROUND
    and waits behind every HIGH and STANDARD message.

    This is TRIAGE/4 with components B and C but without A (the semantic override) or
    E (AAP). It is therefore the sharpest baseline in the set: the difference between
    it and TRIAGE/4 on hardware is the semantic override, measured in isolation.
    """

    def __init__(self, config: TRIAGE4Config) -> None:
        self._classifier = BandClassifier(
            high_zone_max=config.high_zone_max,
            standard_zone_max=config.standard_zone_max,
        )
        self._queues: Dict[int, Deque[int]] = {
            BAND_HIGH: deque(), BAND_STANDARD: deque(), BAND_BACKGROUND: deque(),
        }
        self._buckets: Dict[int, TokenBucket] = {
            BAND_HIGH: TokenBucket(
                budget=config.high_token_budget,
                period=config.high_token_period,
                burst_capacity=int(config.high_token_budget * config.high_burst_multiplier)),
            BAND_STANDARD: TokenBucket(
                budget=config.standard_token_budget,
                period=config.standard_token_period,
                burst_capacity=int(config.standard_token_budget * config.standard_burst_multiplier)),
            BAND_BACKGROUND: TokenBucket(
                budget=config.background_token_budget,
                period=config.background_token_period,
                burst_capacity=int(config.background_token_budget * config.background_burst_multiplier)),
        }

    def enqueue(self, handle: int, device_id: str, zone_priority: int,
                is_alarm: bool, now: float) -> bool:
        # is_alarm is deliberately NOT passed: banding is geographic only, which is
        # what puts a low-zone alarm behind high-zone routine traffic.
        band = self._classifier.classify(zone_priority, False)
        self._queues[band].append(handle)
        return True

    def select_next(self, now: float) -> Optional[int]:
        assert now < 1e6, (
            "TbpEgressDispatcher expects a clock relative to broker start "
            "(monotonic() - t0); a raw monotonic value clamps the buckets to burst.")
        for band in (BAND_HIGH, BAND_STANDARD, BAND_BACKGROUND):
            self._buckets[band].refill(now)
        for band in (BAND_HIGH, BAND_STANDARD, BAND_BACKGROUND):
            if self._queues[band] and self._buckets[band].consume():
                return self._queues[band].popleft()
        return None

    def is_empty(self) -> bool:
        return all(not q for q in self._queues.values())

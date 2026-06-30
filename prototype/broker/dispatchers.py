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

The baselines deliberately keep no token or alarm state: they exist to show
what TRIAGE/4's semantic override and per-band token shaping add over plain
queueing disciplines, mirroring the FIFO and strict-geographic-priority
baselines from the simulation (R1.2 breadth stays simulation-only). ``now`` is
accepted for contract uniformity and ignored — neither baseline is time-driven.
"""

from collections import deque
from typing import Deque, Dict, List, Optional


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

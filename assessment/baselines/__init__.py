"""
Baseline schedulers for TRIAGE/4 evaluation.

Provides comparison schedulers to demonstrate TRIAGE/4 advantages:
- StrictPriorityScheduler: Geographic priority only (shows alarm delay problem)
- FIFOScheduler: No priority (lower bound baseline)
- WFQScheduler: Per-device weighted fair queueing (continuous, no bands)
- DRRScheduler: Deficit round-robin across devices (global fairness, no priority)
- TokenBucketPriorityScheduler: Four-band strict priority with per-band tokens, FIFO within band

The SCHEDULER_REGISTRY maps canonical name → SchedulerEntry for all schedulers
that conform to the plain factory(service_rate, seed) signature. TRIAGE/4 and
its ablation variants are constructed separately because they require additional
parameters (config_factory, AAP flag, scenario-specific token overrides).
"""

from collections import OrderedDict
from typing import Any, Callable, NamedTuple

from .fifo_scheduler import FIFOScheduler
from .strict_priority_scheduler import StrictPriorityScheduler
from .wfq_scheduler import WFQScheduler
from .drr_scheduler import DRRScheduler
from .tbp_scheduler import TokenBucketPriorityScheduler


class SchedulerEntry(NamedTuple):
    """Registry entry pairing a display name with a two-argument factory."""

    name: str
    factory: Callable[[float, int], Any]


# Registry of all non-TRIAGE/4 schedulers. Ordered so that Strict and FIFO
# (the original baselines) always appear first, followed by the three new
# classical baselines whose compositional relationship to TRIAGE/4 is the
# core narrative of the Stage 3 revision.
SCHEDULER_REGISTRY: OrderedDict[str, SchedulerEntry] = OrderedDict(
    [
        ("Strict", SchedulerEntry("Strict", lambda sr, seed: StrictPriorityScheduler(sr, seed))),
        ("FIFO", SchedulerEntry("FIFO", lambda sr, seed: FIFOScheduler(sr, seed))),
        ("WFQ", SchedulerEntry("WFQ", lambda sr, seed: WFQScheduler(sr, seed))),
        ("DRR", SchedulerEntry("DRR", lambda sr, seed: DRRScheduler(sr, seed))),
        ("TBP", SchedulerEntry("TBP", lambda sr, seed: TokenBucketPriorityScheduler(sr, seed))),
    ]
)

__all__ = [
    "StrictPriorityScheduler",
    "FIFOScheduler",
    "WFQScheduler",
    "DRRScheduler",
    "TokenBucketPriorityScheduler",
    "SchedulerEntry",
    "SCHEDULER_REGISTRY",
]

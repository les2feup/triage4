"""
Baseline schedulers for TRIAGE/4 evaluation.

Provides comparison schedulers to demonstrate TRIAGE/4 advantages:
- StrictPriorityScheduler: Geographic priority only (shows alarm delay problem)
- FIFOScheduler: No priority (lower bound baseline)
"""

from .fifo_scheduler import FIFOScheduler
from .strict_priority_scheduler import StrictPriorityScheduler

__all__ = [
    "StrictPriorityScheduler",
    "FIFOScheduler",
]

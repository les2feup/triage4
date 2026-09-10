"""
Data structures for scheduler simulation results.

Provides standardized containers for metrics and performance data.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from triage4.results import SchedulerResult as _CoreSchedulerResult


@dataclass
class SchedulerResult:
    """
    Container for discrete-event simulation results.

    Attributes:
        waiting_times: Per-job waiting time (arrival to service start)
        e2e_times: Per-job end-to-end time (arrival to completion)
        priorities: Priority class for each job (optional, for analysis)
        delivered: Per-job service flag; False marks a job the scheduler dropped
        metadata: Additional scheduler-specific data (e.g., budget exhaustion events)

    Mirrors triage4.results.SchedulerResult. The baselines never drop, so they
    leave `delivered` as None, which means every job was served.
    """

    waiting_times: np.ndarray
    e2e_times: np.ndarray
    priorities: Optional[List[int]] = None
    delivered: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        """Convert lists to numpy arrays if needed."""
        if not isinstance(self.waiting_times, np.ndarray):
            self.waiting_times = np.array(self.waiting_times)
        if not isinstance(self.e2e_times, np.ndarray):
            self.e2e_times = np.array(self.e2e_times)
        if self.delivered is not None and not isinstance(self.delivered, np.ndarray):
            self.delivered = np.array(self.delivered, dtype=bool)

    @property
    def n_jobs(self) -> int:
        """Total number of jobs processed."""
        return len(self.waiting_times)

    def _latency_values(
        self, values: np.ndarray, priority_class: Optional[int]
    ) -> np.ndarray:
        mask = np.ones(len(values), dtype=bool)
        if self.delivered is not None:
            mask &= self.delivered
        if priority_class is not None:
            if self.priorities is None:
                raise ValueError("Cannot filter by priority: priorities not recorded")
            mask &= np.asarray(self.priorities) == priority_class
        return values[mask]

    def avg_waiting_time(self, priority_class: Optional[int] = None) -> float:
        """
        Compute average waiting time, optionally filtered by priority.

        Args:
            priority_class: If specified, compute average only for this class

        Returns:
            Mean waiting time
        """
        values = self._latency_values(self.waiting_times, priority_class)
        return float(np.mean(values)) if values.size else 0.0

    def avg_e2e_time(self, priority_class: Optional[int] = None) -> float:
        """
        Compute average end-to-end time, optionally filtered by priority.

        Args:
            priority_class: If specified, compute average only for this class

        Returns:
            Mean end-to-end time
        """
        values = self._latency_values(self.e2e_times, priority_class)
        return float(np.mean(values)) if values.size else 0.0

    def per_class_waiting_times(self) -> Dict[int, float]:
        """
        Compute average waiting time for each priority class.

        Returns:
            Dict mapping priority class to average waiting time
        """
        if self.priorities is None:
            raise ValueError(
                "Cannot compute per-class metrics: priorities not recorded"
            )

        unique_classes = sorted(set(self.priorities))
        return {prio: self.avg_waiting_time(prio) for prio in unique_classes}

    def per_class_e2e_times(self) -> Dict[int, float]:
        """
        Compute average end-to-end time for each priority class.

        Returns:
            Dict mapping priority class to average E2E time
        """
        if self.priorities is None:
            raise ValueError(
                "Cannot compute per-class metrics: priorities not recorded"
            )

        unique_classes = sorted(set(self.priorities))
        return {prio: self.avg_e2e_time(prio) for prio in unique_classes}

    def percentile_waiting_time(
        self, percentile: float, priority_class: Optional[int] = None
    ) -> float:
        """
        Compute percentile of waiting times.

        Args:
            percentile: Percentile to compute (0-100)
            priority_class: If specified, compute only for this class

        Returns:
            Percentile value of waiting times
        """
        if not 0 <= percentile <= 100:
            raise ValueError(f"Percentile must be in [0, 100], got {percentile}")

        values = self._latency_values(self.waiting_times, priority_class)
        if not values.size:
            return 0.0
        return float(np.percentile(values, percentile))

    def percentile_e2e_time(
        self, percentile: float, priority_class: Optional[int] = None
    ) -> float:
        """
        Compute percentile of end-to-end times.

        Args:
            percentile: Percentile to compute (0-100)
            priority_class: If specified, compute only for this class

        Returns:
            Percentile value of E2E times
        """
        if not 0 <= percentile <= 100:
            raise ValueError(f"Percentile must be in [0, 100], got {percentile}")

        values = self._latency_values(self.e2e_times, priority_class)
        if not values.size:
            return 0.0
        return float(np.percentile(values, percentile))

    def per_class_percentile_waiting_times(self, percentile: float) -> Dict[int, float]:
        """
        Compute percentile waiting time for each priority class.

        Args:
            percentile: Percentile to compute (0-100)

        Returns:
            Dict mapping priority class to percentile waiting time
        """
        if self.priorities is None:
            raise ValueError(
                "Cannot compute per-class metrics: priorities not recorded"
            )

        unique_classes = sorted(set(self.priorities))
        return {
            prio: self.percentile_waiting_time(percentile, prio)
            for prio in unique_classes
        }

    def per_class_percentile_e2e_times(self, percentile: float) -> Dict[int, float]:
        """
        Compute percentile E2E time for each priority class.

        Args:
            percentile: Percentile to compute (0-100)

        Returns:
            Dict mapping priority class to percentile E2E time
        """
        if self.priorities is None:
            raise ValueError(
                "Cannot compute per-class metrics: priorities not recorded"
            )

        unique_classes = sorted(set(self.priorities))
        return {
            prio: self.percentile_e2e_time(percentile, prio) for prio in unique_classes
        }


# Keep the historical assessment import path while using the core contract.
SchedulerResult = _CoreSchedulerResult

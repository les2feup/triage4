"""
Abstract base class for priority schedulers.

Defines the interface that all scheduler implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from triage4.results import SchedulerResult


class Scheduler(ABC):
    """
    Base class for discrete-event priority schedulers.

    All schedulers operate on the same inputs:
    - Arrival times (sorted)
    - Priority classes (0=highest priority)
    - Service rate parameters

    And produce standardized SchedulerResult outputs.
    """

    def __init__(
        self,
        service_rate: float = 1.0,
        service_variance: bool = True,
        scheduler_seed: Optional[int] = None,
    ):
        """
        Initialize scheduler with service time parameters.

        Args:
            service_rate: Mean service rate μ (jobs per unit time)
            service_variance: If True, use exponential service times.
                            If False, use deterministic service times (1/μ)
            scheduler_seed: Seed for the scheduler's independent RNG stream.
                          If None, uses non-deterministic initialization.
        """
        self.service_rate = service_rate
        self.service_variance = service_variance

        # Independent RNG stream for this scheduler instance
        if scheduler_seed is not None:
            self.service_rng = np.random.default_rng(seed=scheduler_seed)
        else:
            self.service_rng = np.random.default_rng()

    @abstractmethod
    def schedule(
        self, arrival_times: List[float], priorities: List[int]
    ) -> SchedulerResult:
        """
        Run discrete-event simulation with given workload.

        Args:
            arrival_times: Sorted list of job arrival times
            priorities: Priority class for each job (0=highest)

        Returns:
            SchedulerResult containing waiting times and E2E times

        Raises:
            ValueError: If arrival_times and priorities have different lengths
        """
        pass

    def _generate_service_time(self) -> float:
        """
        Generate a single service time according to configuration.

        Uses the scheduler's independent RNG stream to avoid correlation
        with other schedulers or job properties.

        Returns:
            Service time (exponential or deterministic)
        """
        if self.service_variance:
            return self.service_rng.exponential(1 / self.service_rate)
        else:
            return 1 / self.service_rate

    def _validate_inputs(self, arrival_times: List[float], priorities: List[int]):
        """
        Validate scheduler inputs.

        Args:
            arrival_times: Job arrival times
            priorities: Job priority classes

        Raises:
            ValueError: If inputs are invalid
        """
        if len(arrival_times) != len(priorities):
            raise ValueError(
                f"arrival_times ({len(arrival_times)}) and priorities "
                f"({len(priorities)}) must have same length"
            )

        if len(arrival_times) == 0:
            raise ValueError("Must have at least one job")

        if not all(p >= 0 for p in priorities):
            raise ValueError("All priorities must be non-negative")

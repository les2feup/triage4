"""
Metrics computation for TRIAGE/4 evaluation.

Implements metrics from REFACTORING_PLAN.md success criteria:
- Alarm latency (avg, P95)
- Minimum bandwidth guarantee (per-device rate)
- Jain fairness index (per band)
- High-priority overhead
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from .results import SchedulerResult


def jain_fairness_index(values: List[float]) -> float:
    """
    Compute Jain's fairness index.

    JFI = (sum(x_i))^2 / (n * sum(x_i^2))

    Range: [1/n, 1.0]
    - 1.0 = perfect fairness (all values equal)
    - 1/n = maximum unfairness (one gets all)

    Args:
        values: List of metric values (e.g., per-device latencies)

    Returns:
        Jain fairness index in [0, 1]

    Example:
        >>> jain_fairness_index([1.0, 1.0, 1.0])  # Perfect fairness
        1.0
        >>> jain_fairness_index([10.0, 0.0, 0.0])  # Maximum unfairness
        0.333...
    """
    if not values or len(values) == 0:
        return 1.0

    n = len(values)
    sum_x = sum(values)
    sum_x2 = sum(x**2 for x in values)

    if sum_x2 == 0:
        return 1.0

    return (sum_x**2) / (n * sum_x2)


def compute_alarm_metrics(
    result: SchedulerResult,
    arrival_times: List[float],
    is_alarm: List[bool],
) -> Dict[str, float]:
    """
    Compute alarm-specific metrics.

    Metrics:
        - alarm_avg_latency: Mean waiting time for alarm messages
        - alarm_p95_latency: 95th percentile waiting time for alarms
        - alarm_count: Number of alarm messages

    Args:
        result: Scheduler simulation result
        arrival_times: Message arrival times
        is_alarm: Alarm flags for each message

    Returns:
        Dictionary with alarm metrics
    """
    alarm_indices = [i for i, alarm in enumerate(is_alarm) if alarm]

    if not alarm_indices:
        return {
            "alarm_avg_latency": 0.0,
            "alarm_p95_latency": 0.0,
            "alarm_count": 0,
        }

    alarm_waiting_times = [result.waiting_times[i] for i in alarm_indices]

    return {
        "alarm_avg_latency": float(np.mean(alarm_waiting_times)),
        "alarm_p95_latency": float(np.percentile(alarm_waiting_times, 95)),
        "alarm_count": len(alarm_indices),
    }


def compute_bandwidth_metrics(
    result: SchedulerResult,
    arrival_times: List[float],
    device_ids: List[str],
) -> Dict[str, float]:
    """
    Compute per-device bandwidth metrics.

    Metrics:
        - min_device_rate: Minimum messages/sec across all devices
        - avg_device_rate: Average messages/sec across all devices
        - max_device_rate: Maximum messages/sec across all devices

    Args:
        result: Scheduler simulation result
        arrival_times: Message arrival times
        device_ids: Device identifier for each message

    Returns:
        Dictionary with bandwidth metrics
    """
    if not arrival_times or not device_ids:
        return {
            "min_device_rate": 0.0,
            "avg_device_rate": 0.0,
            "max_device_rate": 0.0,
        }

    # Count messages per device
    device_counts = Counter(device_ids)

    # Total simulation makespan: from first arrival to last completion
    completion_times = [
        arrival + result.e2e_times[i] for i, arrival in enumerate(arrival_times)
    ]
    start_time = min(arrival_times) if arrival_times else 0.0
    end_time = max(completion_times) if completion_times else 0.0
    duration = max(end_time - start_time, 0.0)
    if duration == 0:
        duration = 1.0

    # Compute rates (messages per second)
    device_rates = [count / duration for count in device_counts.values()]

    return {
        "min_device_rate": float(min(device_rates)),
        "avg_device_rate": float(np.mean(device_rates)),
        "max_device_rate": float(max(device_rates)),
    }


def compute_fairness_per_band(
    result: SchedulerResult,
    device_ids: List[str],
) -> Dict[str, float]:
    """
    Compute Jain fairness index per band.

    For each band, computes fairness of average waiting times across devices.

    Metrics:
        - band_0_fairness: Fairness in ALARM band
        - band_1_fairness: Fairness in HIGH band
        - band_2_fairness: Fairness in STANDARD band
        - band_3_fairness: Fairness in BACKGROUND band

    Args:
        result: Scheduler simulation result with band assignments
        device_ids: Device identifier for each message

    Returns:
        Dictionary with per-band fairness indices
    """
    fairness_metrics = {}

    # Group messages by band
    for band_id in range(4):
        band_name = ["ALARM", "HIGH", "STANDARD", "BACKGROUND"][band_id]

        # Find messages in this band
        band_indices = [
            i for i, priority in enumerate(result.priorities) if priority == band_id
        ]

        if not band_indices:
            fairness_metrics[f"band_{band_id}_fairness"] = 1.0
            continue

        # Group by device within this band
        device_waiting_times: Dict[str, List[float]] = {}
        for idx in band_indices:
            device_id = device_ids[idx]
            waiting_time = result.waiting_times[idx]

            if device_id not in device_waiting_times:
                device_waiting_times[device_id] = []
            device_waiting_times[device_id].append(waiting_time)

        # Compute average waiting time per device
        avg_waiting_times = [np.mean(times) for times in device_waiting_times.values()]

        # Compute Jain fairness index
        jfi = jain_fairness_index(avg_waiting_times)
        fairness_metrics[f"band_{band_id}_fairness"] = float(jfi)

    return fairness_metrics


def compute_device_fairness(
    result: SchedulerResult,
    device_ids: List[str],
) -> Dict[str, float]:
    """
    Compute global device fairness metrics across all bands.

    Measures how fairly the scheduler treats different devices regardless
    of their priority band. Two complementary metrics:
    - Latency fairness: Do all devices experience similar waiting times?
    - Throughput fairness: Do all devices get similar service rates?

    Metrics:
        - device_latency_fairness: Jain index of per-device avg waiting times
        - device_throughput_fairness: Jain index of per-device message counts

    Args:
        result: Scheduler simulation result
        device_ids: Device identifier for each message

    Returns:
        Dictionary with device fairness metrics

    Example:
        >>> # If all devices have similar avg latency: fairness ≈ 1.0
        >>> # If one device has 10x latency of others: fairness < 0.5
    """
    if not device_ids or len(result.waiting_times) == 0:
        return {
            "device_latency_fairness": 1.0,
            "device_throughput_fairness": 1.0,
        }

    # Group waiting times by device
    device_waiting_times: Dict[str, List[float]] = {}
    for idx, device_id in enumerate(device_ids):
        if device_id not in device_waiting_times:
            device_waiting_times[device_id] = []
        device_waiting_times[device_id].append(result.waiting_times[idx])

    # Per-device average latency
    avg_latencies = [float(np.mean(times)) for times in device_waiting_times.values()]

    # Per-device message count (throughput proxy)
    msg_counts = [float(len(times)) for times in device_waiting_times.values()]

    return {
        "device_latency_fairness": jain_fairness_index(avg_latencies),
        "device_throughput_fairness": jain_fairness_index(msg_counts),
    }


def compute_high_priority_overhead(
    result: SchedulerResult,
) -> Dict[str, float]:
    """
    Compute overhead metrics for high-priority messages.

    Overhead = how much delay high-priority traffic experiences due to
    resource reservation for lower bands.

    Metrics:
        - high_avg_latency: Average waiting time for HIGH band (band 1)
        - high_p95_latency: 95th percentile waiting time for HIGH band

    Args:
        result: Scheduler simulation result

    Returns:
        Dictionary with high-priority overhead metrics
    """
    # Band 1 = HIGH
    high_indices = [i for i, priority in enumerate(result.priorities) if priority == 1]

    if not high_indices:
        return {
            "high_avg_latency": 0.0,
            "high_p95_latency": 0.0,
        }

    high_waiting_times = [result.waiting_times[i] for i in high_indices]

    return {
        "high_avg_latency": float(np.mean(high_waiting_times)),
        "high_p95_latency": float(np.percentile(high_waiting_times, 95)),
    }


def compute_per_source_fairness(
    result: SchedulerResult,
    zone_priorities: List[int],
    is_alarm: List[bool],
) -> Dict[str, float]:
    """
    Compute fairness metrics across alarm sources (geographic zones).

    This measures whether alarms from different zones receive fair service,
    which is critical for preventing zone monopolization under malfunction/attack.
    Uses Jain fairness index on average latency per source.

    Args:
        result: Scheduler simulation result
        zone_priorities: Geographic zone for each message
        is_alarm: Alarm flags

    Returns:
        Dictionary with source fairness metrics:
        - alarm_source_fairness: Jain index (0-1, 1=perfect fairness)
        - alarm_source_count: Number of unique alarm sources
        - alarm_source_latency_cv: Coefficient of variation of source latencies
    """
    from collections import defaultdict

    # Get alarm indices
    alarm_indices = [i for i, is_a in enumerate(is_alarm) if is_a]

    # No alarms → perfect fairness by definition
    if len(alarm_indices) == 0:
        return {
            "alarm_source_fairness": 1.0,
            "alarm_source_count": 0,
            "alarm_source_latency_cv": 0.0,
        }

    # Group alarms by source (zone_priority)
    source_latencies = defaultdict(list)
    for idx in alarm_indices:
        zone = zone_priorities[idx]
        source_latencies[zone].append(result.waiting_times[idx])

    # Single source → perfect fairness
    if len(source_latencies) == 1:
        return {
            "alarm_source_fairness": 1.0,
            "alarm_source_count": 1,
            "alarm_source_latency_cv": 0.0,
        }

    # Compute average latency per source
    avg_latencies = [np.mean(lats) for lats in source_latencies.values()]

    # Jain fairness index: (sum x_i)^2 / (n * sum x_i^2)
    n = len(avg_latencies)
    sum_x = sum(avg_latencies)
    sum_x2 = sum(x**2 for x in avg_latencies)
    jain = (sum_x**2) / (n * sum_x2) if sum_x2 > 0 else 1.0

    # Coefficient of variation (CV = std/mean)
    mean_lat = np.mean(avg_latencies)
    cv = np.std(avg_latencies) / mean_lat if mean_lat > 0 else 0.0

    return {
        "alarm_source_fairness": float(jain),
        "alarm_source_count": int(n),
        "alarm_source_latency_cv": float(cv),
    }


@dataclass
class DistributionData:
    """
    Raw distribution vectors for per-device/per-source analysis.

    Preserves the underlying data that aggregate metrics (Jain index, CV)
    are computed from, enabling box/violin/CDF plots.

    Attributes:
        device_avg_latencies: Dict mapping device_id -> average latency
        device_msg_counts: Dict mapping device_id -> message count
        source_avg_latencies: Dict mapping zone_id -> average alarm latency (alarms only)
        per_band_device_latencies: Dict mapping band_id -> {device_id -> avg latency}
    """

    device_avg_latencies: Dict[str, float]
    device_msg_counts: Dict[str, int]
    source_avg_latencies: Dict[int, float]
    per_band_device_latencies: Dict[int, Dict[str, float]]

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "device_avg_latencies": self.device_avg_latencies,
            "device_msg_counts": self.device_msg_counts,
            "source_avg_latencies": {
                str(k): v for k, v in self.source_avg_latencies.items()
            },
            "per_band_device_latencies": {
                str(band): latencies
                for band, latencies in self.per_band_device_latencies.items()
            },
        }


def compute_distribution_data(
    result: SchedulerResult,
    device_ids: List[str],
    is_alarm: List[bool],
    zone_priorities: Optional[List[int]] = None,
) -> DistributionData:
    """
    Compute raw distribution vectors for plotting (box/violin/CDF).

    Unlike aggregate metrics (Jain index, CV), this preserves the underlying
    per-device and per-source data needed for distribution visualizations.

    Args:
        result: Scheduler simulation result
        device_ids: Device identifier for each message
        is_alarm: Alarm flags for each message
        zone_priorities: Geographic zone for each message (for per-source data)

    Returns:
        DistributionData with raw vectors for plotting
    """
    # Per-device latencies (global)
    device_waiting_times: Dict[str, List[float]] = {}
    for idx, device_id in enumerate(device_ids):
        if device_id not in device_waiting_times:
            device_waiting_times[device_id] = []
        device_waiting_times[device_id].append(result.waiting_times[idx])

    device_avg_latencies = {
        dev: float(np.mean(times)) for dev, times in device_waiting_times.items()
    }
    device_msg_counts = {dev: len(times) for dev, times in device_waiting_times.items()}

    # Per-source (zone) alarm latencies
    source_avg_latencies: Dict[int, float] = {}
    if zone_priorities is not None:
        source_waiting_times: Dict[int, List[float]] = {}
        for idx, is_a in enumerate(is_alarm):
            if is_a:
                zone = zone_priorities[idx]
                if zone not in source_waiting_times:
                    source_waiting_times[zone] = []
                source_waiting_times[zone].append(result.waiting_times[idx])

        source_avg_latencies = {
            zone: float(np.mean(times)) for zone, times in source_waiting_times.items()
        }

    # Per-band per-device latencies
    per_band_device_latencies: Dict[int, Dict[str, float]] = {}
    for band_id in range(4):
        band_indices = [
            i for i, priority in enumerate(result.priorities) if priority == band_id
        ]
        if not band_indices:
            per_band_device_latencies[band_id] = {}
            continue

        band_device_waits: Dict[str, List[float]] = {}
        for idx in band_indices:
            dev = device_ids[idx]
            if dev not in band_device_waits:
                band_device_waits[dev] = []
            band_device_waits[dev].append(result.waiting_times[idx])

        per_band_device_latencies[band_id] = {
            dev: float(np.mean(times)) for dev, times in band_device_waits.items()
        }

    return DistributionData(
        device_avg_latencies=device_avg_latencies,
        device_msg_counts=device_msg_counts,
        source_avg_latencies=source_avg_latencies,
        per_band_device_latencies=per_band_device_latencies,
    )


def compute_all_metrics(
    result: SchedulerResult,
    arrival_times: List[float],
    device_ids: List[str],
    is_alarm: List[bool],
    zone_priorities: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a scheduler result.

    Combines metrics from all categories:
    - Alarm latency
    - Bandwidth guarantees
    - Per-band fairness
    - Device fairness
    - High-priority overhead
    - Adaptive protection metrics (if zone_priorities provided)

    Args:
        result: Scheduler simulation result
        arrival_times: Message arrival times
        device_ids: Device identifiers
        is_alarm: Alarm flags
        zone_priorities: Geographic zone for each message (optional, for source fairness)

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Alarm metrics
    metrics.update(compute_alarm_metrics(result, arrival_times, is_alarm))

    # Bandwidth metrics
    metrics.update(compute_bandwidth_metrics(result, arrival_times, device_ids))

    # Per-band fairness
    metrics.update(compute_fairness_per_band(result, device_ids))

    # Global device fairness
    metrics.update(compute_device_fairness(result, device_ids))

    # Band-level waiting statistics (mean and p95)
    for band_id in range(4):
        band_indices = [
            i for i, priority in enumerate(result.priorities) if priority == band_id
        ]
        if not band_indices:
            metrics[f"band_{band_id}_wait_mean"] = 0.0
            metrics[f"band_{band_id}_wait_p95"] = 0.0
            continue
        waits = [result.waiting_times[i] for i in band_indices]
        metrics[f"band_{band_id}_wait_mean"] = float(np.mean(waits))
        metrics[f"band_{band_id}_wait_p95"] = float(np.percentile(waits, 95))

    # High-priority overhead
    metrics.update(compute_high_priority_overhead(result))

    # Overall statistics
    metrics["total_messages"] = result.n_jobs
    metrics["avg_waiting_time"] = float(np.mean(result.waiting_times))
    metrics["p95_waiting_time"] = float(np.percentile(result.waiting_times, 95))
    metrics["avg_e2e_time"] = float(np.mean(result.e2e_times))

    # Adaptive protection metadata extraction
    if result.metadata:
        metrics["alarm_dropped"] = float(result.metadata.get("alarm_dropped", 0))
        metrics["protection_enabled"] = float(
            result.metadata.get("alarm_protection_enabled", False)
        )

        # Compute alarm drop rate
        total_alarms = sum(is_alarm)
        if total_alarms > 0:
            metrics["alarm_dropped_rate"] = metrics["alarm_dropped"] / total_alarms
        else:
            metrics["alarm_dropped_rate"] = 0.0
    else:
        metrics["alarm_dropped"] = 0.0
        metrics["protection_enabled"] = 0.0
        metrics["alarm_dropped_rate"] = 0.0

    # Per-source fairness (if zone_priorities provided)
    if zone_priorities is not None:
        metrics.update(compute_per_source_fairness(result, zone_priorities, is_alarm))
    else:
        # Default values when zone_priorities not provided
        metrics["alarm_source_fairness"] = 1.0
        metrics["alarm_source_count"] = 0
        metrics["alarm_source_latency_cv"] = 0.0

    return metrics


# =============================================================================
# Statistical Analysis Infrastructure (Multi-Run Support)
# =============================================================================


@dataclass
class StatisticsSummary:
    """
    Statistical aggregation of metric values across multiple runs.

    Includes mean, standard deviation, 95% confidence interval,
    and coefficient of variation.

    Attributes:
        mean: Sample mean
        std: Sample standard deviation (ddof=1)
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
        n_samples: Number of samples

    Example:
        >>> values = [1.0, 1.2, 0.9, 1.1, 1.0]
        >>> stat = compute_statistics(values)
        >>> print(f"Mean: {stat.mean:.3f} ± {stat.std:.3f}")
        Mean: 1.040 ± 0.110
    """

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_samples: int

    def relative_std(self) -> float:
        """
        Compute coefficient of variation (CV%).

        CV% = (std / mean) * 100

        Returns:
            Coefficient of variation as percentage
        """
        if self.mean > 0:
            return (self.std / self.mean) * 100.0
        return 0.0

    def __str__(self) -> str:
        """
        Format as 'mean±std [ci_lower, ci_upper]'.

        Example: '1.234±0.056 [1.198, 1.270]'
        """
        return (
            f"{self.mean:.3f}±{self.std:.3f} "
            f"[{self.ci_lower:.3f}, {self.ci_upper:.3f}]"
        )


@dataclass
class ComparisonResult:
    """
    Statistical comparison between two schedulers on a single metric.

    Performs independent samples t-test to determine if the difference
    between schedulers is statistically significant.

    Attributes:
        scheduler_a: Name of first scheduler
        scheduler_b: Name of second scheduler
        metric_name: Name of metric being compared
        mean_a: Mean value for scheduler A
        mean_b: Mean value for scheduler B
        delta_pct: Relative difference (B-A)/A * 100
        t_statistic: t-test statistic
        p_value: Two-tailed p-value
        is_significant: True if p < 0.05

    Example:
        >>> comp = compare_schedulers("TRIAGE/4", "Strict", "alarm_latency", [0.1, 0.12], [0.4, 0.45])
        >>> print(f"{comp.delta_pct:+.1f}% change, p={comp.p_value:.4f} {comp.significance_marker()}")
        -72.0% change, p=0.0023 **
    """

    scheduler_a: str
    scheduler_b: str
    metric_name: str
    mean_a: float
    mean_b: float
    delta_pct: float
    t_statistic: float
    p_value: float
    is_significant: bool

    def significance_marker(self) -> str:
        """
        Return significance marker for p-value.

        Markers:
            *** : p < 0.001 (highly significant)
            **  : p < 0.01  (very significant)
            *   : p < 0.05  (significant)
            n.s.: p ≥ 0.05  (not significant)

        Returns:
            Significance marker string
        """
        if self.p_value < 0.001:
            return "***"
        elif self.p_value < 0.01:
            return "**"
        elif self.p_value < 0.05:
            return "*"
        return "n.s."


def compute_statistics(values: List[float]) -> StatisticsSummary:
    """
    Compute summary statistics with 95% confidence interval.

    Uses Student's t-distribution for CI calculation (appropriate for
    small sample sizes n < 30).

    Args:
        values: List of metric values from multiple runs

    Returns:
        StatisticsSummary with mean, std, CI bounds, and sample size

    Raises:
        ValueError: If values list is empty

    Example:
        >>> values = [0.124, 0.118, 0.131, 0.122, 0.127]
        >>> stat = compute_statistics(values)
        >>> print(stat)
        0.124±0.005 [0.117, 0.131]
    """
    if not values:
        raise ValueError("Cannot compute statistics for empty values list")

    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    # 95% confidence interval using t-distribution
    if n > 1 and std > 0:
        sem = stats.sem(values)  # Standard error of the mean
        ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    else:
        # Single sample or zero variance: CI collapses to mean
        ci_lower, ci_upper = mean, mean

    return StatisticsSummary(
        mean=mean, std=std, ci_lower=ci_lower, ci_upper=ci_upper, n_samples=n
    )


def compare_schedulers(
    scheduler_a_name: str,
    scheduler_b_name: str,
    metric_name: str,
    values_a: List[float],
    values_b: List[float],
) -> ComparisonResult:
    """
    Perform statistical comparison between two schedulers.

    Uses independent samples t-test (Welch's t-test, unequal variances assumed).

    Args:
        scheduler_a_name: Name of first scheduler (baseline)
        scheduler_b_name: Name of second scheduler (comparison)
        metric_name: Name of metric being compared
        values_a: Metric values from scheduler A
        values_b: Metric values from scheduler B

    Returns:
        ComparisonResult with mean difference, t-statistic, p-value, and significance

    Example:
        >>> seps_latencies = [0.12, 0.13, 0.11]
        >>> strict_latencies = [0.43, 0.42, 0.45]
        >>> comp = compare_schedulers("TRIAGE/4", "Strict", "alarm_latency",
        ...                           seps_latencies, strict_latencies)
        >>> print(f"TRIAGE/4 reduces latency by {-comp.delta_pct:.1f}% (p={comp.p_value:.4f})")
        TRIAGE/4 reduces latency by 72.0% (p=0.0012)
    """
    if not values_a or not values_b:
        raise ValueError("Cannot compare empty value lists")

    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    std_a = float(np.std(values_a, ddof=1)) if len(values_a) > 1 else 0.0
    std_b = float(np.std(values_b, ddof=1)) if len(values_b) > 1 else 0.0

    # Relative difference: (B-A)/A * 100
    if mean_a != 0:
        delta_pct = ((mean_b - mean_a) / mean_a) * 100.0
    else:
        delta_pct = 0.0

    # Coefficient of variation threshold for near-constant data detection
    # When CV < 1e-6, data is effectively constant and t-test is unreliable
    cv_threshold = 1e-6
    cv_a = std_a / abs(mean_a) if mean_a != 0 else (0.0 if std_a == 0 else float("inf"))
    cv_b = std_b / abs(mean_b) if mean_b != 0 else (0.0 if std_b == 0 else float("inf"))
    effectively_constant_a = std_a == 0.0 or cv_a < cv_threshold
    effectively_constant_b = std_b == 0.0 or cv_b < cv_threshold

    # Handle degenerate cases: zero/near-zero variance in one or both samples
    # Welch's t-test requires meaningful variance; skip test if data is constant
    if effectively_constant_a and effectively_constant_b:
        # Both constant: significant only if means differ
        t_stat = 0.0
        p_val = 0.0 if mean_a != mean_b else 1.0
    elif effectively_constant_a or effectively_constant_b:
        # One constant, one variable: use large t-stat placeholder when means differ
        t_stat = float("inf") if mean_a != mean_b else 0.0
        p_val = 0.0 if mean_a != mean_b else 1.0
    else:
        # Normal case: Welch's t-test (unequal variances, unequal sample sizes)
        t_stat, p_val = stats.ttest_ind(values_a, values_b, equal_var=False)

    is_sig = p_val < 0.05

    return ComparisonResult(
        scheduler_a=scheduler_a_name,
        scheduler_b=scheduler_b_name,
        metric_name=metric_name,
        mean_a=mean_a,
        mean_b=mean_b,
        delta_pct=delta_pct,
        t_statistic=float(t_stat),
        p_value=float(p_val),
        is_significant=is_sig,
    )


def format_comparison_table(
    comparisons: List[ComparisonResult], title: str = "SCHEDULER COMPARISON"
) -> str:
    """
    Format comparison results as ASCII table.

    Args:
        comparisons: List of comparison results
        title: Table title

    Returns:
        Formatted ASCII table string

    Example:
        >>> comp1 = ComparisonResult("TRIAGE/4", "Strict", "alarm_latency",
        ...                          0.12, 0.43, -72.0, -12.5, 0.001, True)
        >>> comp2 = ComparisonResult("TRIAGE/4", "Strict", "fairness",
        ...                          0.87, 0.42, 107.0, 14.2, 0.0001, True)
        >>> print(format_comparison_table([comp1, comp2]))
    """
    lines = []
    lines.append("=" * 100)
    lines.append(title)
    lines.append("=" * 100)
    lines.append(
        f"{'Metric':<25} {'A vs B':<20} {'Mean A':<12} {'Mean B':<12} "
        f"{'Δ%':<10} {'t-stat':<10} {'p-value':<12} {'Sig':<5}"
    )
    lines.append("-" * 100)

    for comp in comparisons:
        comparison_label = f"{comp.scheduler_a} vs {comp.scheduler_b}"
        p_value_str = f"{comp.p_value:.4f}" if comp.p_value >= 0.0001 else "<0.0001"

        lines.append(
            f"{comp.metric_name:<25} {comparison_label:<20} "
            f"{comp.mean_a:<12.4f} {comp.mean_b:<12.4f} "
            f"{comp.delta_pct:+<10.1f} {comp.t_statistic:<10.2f} "
            f"{p_value_str:<12} {comp.significance_marker():<5}"
        )

    return "\n".join(lines)

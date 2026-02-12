"""
Metrics and evaluation tools for TRIAGE/4.
"""

from .compute import (
    ComparisonResult,
    DistributionData,
    StatisticsSummary,
    compare_schedulers,
    compute_alarm_metrics,
    compute_all_metrics,
    compute_bandwidth_metrics,
    compute_device_fairness,
    compute_distribution_data,
    compute_fairness_per_band,
    compute_high_priority_overhead,
    compute_statistics,
    format_comparison_table,
    jain_fairness_index,
)
from .order_analysis import (
    compute_order_metrics,
    compute_service_order,
    export_input_order,
    export_output_order,
)
from .results import SchedulerResult
from .visualization import (
    plot_alarm_latency_cdf,
    plot_band_latency_with_error_bars,
    plot_band_priority_heatmap,
    plot_device_fairness_timeline,
    plot_position_jump_distribution,
    plot_rank_change_comparison,
)

__all__ = [
    # Core data structures
    "SchedulerResult",
    "DistributionData",
    # Time-based metrics
    "jain_fairness_index",
    "compute_alarm_metrics",
    "compute_bandwidth_metrics",
    "compute_device_fairness",
    "compute_distribution_data",
    "compute_fairness_per_band",
    "compute_high_priority_overhead",
    "compute_all_metrics",
    # Statistical analysis
    "StatisticsSummary",
    "ComparisonResult",
    "compute_statistics",
    "compare_schedulers",
    "format_comparison_table",
    # Order-based metrics
    "compute_service_order",
    "compute_order_metrics",
    "export_input_order",
    "export_output_order",
    # Visualization
    "plot_rank_change_comparison",
    "plot_position_jump_distribution",
    "plot_device_fairness_timeline",
    "plot_band_priority_heatmap",
    # Enhanced visualizations
    "plot_alarm_latency_cdf",
    "plot_band_latency_with_error_bars",
]

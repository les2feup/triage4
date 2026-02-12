"""
Visualization tools for order-based scheduler analysis.

Creates plots comparing message reordering behavior across schedulers.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .order_analysis import compute_service_order
from .results import SchedulerResult


def plot_rank_change_comparison(
    results_dict: Dict[str, Tuple[SchedulerResult, List[float], List[str]]],
    scenario_name: str,
    output_dir: str = "results"
) -> str:
    """
    Scatter plot comparing arrival rank vs service rank for all schedulers.

    Shows how each scheduler reorders messages:
    - Points on diagonal = FIFO (no reordering)
    - Points below diagonal = jumped forward (prioritized)
    - Points above diagonal = pushed back (deprioritized)

    Args:
        results_dict: {
            "TRIAGE/4": (result, arrival_times, device_ids),
            "Strict": (result, arrival_times, device_ids),
            "FIFO": (result, arrival_times, device_ids)
        }
        scenario_name: Scenario name for plot title and filename
        output_dir: Base directory for output

    Returns:
        Path to created plot file
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {
        "TRIAGE/4": "#2ca02c",  # Green
        "Strict": "#ff7f0e",  # Orange
        "FIFO": "#1f77b4",  # Blue
    }

    for scheduler_name, (result, arrival_times, _) in results_dict.items():
        service_order = compute_service_order(result, arrival_times)
        n = len(arrival_times)

        # arrival_rank[i] = i, service_rank[i] = where job i was served
        arrival_ranks = list(range(n))
        service_ranks = [service_order.index(i) for i in arrival_ranks]

        ax.scatter(
            arrival_ranks, service_ranks,
            alpha=0.5,
            s=20,
            label=scheduler_name,
            color=colors.get(scheduler_name, None)
        )

    # Diagonal line (FIFO baseline - no reordering)
    max_rank = max(len(arrival_times) for _, (_, arrival_times, _) in results_dict.items())
    ax.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.3, linewidth=2, label="No reordering (FIFO baseline)")

    ax.set_xlabel("Arrival Rank", fontsize=12)
    ax.set_ylabel("Service Rank", fontsize=12)
    ax.set_title(f"Message Reordering Comparison: {scenario_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.2)

    # Save plot
    scenario_dir = Path(output_dir) / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    plot_path = scenario_dir / "rank_change_comparison.png"

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(plot_path)


def plot_position_jump_distribution(
    results_dict: Dict[str, Tuple[SchedulerResult, List[float], List[bool]]],
    scenario_name: str,
    output_dir: str = "results"
) -> str:
    """
    Histogram of position jumps, separated by alarm vs non-alarm messages.

    Position jump = service_rank - arrival_rank
    - Negative = jumped forward (prioritized)
    - Positive = pushed back (deprioritized)

    Args:
        results_dict: {
            "TRIAGE/4": (result, arrival_times, is_alarm),
            ...
        }
        scenario_name: Scenario name for plot title and filename
        output_dir: Base directory for output

    Returns:
        Path to created plot file
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "TRIAGE/4": "#2ca02c",
        "Strict": "#ff7f0e",
        "FIFO": "#1f77b4",
    }

    for ax_idx, message_type in enumerate(["Alarm", "Non-Alarm"]):
        ax = axes[ax_idx]

        all_jumps = []
        labels = []

        for scheduler_name, (result, arrival_times, is_alarm) in results_dict.items():
            service_order = compute_service_order(result, arrival_times)
            n = len(arrival_times)

            # Compute position jumps
            service_rank_by_job = {job_idx: rank for rank, job_idx in enumerate(service_order)}
            position_jumps = [service_rank_by_job[i] - i for i in range(n)]

            # Filter by message type
            if message_type == "Alarm":
                filtered_jumps = [position_jumps[i] for i in range(n) if is_alarm[i]]
            else:
                filtered_jumps = [position_jumps[i] for i in range(n) if not is_alarm[i]]

            if filtered_jumps:
                all_jumps.append(filtered_jumps)
                labels.append(scheduler_name)

        # Plot histograms
        if all_jumps:
            bins = np.linspace(
                min(min(jumps) for jumps in all_jumps),
                max(max(jumps) for jumps in all_jumps),
                30
            )

            for jumps, label in zip(all_jumps, labels):
                ax.hist(
                    jumps, bins=bins, alpha=0.5,
                    label=label, color=colors.get(label, None)
                )

            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=1.5)
            ax.set_xlabel("Position Jump (service_rank - arrival_rank)", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"{message_type} Messages", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2, axis='y')

            # Add annotation
            ax.text(
                0.02, 0.98, "← Forward\n(Prioritized)",
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', alpha=0.6
            )
            ax.text(
                0.98, 0.98, "Backward →\n(Deprioritized)",
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right', alpha=0.6
            )
        else:
            ax.text(
                0.5, 0.5, f"No {message_type} messages",
                transform=ax.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12, alpha=0.5
            )
            ax.set_xlabel("Position Jump", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"{message_type} Messages", fontsize=12, fontweight="bold")

    fig.suptitle(f"Position Jump Distribution: {scenario_name}", fontsize=14, fontweight="bold")

    # Save plot
    scenario_dir = Path(output_dir) / scenario_name
    plot_path = scenario_dir / "position_jump_distribution.png"

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(plot_path)


def plot_device_fairness_timeline(
    results_dict: Dict[str, Tuple[SchedulerResult, List[float], List[str]]],
    scenario_name: str,
    output_dir: str = "results",
    max_messages: int = 100
) -> str:
    """
    Timeline showing which device was served at each position.

    Visualizes device interleaving patterns:
    - Perfect round-robin: devices cycle evenly
    - Monopolization: long runs of same device

    Args:
        results_dict: {
            "TRIAGE/4": (result, arrival_times, device_ids),
            ...
        }
        scenario_name: Scenario name for plot title and filename
        output_dir: Base directory for output
        max_messages: Maximum messages to plot (avoid overcrowding)

    Returns:
        Path to created plot file
    """
    n_schedulers = len(results_dict)
    fig, axes = plt.subplots(n_schedulers, 1, figsize=(14, 3 * n_schedulers), sharex=True)

    if n_schedulers == 1:
        axes = [axes]

    for ax_idx, (scheduler_name, (result, arrival_times, device_ids)) in enumerate(results_dict.items()):
        ax = axes[ax_idx]

        service_order = compute_service_order(result, arrival_times)

        # Limit to max_messages for readability
        plot_order = service_order[:max_messages]

        # Get unique devices and assign colors
        unique_devices = sorted(set(device_ids))
        device_to_idx = {dev: idx for idx, dev in enumerate(unique_devices)}
        n_devices = len(unique_devices)

        # Prepare data for timeline
        service_positions = list(range(len(plot_order)))
        device_indices = [device_to_idx[device_ids[job_idx]] for job_idx in plot_order]

        # Create timeline using scatter
        cmap = plt.cm.get_cmap('tab20' if n_devices <= 20 else 'viridis')
        colors_list = [cmap(i / n_devices) for i in range(n_devices)]

        for dev_name, dev_idx in device_to_idx.items():
            positions = [pos for pos, d_idx in zip(service_positions, device_indices) if d_idx == dev_idx]
            y_values = [dev_idx] * len(positions)
            ax.scatter(positions, y_values, label=dev_name, alpha=0.7, s=50, color=colors_list[dev_idx])

        ax.set_ylabel("Device", fontsize=10)
        ax.set_title(f"{scheduler_name} - Device Service Pattern", fontsize=11, fontweight="bold")
        ax.set_yticks(range(n_devices))
        ax.set_yticklabels(unique_devices, fontsize=8)
        ax.grid(True, alpha=0.2, axis='x')

        if n_devices <= 10:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8, ncol=1)

    axes[-1].set_xlabel("Service Position", fontsize=11)
    fig.suptitle(f"Device Fairness Timeline: {scenario_name}", fontsize=14, fontweight="bold")

    # Save plot
    scenario_dir = Path(output_dir) / scenario_name
    plot_path = scenario_dir / "device_fairness_timeline.png"

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(plot_path)


def plot_band_priority_heatmap(
    results_dict: Dict[str, Tuple[SchedulerResult, List[float]]],
    scenario_name: str,
    output_dir: str = "results",
    max_messages: int = 100
) -> str:
    """
    Heatmap showing which band was served at each service position.

    Visualizes band hierarchy compliance:
    - Proper hierarchy: ALARM → HIGH → STANDARD → BACKGROUND
    - Inversions: lower bands served before higher bands

    Args:
        results_dict: {
            "TRIAGE/4": (result, arrival_times),
            ...
        }
        scenario_name: Scenario name for plot title and filename
        output_dir: Base directory for output
        max_messages: Maximum messages to plot

    Returns:
        Path to created plot file
    """
    band_names = {0: "ALARM", 1: "HIGH", 2: "STANDARD", 3: "BACKGROUND"}
    band_colors = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c", 3: "#1f77b4"}  # Red, Orange, Green, Blue

    n_schedulers = len(results_dict)
    fig, axes = plt.subplots(n_schedulers, 1, figsize=(14, 2.5 * n_schedulers), sharex=True)

    if n_schedulers == 1:
        axes = [axes]

    for ax_idx, (scheduler_name, (result, arrival_times)) in enumerate(results_dict.items()):
        ax = axes[ax_idx]

        service_order = compute_service_order(result, arrival_times)

        # Limit to max_messages
        plot_order = service_order[:max_messages]

        if not result.priorities:
            ax.text(
                0.5, 0.5, "No band information available",
                transform=ax.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=11
            )
            ax.set_title(f"{scheduler_name} - Band Priority Timeline", fontsize=11, fontweight="bold")
            continue

        # Get bands for service order
        bands_served = [result.priorities[job_idx] for job_idx in plot_order]

        # Create timeline colored by band
        service_positions = list(range(len(plot_order)))

        for band_id, band_name in band_names.items():
            positions = [pos for pos, b in zip(service_positions, bands_served) if b == band_id]
            if positions:
                ax.scatter(
                    positions, [band_id] * len(positions),
                    label=band_name, alpha=0.7, s=80,
                    color=band_colors[band_id], marker='s'
                )

        ax.set_ylabel("Band", fontsize=10)
        ax.set_title(f"{scheduler_name} - Band Priority Timeline", fontsize=11, fontweight="bold")
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["ALARM", "HIGH", "STANDARD", "BACKGROUND"], fontsize=9)
        ax.set_ylim(-0.5, 3.5)
        ax.grid(True, alpha=0.2, axis='x')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)

    axes[-1].set_xlabel("Service Position", fontsize=11)
    fig.suptitle(f"Band Priority Order: {scenario_name}", fontsize=14, fontweight="bold")

    # Save plot
    scenario_dir = Path(output_dir) / scenario_name
    plot_path = scenario_dir / "band_priority_heatmap.png"

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(plot_path)


# =============================================================================
# Enhanced Visualizations for Statistical Benchmarks
# =============================================================================


def plot_alarm_latency_cdf(
    alarm_latencies_dict: Dict[str, List[float]],
    output_path: str,
    title: str = "Alarm Latency CDF",
) -> str:
    """
    Plot cumulative distribution function (CDF) of alarm latencies.

    Shows P(latency < t) for multiple schedulers, demonstrating how
    TRIAGE/4 shifts the distribution leftward (faster alarms).

    Args:
        alarm_latencies_dict: Dict[scheduler_name, List[alarm_latencies]]
        output_path: Path to save plot
        title: Plot title

    Returns:
        Path to saved plot

    Example:
        >>> alarm_latencies = {
        ...     "TRIAGE/4": [0.1, 0.12, 0.08, 0.15, 0.11],
        ...     "Strict": [0.4, 0.45, 0.38, 0.42, 0.41],
        ...     "FIFO": [1.2, 1.3, 1.1, 1.4, 1.2]
        ... }
        >>> plot_alarm_latency_cdf(alarm_latencies, "alarm_cdf.pdf")
    """
    colors = {"TRIAGE/4": "#C73E1D", "Strict": "#2E86AB", "FIFO": "#6C757D"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for scheduler_name, latencies in alarm_latencies_dict.items():
        if not latencies:
            continue

        # Sort latencies
        sorted_latencies = np.sort(latencies)
        n = len(sorted_latencies)

        # Compute CDF: P(X <= x)
        cdf = np.arange(1, n + 1) / n

        # Plot
        ax.plot(
            sorted_latencies,
            cdf,
            linewidth=2.5,
            label=scheduler_name,
            color=colors.get(scheduler_name, "black"),
        )

        # Add median marker
        median = np.median(latencies)
        median_cdf = 0.5
        ax.plot(
            median,
            median_cdf,
            marker="o",
            markersize=10,
            color=colors.get(scheduler_name, "black"),
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

    ax.set_xlabel("Alarm Latency (s)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Add reference lines
    for percentile, alpha in [(0.5, 0.3), (0.95, 0.3)]:
        ax.axhline(percentile, color="gray", linestyle="--", alpha=alpha, linewidth=1)
        ax.text(
            ax.get_xlim()[1] * 0.98,
            percentile,
            f"P{int(percentile*100)}",
            ha="right",
            va="bottom",
            fontsize=9,
            color="gray",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def plot_band_latency_with_error_bars(
    aggregated_metrics: Dict,
    output_path: str,
    title: str = "Band-Level Waiting Times",
) -> str:
    """
    Plot band-level latencies with error bars for statistical benchmarks.

    Bar chart showing mean ± std for ALARM, HIGH, STANDARD, BACKGROUND bands
    across multiple schedulers.

    Args:
        aggregated_metrics: Dict[scheduler_name, Dict[metric_name, StatisticsSummary]]
        output_path: Path to save plot
        title: Plot title

    Returns:
        Path to saved plot

    Example:
        >>> from assessment.metrics import StatisticsSummary
        >>> aggregated = {
        ...     "TRIAGE/4": {
        ...         "band_0_avg": StatisticsSummary(0.02, 0.005, 0.015, 0.025, 30),
        ...         "band_1_avg": StatisticsSummary(0.11, 0.02, 0.09, 0.13, 30),
        ...     },
        ...     "Strict": {
        ...         "band_0_avg": StatisticsSummary(0.04, 0.01, 0.03, 0.05, 30),
        ...         "band_1_avg": StatisticsSummary(0.09, 0.015, 0.075, 0.105, 30),
        ...     }
        ... }
        >>> plot_band_latency_with_error_bars(aggregated, "band_latency.pdf")
    """
    from .compute import StatisticsSummary

    colors = {"TRIAGE/4": "#C73E1D", "Strict": "#2E86AB", "FIFO": "#6C757D"}
    band_names = ["ALARM", "HIGH", "STANDARD", "BACKGROUND"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(band_names))
    width = 0.25
    schedulers = list(aggregated_metrics.keys())

    for i, scheduler_name in enumerate(schedulers):
        metrics = aggregated_metrics[scheduler_name]

        # Extract band average waiting times
        means = []
        stds = []

        for band_id in range(4):
            # Try different metric keys
            metric_key = None
            for key_candidate in [
                f"band_{band_id}_avg_wait",
                f"band_{band_id}_latency",
                f"alarm_avg_latency" if band_id == 0 else None,
                f"high_avg_latency" if band_id == 1 else None,
            ]:
                if key_candidate and key_candidate in metrics:
                    metric_key = key_candidate
                    break

            if metric_key and isinstance(metrics[metric_key], StatisticsSummary):
                stat = metrics[metric_key]
                means.append(stat.mean)
                stds.append(stat.std)
            else:
                # Fallback: use 0
                means.append(0.0)
                stds.append(0.0)

        # Plot bars
        offset = (i - len(schedulers) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=scheduler_name,
            color=colors.get(scheduler_name, "gray"),
            edgecolor="black",
            capsize=5,
            alpha=0.8,
        )

    ax.set_xlabel("Band", fontsize=12)
    ax.set_ylabel("Average Wait Time (s)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path

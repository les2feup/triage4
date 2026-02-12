"""
Order-based analysis for scheduler evaluation.

Computes platform-independent metrics by comparing arrival order vs service order.
Complements time-based metrics with reordering analysis.
"""

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .results import SchedulerResult


def compute_service_order(
    result: SchedulerResult,
    arrival_times: List[float]
) -> List[int]:
    """
    Reconstruct service order from scheduler result.

    Job indices are preserved from arrival order in SchedulerResult.
    Service order is reconstructed by sorting jobs by service start time.

    Args:
        result: Scheduler simulation result
        arrival_times: Message arrival times (indexed by job)

    Returns:
        List of job indices in service order
        - service_order[0] = job index served first
        - service_order[1] = job index served second
        - etc.

    Example:
        >>> # If job 5 was served first, job 0 second, job 1 third:
        >>> service_order = [5, 0, 1, ...]
    """
    n = len(arrival_times)

    # Compute service start time for each job
    service_start_times = [
        arrival_times[i] + result.waiting_times[i]
        for i in range(n)
    ]

    # Sort jobs by service start time
    service_order = sorted(range(n), key=lambda i: service_start_times[i])

    return service_order


def compute_consecutive_serves(
    service_order: List[int],
    device_ids: List[str]
) -> List[int]:
    """
    Count consecutive serves from each device.

    Measures device monopolization vs fairness.
    Perfect round-robin = all counts = 1

    Args:
        service_order: Job indices in service order
        device_ids: Device ID for each job (indexed by job)

    Returns:
        List of consecutive serve counts per device occurrence
        - [1, 1, 1, ...] = perfect interleaving
        - [5, 1, 10, ...] = some monopolization

    Example:
        >>> # Service order: [A, A, A, B, C, C]
        >>> # Returns: [3, 1, 2]
        >>> # Device A served 3 times consecutively, then B once, then C twice
    """
    if not service_order:
        return []

    consecutive_counts = []
    current_device = device_ids[service_order[0]]
    current_count = 1

    for i in range(1, len(service_order)):
        job_idx = service_order[i]
        device = device_ids[job_idx]

        if device == current_device:
            current_count += 1
        else:
            consecutive_counts.append(current_count)
            current_device = device
            current_count = 1

    # Don't forget last run
    consecutive_counts.append(current_count)

    return consecutive_counts


def count_band_inversions(
    service_order: List[int],
    band_assignments: Optional[List[int]]
) -> int:
    """
    Count band priority inversions.

    An inversion occurs when a lower-priority band is served before
    a higher-priority band that is already queued.

    Args:
        service_order: Job indices in service order
        band_assignments: Band for each job (0=ALARM, 1=HIGH, 2=STANDARD, 3=BACKGROUND)

    Returns:
        Number of inversions (0 = perfect priority ordering)

    Example:
        >>> # ALARM < HIGH < STANDARD < BACKGROUND (lower number = higher priority)
        >>> # Service: [HIGH, STANDARD, HIGH] → 1 inversion (STANDARD before 2nd HIGH)
    """
    if not band_assignments or len(service_order) <= 1:
        return 0

    inversions = 0

    for i in range(len(service_order)):
        job_i = service_order[i]
        band_i = band_assignments[job_i]

        # Check all jobs served after this one
        for j in range(i + 1, len(service_order)):
            job_j = service_order[j]
            band_j = band_assignments[job_j]

            # If job_i has lower priority (higher band number) but was served first
            if band_i > band_j:
                inversions += 1

    return inversions


def compute_order_metrics(
    result: SchedulerResult,
    arrival_times: List[float],
    device_ids: List[str],
    is_alarm: List[bool]
) -> Dict[str, float]:
    """
    Compute order-based metrics from scheduler result.

    Metrics quantify reordering behavior independent of hardware timing:
    - Position jumps: How far messages moved from arrival to service order
    - Alarm prioritization: How much alarms jumped forward
    - Device fairness: Consecutive serve patterns
    - Band ordering: Priority inversion counts

    Args:
        result: Scheduler simulation result
        arrival_times: Message arrival times
        device_ids: Device identifiers
        is_alarm: Alarm flags

    Returns:
        Dictionary of order-based metrics
    """
    n = len(arrival_times)
    service_order = compute_service_order(result, arrival_times)

    # Create position jump mapping
    # position_jumps[i] = how much job i moved
    # Negative = jumped forward (higher priority)
    # Positive = pushed back (lower priority)
    service_rank_by_job = {job_idx: rank for rank, job_idx in enumerate(service_order)}
    position_jumps = [
        service_rank_by_job[i] - i  # service_rank - arrival_rank
        for i in range(n)
    ]

    # Alarm metrics
    alarm_indices = [i for i, alarm in enumerate(is_alarm) if alarm]
    alarm_jumps = [position_jumps[i] for i in alarm_indices]

    # Device monopolization metrics
    consecutive_counts = compute_consecutive_serves(service_order, device_ids)

    # Band inversion metrics
    inversions = count_band_inversions(service_order, result.priorities)

    # Compute metrics
    metrics = {
        # Position jump statistics
        "position_jump_mean": float(np.mean(position_jumps)),
        "position_jump_std": float(np.std(position_jumps)),
        "position_jump_min": float(np.min(position_jumps)),  # Most forward jump
        "position_jump_max": float(np.max(position_jumps)),  # Most backward push

        # Alarm prioritization
        "alarm_count": len(alarm_indices),
        "alarm_avg_jump": float(np.mean(alarm_jumps)) if alarm_jumps else 0.0,
        "alarm_max_forward_jump": float(np.min(alarm_jumps)) if alarm_jumps else 0.0,  # Most negative = most forward

        # Device fairness
        "max_consecutive_serves": max(consecutive_counts) if consecutive_counts else 0,
        "avg_consecutive_serves": float(np.mean(consecutive_counts)) if consecutive_counts else 0.0,

        # Band ordering
        "band_inversion_count": inversions,
    }

    return metrics


def export_input_order(
    arrival_times: List[float],
    device_ids: List[str],
    zone_priorities: List[int],
    is_alarm: List[bool],
    scenario_name: str,
    output_dir: str = "results"
) -> str:
    """
    Export arrival order to CSV.

    Creates: {output_dir}/{scenario_name}/input.csv

    CSV format:
        arrival_rank,arrival_time,device_id,zone_priority,is_alarm,message_id
        0,0.0,EDU_0,0,False,msg_0
        1,0.01,EDU_1,0,False,msg_1
        ...

    Args:
        arrival_times: Message arrival times
        device_ids: Device identifiers
        zone_priorities: Geographic zone priorities
        is_alarm: Alarm flags
        scenario_name: Scenario name for directory
        output_dir: Base output directory

    Returns:
        Path to created CSV file
    """
    # Create directory structure
    scenario_dir = Path(output_dir) / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Build CSV rows
    rows = []
    for i in range(len(arrival_times)):
        rows.append({
            "arrival_rank": i,
            "arrival_time": f"{arrival_times[i]:.6f}",
            "device_id": device_ids[i],
            "zone_priority": zone_priorities[i],
            "is_alarm": is_alarm[i],
            "message_id": f"msg_{i}",
        })

    # Write CSV
    csv_path = scenario_dir / "input.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["arrival_rank", "arrival_time", "device_id", "zone_priority", "is_alarm", "message_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(csv_path)


def export_output_order(
    result: SchedulerResult,
    arrival_times: List[float],
    device_ids: List[str],
    zone_priorities: List[int],
    is_alarm: List[bool],
    scheduler_name: str,
    scenario_name: str,
    output_dir: str = "results"
) -> str:
    """
    Export service order to CSV.

    Creates: {output_dir}/{scenario_name}/{scheduler_name}_output.csv

    CSV format:
        service_rank,service_time,device_id,zone_priority,is_alarm,message_id,arrival_rank,position_jump,band
        0,0.0,EDU_0,0,False,msg_0,0,0,1
        1,0.05,sensor_42,5,True,msg_alarm,500,-499,0
        ...

    Args:
        result: Scheduler simulation result
        arrival_times: Message arrival times
        device_ids: Device identifiers
        zone_priorities: Geographic zone priorities
        is_alarm: Alarm flags
        scheduler_name: Scheduler name for filename
        scenario_name: Scenario name for directory
        output_dir: Base output directory

    Returns:
        Path to created CSV file
    """
    service_order = compute_service_order(result, arrival_times)

    # Build CSV rows
    rows = []
    for service_rank, job_idx in enumerate(service_order):
        position_jump = service_rank - job_idx
        service_time = arrival_times[job_idx] + result.waiting_times[job_idx]

        rows.append({
            "service_rank": service_rank,
            "service_time": f"{service_time:.6f}",
            "device_id": device_ids[job_idx],
            "zone_priority": zone_priorities[job_idx],
            "is_alarm": is_alarm[job_idx],
            "message_id": f"msg_{job_idx}",
            "arrival_rank": job_idx,
            "position_jump": position_jump,
            "band": result.priorities[job_idx] if result.priorities else -1,
        })

    # Write CSV
    scenario_dir = Path(output_dir) / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    csv_path = scenario_dir / f"{scheduler_name.lower()}_output.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "service_rank", "service_time", "device_id", "zone_priority",
            "is_alarm", "message_id", "arrival_rank", "position_jump", "band"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(csv_path)

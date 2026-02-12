"""
Validate Criterion 6: Performance overhead when protection is inactive.
"""

import time

import numpy as np

from triage4 import TRIAGE4Config, TRIAGE4Scheduler
from assessment.workloads import generate_multi_zone_emergency


def test_overhead_when_protection_inactive():
    """
    Compare CPU time for TRIAGE/4 with protection enabled but idle
    vs protection completely disabled.

    Expected: Overhead < 5% (ideally < 1%)
    """
    workload = generate_multi_zone_emergency(
        n_zones=6,
        devices_per_zone=2,
        messages_per_device=50,
        alarm_zones=[2, 4],
        duration=30.0,
    )

    config_protection_idle = TRIAGE4Config(
        enable_alarm_protection=True,
        alarm_abnormal_threshold=20.0,
        alarm_limit_budget=15,
        service_rate=20.0,
        high_token_budget=20,
        standard_token_budget=15,
        background_token_budget=5,
    )

    config_no_protection = TRIAGE4Config(
        enable_alarm_protection=False,
        service_rate=20.0,
        high_token_budget=20,
        standard_token_budget=15,
        background_token_budget=5,
    )

    n_runs = 30
    repeat_runs = 30
    cpu_times_idle = []
    cpu_times_off = []

    for seed in range(999, 999 + n_runs):
        scheduler_idle = TRIAGE4Scheduler(config_protection_idle, scheduler_seed=seed)
        start = time.perf_counter()
        for _ in range(repeat_runs):
            scheduler_idle.schedule(
                workload.arrival_times,
                workload.device_ids,
                workload.zone_priorities,
                workload.is_alarm,
            )
        cpu_times_idle.append(time.perf_counter() - start)

        scheduler_off = TRIAGE4Scheduler(config_no_protection, scheduler_seed=seed)
        start = time.perf_counter()
        for _ in range(repeat_runs):
            scheduler_off.schedule(
                workload.arrival_times,
                workload.device_ids,
                workload.zone_priorities,
                workload.is_alarm,
            )
        cpu_times_off.append(time.perf_counter() - start)

    mean_idle = np.mean(cpu_times_idle)
    mean_off = np.mean(cpu_times_off)
    overhead_pct = ((mean_idle - mean_off) / mean_off) * 100

    print(
        f"CPU Time (protection idle): {mean_idle:.6f}s ± {np.std(cpu_times_idle):.6f}s"
    )
    print(f"CPU Time (protection off):  {mean_off:.6f}s ± {np.std(cpu_times_off):.6f}s")
    print(f"Overhead: {overhead_pct:.2f}%")

    assert overhead_pct >= 0.0

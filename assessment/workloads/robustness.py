"""
Configured robustness scenarios (R1-R3) as evaluated in the paper.

``scenarios.py`` provides the generators; this module pins the arguments those
generators are called with. The statistical benchmark and the AAP sensitivity
sweep both consume it, so they provably characterize the same workloads.

This module exists because they previously did not. Each kept its own call
sites, and the sweep silently drifted onto a different malfunction surge than
the results tables reported. A sweep that justifies a threshold has to sweep the
scenario the tables describe, or the paper contradicts itself.

Parameter notes:
  - Legitimate alarms are present in R1 and R2 so that shedding can be attributed
    by source class. Without them a drop rate cannot distinguish "contained the
    attack" from "discarded the emergency".
  - Jitter decorrelates runs. Both generators are otherwise deterministic, which
    left the n-run protocol varying only the scheduler seed. Each value is one
    inter-arrival of its scenario's densest stream.
"""

from typing import Optional

from .scenarios import (
    Workload,
    generate_alarm_flood_attack,
    generate_alarm_malfunction_surge,
    generate_legit_extreme_emergency,
)


def build_alarm_flood_attack(seed: Optional[int] = None) -> Workload:
    """R1: a single attacker floods at 20 alarms/s while real emergencies arrive."""
    return generate_alarm_flood_attack(
        legitimate_alarms=20,
        jitter_std=0.05,  # one attacker inter-arrival at 20/s
        seed=seed,
    )


def build_alarm_malfunction_surge(seed: Optional[int] = None) -> Workload:
    """R2: eight sensors malfunction at once while real emergencies arrive.

    Rates are chosen per-device rather than to hit an aggregate target: the heavy
    device at 5/s and the light devices at 2/s both sit clearly above the 1.0/s
    per-source limit, keeping this a *multiple*-malfunction scenario distinct
    from skewed_alarm_sources. Arrivals are already drawn at random, so no jitter
    is needed.
    """
    return generate_alarm_malfunction_surge(
        heavy_zone=0,
        light_zones=[1, 2, 3, 4, 5, 6, 7],
        heavy_rate=5.0,
        light_rate=2.0,
        duration=20.0,
        legitimate_alarms=20,
        seed=seed,
    )


def build_legit_extreme_emergency(seed: Optional[int] = None) -> Workload:
    """R3: every source is a genuine emergency. The control: nothing may be shed."""
    return generate_legit_extreme_emergency(
        jitter_std=0.1,  # one telemetry inter-arrival at ~10/s
        seed=seed,
    )


# (builder, display name, service_rate_override) — the shape both the statistical
# benchmark and the sensitivity sweep expect in their scenario registries.
ROBUSTNESS_SCENARIOS = {
    "alarm_flood_attack": (
        build_alarm_flood_attack,
        "Alarm Flood Attack",
        20.0,
    ),
    "alarm_malfunction_surge": (
        build_alarm_malfunction_surge,
        "Alarm Malfunction Surge",
        None,
    ),
    "legit_extreme_emergency": (
        build_legit_extreme_emergency,
        "Legitimate Extreme Emergency",
        None,
    ),
}

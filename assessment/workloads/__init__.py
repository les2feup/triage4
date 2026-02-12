"""
Workload generators for TRIAGE/4 evaluation.

Provides scenarios for testing TRIAGE/4 performance characteristics:
- Alarm Under Burst Load: Alarms preempting high-rate telemetry
- Device Monopolization: Fair device scheduling verification
- Multi-Zone Emergency: Complex multi-band interaction
"""

from .scenarios import (
    Workload,
    generate_alarm_load_regime,
    generate_alarm_load_near_saturation_constrained,
    generate_alarm_rate_sweep,
    generate_alarm_under_burst,
    generate_alarm_under_burst_phased,
    generate_alarm_flood_attack,
    generate_alarm_malfunction_surge,
    generate_device_monopolization,
    generate_device_monopolization_sweep,
    generate_legit_extreme_emergency,
    generate_multi_zone_emergency,
    generate_multi_zone_emergency_cascade,
    generate_skewed_alarm_sources,
)

__all__ = [
    "Workload",
    # Original scenarios
    "generate_alarm_under_burst",
    "generate_device_monopolization",
    "generate_multi_zone_emergency",
    # Phase-based variants (enhanced)
    "generate_alarm_under_burst_phased",
    "generate_device_monopolization_sweep",
    "generate_multi_zone_emergency_cascade",
    # Alarm protection scenarios
    "generate_alarm_flood_attack",
    "generate_alarm_malfunction_surge",
    "generate_legit_extreme_emergency",
    # Adaptive protection experimental design helpers
    "generate_alarm_rate_sweep",
    "generate_skewed_alarm_sources",
    "generate_alarm_load_regime",
    "generate_alarm_load_near_saturation_constrained",
]

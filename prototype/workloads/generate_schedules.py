"""
Pre-generate C3/R3 arrival schedules — STUB (implemented in Part E).

Runs in the MAIN repo .venv (which has ``assessment.workloads``), calls the
canonical generators with a fixed seed, and dumps each schedule as
``[{t, device_id, zone_priority, is_alarm}, ...]`` JSON committed alongside this
file. The prototype venv never imports ``assessment`` — it only replays the JSON
— so the hardware workload is byte-identical to the simulation.

Planned outputs (plan §7):
    c3_multi_zone_emergency.json    from generate_multi_zone_emergency
        (6 zones, 2 devices/zone, ~90% load, simultaneous emergencies zones 2&4;
         12 logical devices)
    r3_legit_extreme_emergency.json from generate_legit_extreme_emergency
        (10 zones, legitimate alarms below the AAP abnormal threshold + background)

The seed is recorded inside each JSON. Saturation is controlled at replay time
by the broker egress rate C, not by the schedule.
"""

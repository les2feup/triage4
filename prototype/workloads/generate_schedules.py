"""
Pre-generate the C3, HW-Flood, and R3 arrival schedules as committed JSON.

The testbed has six wireless devices, one per zone, each on its own radio. That
is a faithful setting for the geographic scenarios (C3, R3), but not for the
large-scale AAP scenarios: the simulated flood and malfunction spread their
sources across two dozen zones, and emulating that many independent radios on
six devices would misrepresent exactly the channel behaviour a network reviewer
checks. So the hardware AAP test is built to the testbed instead of squeezed
onto it. HW-Flood uses six real devices: one attacker floods the ALARM band,
five legitimate sources emit genuine alarms below the per-source threshold, and
every source is one device on one radio. The large-scale flood and malfunction
stay in simulation, where source count is free and faithful.

The three hardware scenarios then read as: C3 stages the geographic inversion,
HW-Flood shows adaptive protection shedding the source that causes the overload,
and R3 is the control where every source is genuine and nothing may be shed.

Run this in the MAIN repository .venv (which provides ``assessment.workloads``);
the prototype venv never imports ``assessment``. The schedules are dumped to
JSON so the broker replays a fixed workload rather than regenerating one.

Each message carries a scenario-prefixed, globally-unique ``msg_id`` so the
broker-side overhead CSV and the client-side RTT CSV join cleanly on it.

Usage (from repo root):
    .venv/bin/python -m prototype.workloads.generate_schedules
or (from prototype/, with the main venv on PYTHONPATH):
    ../.venv/bin/python workloads/generate_schedules.py
"""

import json
import os

import numpy as np

from assessment.workloads import build_legit_extreme_emergency
from assessment.workloads.scenarios import Workload, generate_multi_zone_emergency

# C3 and R3 come from generators/builders. C3 is deterministic at jitter_std=0,
# so its seed is only provenance (D14); R3's builder carries jitter, so its seed
# selects which sample is frozen. HW-Flood is built here, deterministically, and
# takes no seed. Its deterministic operating point is not a knife edge: the
# band-global ablation sheds ~53% of legitimate alarms there and stays within a
# few points under the testbed's ~0.2 ms replay jitter, so the committed
# schedule reproduces on hardware without depending on a lucky draw.
SEED = 999
HERE = os.path.dirname(os.path.abspath(__file__))


def build_hw_flood_attack() -> Workload:
    """Six-device alarm flood for the testbed: one attacker, five legitimate sources.

    Zone 5 hosts the attacker, flooding the ALARM band at 25 alarms/s — far above
    the 0.5/s per-source threshold, so the per-source layer must shed it. Zones
    0-4 each host one legitimate device emitting genuine alarms at 0.3/s, below
    the threshold, plus routine telemetry at 1/s. The flood pushes the aggregate
    ALARM rate past the band-global budget, so a scheduler reduced to that
    backstop sheds indiscriminately and catches the legitimate alarms; the
    per-source layer does not. One device per zone, so no radio carries more than
    one logical source.
    """
    duration, attacker_rate, legit_alarm_rate, routine_rate = 20.0, 25.0, 0.3, 1.0
    times, devices, zones, is_alarm = [], [], [], []

    for i in range(int(duration * attacker_rate)):
        times.append((i + 0.5) / attacker_rate)
        devices.append("attacker")
        zones.append(5)
        is_alarm.append(True)

    for z in range(5):
        for i in range(int(duration * legit_alarm_rate)):
            times.append((i + 0.3) / legit_alarm_rate)
            devices.append(f"legit_z{z}")
            zones.append(z)
            is_alarm.append(True)
        for i in range(int(duration * routine_rate)):
            times.append((i + 0.1) / routine_rate)
            devices.append(f"legit_z{z}")
            zones.append(z)
            is_alarm.append(False)

    order = np.argsort(times, kind="stable")
    # source_is_legitimate marks the attacker so drop attribution can separate
    # shed attacker traffic from shed legitimate alarms.
    return Workload(
        arrival_times=[times[i] for i in order],
        device_ids=[devices[i] for i in order],
        zone_priorities=[zones[i] for i in order],
        is_alarm=[bool(is_alarm[i]) for i in order],
        source_is_legitimate=[devices[i] != "attacker" for i in order],
        description="Six-device alarm flood: one attacker, five legitimate sources",
    )


def _dump(workload, scenario: str, generator: str, prefix: str, path: str) -> None:
    """Serialise a Workload to the prototype's replay JSON schema."""
    messages = [
        {
            "msg_id": f"{prefix}-{i:04d}",
            "t": float(workload.arrival_times[i]),
            "device_id": workload.device_ids[i],
            "zone_priority": int(workload.zone_priorities[i]),
            "is_alarm": bool(workload.is_alarm[i]),
        }
        for i in range(workload.n_messages)
    ]
    document = {
        "scenario": scenario,
        "generator": generator,
        "seed": SEED,
        "description": workload.description,
        "n_messages": workload.n_messages,
        "n_alarms": workload.n_alarms,
        "duration": round(workload.duration, 6),
        "messages": messages,
    }
    with open(path, "w") as handle:
        json.dump(document, handle, indent=2)
    print(f"{scenario}: {workload.n_messages} msgs ({workload.n_alarms} alarms), "
          f"duration {workload.duration:.2f}s -> {os.path.relpath(path)}")


def main() -> None:
    # C3 — multizone emergency: 6 zones, 2 devices/zone, emergencies in zones 2 & 4.
    c3 = generate_multi_zone_emergency(seed=SEED)
    _dump(c3, "c3_multi_zone_emergency", "generate_multi_zone_emergency", "c3",
          os.path.join(HERE, "c3_multi_zone_emergency.json"))

    # HW-Flood — six-device alarm flood: the hardware AAP shedding test, one
    # attacker and five legitimate sources, each on its own radio.
    hw = build_hw_flood_attack()
    _dump(hw, "hw_flood_attack", "build_hw_flood_attack", "hw",
          os.path.join(HERE, "hw_flood_attack.json"))

    # R3 — legitimate extreme emergency: many zones of genuine alarms below the
    # AAP threshold + background telemetry. The control: adaptive protection must
    # stay inactive and shed nothing, per-source layer and backstop alike.
    r3 = build_legit_extreme_emergency(SEED)
    _dump(r3, "r3_legit_extreme_emergency", "build_legit_extreme_emergency", "r3",
          os.path.join(HERE, "r3_legit_extreme_emergency.json"))


if __name__ == "__main__":
    main()

"""
Pre-generate the C3, R1, R2, and R3 arrival schedules as committed JSON.

R1, R2, and R3 are the complete AAP family: an attack, a malfunction, and a
legitimate mass emergency. C3 and R3 alone never drive either AAP layer above
its threshold, so they cannot show whether shedding lands on the source that
caused the overload. R1 and R2 do, which is what makes them worth the hardware.

Run this in the MAIN repository .venv (which provides ``assessment.workloads``);
the prototype venv never imports ``assessment``. The schedules are dumped to
JSON so the hardware workload is byte-identical to the simulation — the broker
replays the JSON rather than regenerating, removing any divergence.

Each message carries a scenario-prefixed, globally-unique ``msg_id`` so the
broker-side overhead CSV and the client-side RTT CSV join cleanly on it.

Usage (from repo root):
    .venv/bin/python -m prototype.workloads.generate_schedules
or (from prototype/, with the main venv on PYTHONPATH):
    ../.venv/bin/python workloads/generate_schedules.py
"""

import json
import os

from assessment.workloads import (
    build_alarm_flood_attack,
    build_alarm_malfunction_surge,
)
from assessment.workloads.scenarios import (
    generate_legit_extreme_emergency,
    generate_multi_zone_emergency,
)

# C3 and R3 are deterministic at jitter_std=0, so for them the seed is only
# provenance (D14). R1 and R2 are not: the canonical builders carry jitter and
# random arrival draws, so the seed genuinely selects which sample gets frozen.
# Replay determinism does not depend on this either way — the committed JSON is
# what the broker replays, and it is fixed the moment it is written.
SEED = 999
HERE = os.path.dirname(os.path.abspath(__file__))


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

    # R1 — alarm flood attack: one attacker at 20 alarms/s while 20 real
    # emergencies arrive from other sources. The per-source layer must shed the
    # attacker and spare the 20; a band-global backstop cannot tell them apart.
    r1 = build_alarm_flood_attack(SEED)
    _dump(r1, "r1_alarm_flood_attack", "build_alarm_flood_attack", "r1",
          os.path.join(HERE, "r1_alarm_flood_attack.json"))

    # R2 — alarm malfunction surge: eight sensors malfunction at once, again
    # alongside 20 real emergencies. Tests containment when the abnormal traffic
    # is spread across several sources instead of concentrated in one.
    r2 = build_alarm_malfunction_surge(SEED)
    _dump(r2, "r2_alarm_malfunction_surge", "build_alarm_malfunction_surge", "r2",
          os.path.join(HERE, "r2_alarm_malfunction_surge.json"))

    # R3 — legitimate extreme emergency: 10 zones of legitimate alarms below the
    # AAP abnormal threshold + background telemetry. Confirms no false-positive
    # shedding under contention with AAP on.
    r3 = generate_legit_extreme_emergency(seed=SEED)
    _dump(r3, "r3_legit_extreme_emergency", "generate_legit_extreme_emergency", "r3",
          os.path.join(HERE, "r3_legit_extreme_emergency.json"))


if __name__ == "__main__":
    main()

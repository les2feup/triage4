"""
Count priority inversions in the delivery order an external subscriber observed.

A priority inversion is a routine message that ARRIVED AFTER an alarm but was
DELIVERED BEFORE it — the failure TRIAGE/4 exists to prevent. It is the most
direct statement of R1.1 available, and unlike the RTT tables it does not depend
on trusting the broker: the order comes from ``clients/observer.py``, an
unmodified MQTT subscriber, and the arrival times come from the committed
schedule. The broker is not asked to vouch for itself.

Reported per scheduler as inversions per rep (mean ±95% CI), alongside the worst
alarm's overtake count — the single alarm that the most routine traffic jumped.

Usage (from prototype/):
    python inversions.py --results-dir results --scenario r3_legit_extreme_emergency
"""

import argparse
import bisect
import csv
import glob
import json
import os
import re
from typing import Dict, List

import numpy as np

SCHEDULERS = ("fifo", "strict", "wfq", "triage4")


def _schedule(scenario: str) -> tuple:
    """Arrival time per msg_id, and the alarm msg_ids, from the committed schedule."""
    messages = json.load(open(f"workloads/{scenario}.json"))["messages"]
    arrival = {m["msg_id"]: m["t"] for m in messages}
    alarms = {m["msg_id"] for m in messages if m["is_alarm"]}
    return arrival, alarms


def _count(order: List[str], arrival: Dict[str, float], alarms: set) -> tuple:
    """Inversions in one delivered order, and the worst single alarm's overtakes.

    For each alarm, count the routine messages delivered before it that arrived
    after it. Routine arrivals are pre-sorted so each alarm costs one binary
    search rather than a scan over every routine message.
    """
    routine_arrivals: List[float] = []
    total = 0
    worst = 0
    for msg_id in order:
        if msg_id in alarms:
            # Routine messages already delivered whose arrival is later than this
            # alarm's: each one overtook the alarm.
            t_alarm = arrival[msg_id]
            overtakes = len(routine_arrivals) - bisect.bisect_right(
                routine_arrivals, t_alarm)
            total += overtakes
            worst = max(worst, overtakes)
        else:
            bisect.insort(routine_arrivals, arrival[msg_id])
    return total, worst


def main() -> None:
    parser = argparse.ArgumentParser(description="Priority inversions observed on the wire")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    arrival, alarms = _schedule(args.scenario)

    print(f"\n=== {args.scenario} — priority inversions (observed by an external subscriber) ===")
    print(f"{'scheduler':<9} {'reps':>5} {'inversions/rep':>22} {'worst alarm overtaken by':>26}")
    rows = []
    for scheduler in SCHEDULERS:
        traces = sorted(glob.glob(os.path.join(
            args.results_dir, f"observed_{scheduler}_{args.scenario}_rep*.csv")))
        if not traces:
            continue
        totals, worsts = [], []
        for trace in traces:
            with open(trace) as handle:
                order = [row["msg_id"] for row in csv.DictReader(handle)]
            total, worst = _count(order, arrival, alarms)
            totals.append(total)
            worsts.append(worst)
        values = np.array(totals, dtype=float)
        mean = float(values.mean())
        ci = (1.96 * float(values.std(ddof=1)) / np.sqrt(values.size)
              if values.size > 1 else 0.0)
        print(f"{scheduler:<9} {len(traces):>5} {mean:>12.1f} ±{ci:<9.1f} "
              f"{max(worsts):>20} msgs")
        rows.append({
            "scenario": args.scenario, "scheduler": scheduler, "reps": len(traces),
            "inversions_per_rep_mean": mean, "inversions_per_rep_ci95": ci,
            "worst_alarm_overtaken_by": max(worsts),
        })

    if not rows:
        raise SystemExit(f"no observer traces for {args.scenario} in {args.results_dir}")

    out = os.path.join(args.results_dir, f"inversions_{args.scenario}.csv")
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\ninversions -> {os.path.relpath(out)}")


if __name__ == "__main__":
    main()

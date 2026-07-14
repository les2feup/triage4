"""
Join broker overhead and client RTT, then report the R1.1 / R2.1 results.

For each scheduler present in the results directory, merges the broker-side
overhead CSV (``broker_<sched>_<scenario>.csv``, carrying band/overhead/dropped)
with the client-side RTT CSV(s) (``rtt_<sched>_<scenario>_*.csv``) on ``msg_id``
and reports:

    R1.1 — alarm-RTT vs routine-RTT (mean + 95% CI) per scheduler. The claim is
           the RELATIVE ordering across schedulers; absolute magnitudes differ
           from the DES because RTT adds fixed network + broker terms.
    R2.1 — per-message scheduling overhead (enqueue+select) mean/p50/p99 in µs,
           reported alongside the active-device count.

Writes ``summary_<scenario>.csv`` and prints a table. Plotting is attempted only
if matplotlib is available (it is not a prototype dependency).

Usage (from prototype/): python analyze.py --scenario c3_multi_zone_emergency
"""

import argparse
import csv
import glob
import os
from typing import Dict, List

import numpy as np

SCHEDULERS = ("fifo", "strict", "triage4")
BAND_ALARM = 0


def _read_csv(path: str) -> List[dict]:
    with open(path) as handle:
        return list(csv.DictReader(handle))


def _load_scheduler(results_dir: str, scheduler: str, scenario: str) -> List[dict]:
    """Merge broker overhead and all RTT shards for one scheduler on msg_id."""
    broker_path = os.path.join(results_dir, f"broker_{scheduler}_{scenario}.csv")
    if not os.path.exists(broker_path):
        return []
    # Join on (rep, msg_id): a msg_id is unique within a rep but repeats across reps.
    broker = {(row["rep"], row["msg_id"]): row for row in _read_csv(broker_path)}

    # The shard filename carries the zone that recorded it, which is the only
    # place the client's identity survives into the results. It matters when the
    # zone devices are not homogeneous (e.g. one on 2.4 GHz): the per-zone network
    # term is then a real offset that should be reported, not averaged away.
    prefix = f"rtt_{scheduler}_{scenario}_"
    rtt: Dict[tuple, dict] = {}
    for shard in glob.glob(os.path.join(results_dir, prefix + "*.csv")):
        zone = os.path.basename(shard)[len(prefix):].rsplit("_rep", 1)[0]
        for row in _read_csv(shard):
            rtt[(row["rep"], row["msg_id"])] = dict(row, zone=zone)

    merged = []
    for key, brow in broker.items():
        record = {
            "msg_id": brow["msg_id"],
            "rep": brow["rep"],
            "band": int(brow["band"]),
            "dropped": int(brow["dropped"]),
            "enqueue_ns": int(brow["enqueue_ns"]),
            "select_ns": int(brow["select_ns"]),
            "active_devices": int(brow["active_devices"]),
        }
        if key in rtt:
            record["rtt_ms"] = float(rtt[key]["rtt_ms"])
            record["zone"] = rtt[key]["zone"]
        merged.append(record)
    return merged


def _mean_ci(values: np.ndarray) -> tuple:
    """Mean and 95% CI half-width (normal approximation)."""
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, 0.0
    half = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return mean, half


def analyze(results_dir: str, scenario: str) -> List[dict]:
    summary = []
    print(f"\n=== {scenario} ===")
    header = (f"{'scheduler':<9} {'alarm_rtt_ms':>22} {'routine_rtt_ms':>22} "
              f"{'overhead_us(p50/p99)':>22} {'dropped':>8}")
    print(header)
    for scheduler in SCHEDULERS:
        records = _load_scheduler(results_dir, scheduler, scenario)
        if not records:
            continue
        rtts = {True: [], False: []}
        overheads = []
        dropped = 0
        for r in records:
            dropped += r["dropped"]
            overheads.append((r["enqueue_ns"] + r["select_ns"]) / 1000.0)
            if "rtt_ms" in r:
                rtts[r["band"] == BAND_ALARM].append(r["rtt_ms"])
        alarm = np.array(rtts[True])
        routine = np.array(rtts[False])
        overhead = np.array(overheads)
        a_mean, a_ci = _mean_ci(alarm)
        r_mean, r_ci = _mean_ci(routine)
        p50 = float(np.percentile(overhead, 50)) if overhead.size else float("nan")
        p99 = float(np.percentile(overhead, 99)) if overhead.size else float("nan")
        print(f"{scheduler:<9} {a_mean:>10.2f} ±{a_ci:<10.2f} "
              f"{r_mean:>10.2f} ±{r_ci:<10.2f} {p50:>9.2f}/{p99:<11.2f} {dropped:>8}")
        summary.append({
            "scenario": scenario, "scheduler": scheduler,
            "alarm_rtt_mean_ms": a_mean, "alarm_rtt_ci95_ms": a_ci,
            "routine_rtt_mean_ms": r_mean, "routine_rtt_ci95_ms": r_ci,
            "overhead_p50_us": p50, "overhead_p99_us": p99,
            "alarm_n": int(alarm.size), "routine_n": int(routine.size),
            "dropped": dropped,
        })
    return summary


def per_zone(results_dir: str, scenario: str) -> None:
    """RTT split by the zone device that recorded it.

    Exposes client heterogeneity. A device on a slower radio (2.4 GHz) carries a
    network offset that is common-mode across schedulers — so it cannot change the
    ordering result — but it does inflate that zone's absolute RTT. Reporting it
    per zone turns an unstated caveat into a measured number. Read it against the
    unsaturated baseline pass, where the queueing term is absent and what remains
    IS the network term.
    """
    print(f"\n=== {scenario} — RTT by zone device (ms) ===")
    print(f"{'scheduler':<9} {'zone':>5} {'alarm_rtt_ms':>22} {'routine_rtt_ms':>22} {'n':>6}")
    for scheduler in SCHEDULERS:
        records = [r for r in _load_scheduler(results_dir, scheduler, scenario)
                   if "rtt_ms" in r]
        if not records:
            continue
        zones: Dict[str, Dict[bool, list]] = {}
        for r in records:
            zones.setdefault(r["zone"], {True: [], False: []})
            zones[r["zone"]][r["band"] == BAND_ALARM].append(r["rtt_ms"])
        for zone in sorted(zones):
            a_mean, a_ci = _mean_ci(np.array(zones[zone][True]))
            r_mean, r_ci = _mean_ci(np.array(zones[zone][False]))
            n = len(zones[zone][True]) + len(zones[zone][False])
            print(f"{scheduler:<9} {zone:>5} {a_mean:>10.2f} ±{a_ci:<10.2f} "
                  f"{r_mean:>10.2f} ±{r_ci:<10.2f} {n:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prototype RTT + overhead")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--per-zone", action="store_true",
                        help="also break RTT down by zone device (heterogeneous clients)")
    args = parser.parse_args()

    summary = analyze(args.results_dir, args.scenario)
    if args.per_zone:
        per_zone(args.results_dir, args.scenario)
    if not summary:
        raise SystemExit(f"no results found for scenario {args.scenario}")

    out = os.path.join(args.results_dir, f"summary_{args.scenario}.csv")
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nsummary -> {os.path.relpath(out)}")


if __name__ == "__main__":
    main()

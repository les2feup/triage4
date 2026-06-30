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

    rtt: Dict[tuple, dict] = {}
    for shard in glob.glob(
            os.path.join(results_dir, f"rtt_{scheduler}_{scenario}_*.csv")):
        for row in _read_csv(shard):
            rtt[(row["rep"], row["msg_id"])] = row

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prototype RTT + overhead")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    summary = analyze(args.results_dir, args.scenario)
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

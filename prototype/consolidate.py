"""
Consolidate the raw Pi campaign into tidy, paper-ready CSVs.

The raw campaign is ~2500 per-zone / per-rep shards under ``--results-dir`` (the
data pulled off the Pi into ``results-pi/``). Tables and plots should not read
that sprawl directly. This reduces it to a few tidy CSVs under ``--out-dir``
(``results-consolidated/``, tracked in git) that are the single source of truth
for the hardware section:

    summary.csv         one row per (scheduler, scenario): alarm / routine RTT
                        (mean +/- 95% CI), overhead p50/p99, drops, inversions
                        per rep, worst-overtaken, and mean broker core use.
    per_zone.csv        one row per (scheduler, scenario, zone): RTT by the
                        device that recorded it — the geographic gradient.
    rtt_long.csv        one row per delivered message: for distribution plots.
    inversions_long.csv one row per (scheduler, scenario, rep): for box plots.

Stats reuse ``analyze._load_scheduler`` and ``inversions._count`` so there is one
implementation, not a second that can drift. Each row carries the manuscript
scenario ID (C3, R3) alongside the code key, so tables cite the paper's labels
directly (see ``docs/chat-reports/REFERENCE_Scenario_Crosswalk.md``).

Run from ``prototype/`` (paths to ``workloads/`` are relative, as in analyze.py):
    python consolidate.py --results-dir results-pi --out-dir results-consolidated
"""

import argparse
import csv
import glob
import os
import re
from typing import Dict, List

import numpy as np

from analyze import BAND_ALARM, SCHEDULERS, _load_scheduler, _mean_ci
from inversions import _count, _schedule

# Code key -> manuscript scenario ID. Extend here if more scenarios are run on
# hardware; the crosswalk reference is authoritative for the full mapping.
MANUSCRIPT_ID = {
    "c3_multi_zone_emergency": "C3",
    "r3_legit_extreme_emergency": "R3",
}


def _scenarios(results_dir: str) -> List[str]:
    """Scenario code keys present, discovered from the broker CSVs."""
    found = set()
    for path in glob.glob(os.path.join(results_dir, "broker_*.csv")):
        stem = os.path.basename(path)[len("broker_"):-len(".csv")]
        # broker_<scheduler>_<scenario>.csv, scheduler is a known token.
        for sched in SCHEDULERS:
            if stem.startswith(sched + "_"):
                found.add(stem[len(sched) + 1:])
    return sorted(found)


def _inversions_per_rep(results_dir: str, scheduler: str, scenario: str,
                        arrival: dict, alarms: set) -> List[dict]:
    """Inversions and worst-overtake for each observer trace of this cell."""
    rows = []
    prefix = f"observed_{scheduler}_{scenario}_rep"
    for trace in sorted(glob.glob(os.path.join(results_dir, prefix + "*.csv"))):
        rep = int(re.search(r"_rep(\d+)\.csv$", trace).group(1))
        with open(trace) as handle:
            order = [row["msg_id"] for row in csv.DictReader(handle)]
        total, worst = _count(order, arrival, alarms)
        rows.append({"rep": rep, "inversions": total, "worst_overtaken": worst})
    return rows


def _cpu_by_cell(results_dir: str) -> Dict[tuple, float]:
    """Mean broker core utilisation per (scheduler, scenario) from cpu.csv."""
    path = os.path.join(results_dir, "cpu.csv")
    if not os.path.exists(path):
        return {}
    util: Dict[tuple, List[float]] = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            key = (row["scheduler"], row["scenario"])
            util.setdefault(key, []).append(float(row["core_utilisation"]))
    return {k: float(np.mean(v)) for k, v in util.items()}


def consolidate(results_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    scenarios = _scenarios(results_dir)
    if not scenarios:
        raise SystemExit(f"no broker_*.csv found under {results_dir}")
    cpu = _cpu_by_cell(results_dir)

    summary, per_zone, rtt_long, inv_long = [], [], [], []
    complete = True

    for scenario in scenarios:
        mid = MANUSCRIPT_ID.get(scenario, scenario)
        arrival, alarms = _schedule(scenario)
        n_msgs = len(arrival)
        for scheduler in SCHEDULERS:
            records = _load_scheduler(results_dir, scheduler, scenario)
            if not records:
                continue
            delivered = [r for r in records if "rtt_ms" in r]
            reps = sorted({int(r["rep"]) for r in records})

            # --- aggregate RTT + overhead + drops (one summary row) ---
            alarm = np.array([r["rtt_ms"] for r in delivered if r["band"] == BAND_ALARM])
            routine = np.array([r["rtt_ms"] for r in delivered if r["band"] != BAND_ALARM])
            overhead = np.array([(r["enqueue_ns"] + r["select_ns"]) / 1000.0 for r in records])
            dropped = sum(r["dropped"] for r in records)
            a_mean, a_ci = _mean_ci(alarm)
            r_mean, r_ci = _mean_ci(routine)

            inv_rows = _inversions_per_rep(results_dir, scheduler, scenario, arrival, alarms)
            inv = np.array([x["inversions"] for x in inv_rows], dtype=float)
            i_mean, i_ci = _mean_ci(inv) if inv.size else (float("nan"), float("nan"))
            worst = max((x["worst_overtaken"] for x in inv_rows), default=0)
            for x in inv_rows:
                inv_long.append({"manuscript_id": mid, "scenario": scenario,
                                 "scheduler": scheduler, "rep": x["rep"],
                                 "inversions": x["inversions"],
                                 "worst_overtaken": x["worst_overtaken"]})

            summary.append({
                "manuscript_id": mid, "scenario": scenario, "scheduler": scheduler,
                "alarm_rtt_mean_ms": a_mean, "alarm_rtt_ci95_ms": a_ci,
                "routine_rtt_mean_ms": r_mean, "routine_rtt_ci95_ms": r_ci,
                "overhead_p50_us": float(np.percentile(overhead, 50)) if overhead.size else float("nan"),
                "overhead_p99_us": float(np.percentile(overhead, 99)) if overhead.size else float("nan"),
                "dropped": dropped,
                "inversions_per_rep_mean": i_mean, "inversions_per_rep_ci95": i_ci,
                "worst_alarm_overtaken": worst,
                "cpu_core_util_mean": cpu.get((scheduler, scenario), float("nan")),
                "alarm_n": int(alarm.size), "routine_n": int(routine.size),
                "reps": len(reps),
            })

            # --- per-zone RTT (the geographic gradient) ---
            zones: Dict[str, Dict[bool, list]] = {}
            for r in delivered:
                zones.setdefault(r["zone"], {True: [], False: []})
                zones[r["zone"]][r["band"] == BAND_ALARM].append(r["rtt_ms"])
                rtt_long.append({
                    "manuscript_id": mid, "scenario": scenario, "scheduler": scheduler,
                    "rep": r["rep"], "zone": r["zone"], "msg_id": r["msg_id"],
                    "band": r["band"], "is_alarm": int(r["band"] == BAND_ALARM),
                    "rtt_ms": r["rtt_ms"],
                })
            for zone in sorted(zones):
                za, zr = np.array(zones[zone][True]), np.array(zones[zone][False])
                za_m, za_c = _mean_ci(za)
                zr_m, zr_c = _mean_ci(zr)
                per_zone.append({
                    "manuscript_id": mid, "scenario": scenario, "scheduler": scheduler,
                    "zone": zone,
                    "alarm_rtt_mean_ms": za_m, "alarm_rtt_ci95_ms": za_c,
                    "routine_rtt_mean_ms": zr_m, "routine_rtt_ci95_ms": zr_c,
                    "n": int(za.size + zr.size),
                })

            # --- completeness: every rep must carry the whole workload ---
            broker_rows = len(records)
            expected = n_msgs * len(reps)
            zone_shards = len({(r["rep"], r.get("zone")) for r in delivered if r.get("zone")})
            flag = ""
            if broker_rows != expected:
                flag, complete = f"  !! broker rows {broker_rows} != {expected}", False
            print(f"{scheduler:<8} {scenario:<28} reps={len(reps):>2} "
                  f"broker={broker_rows:>5}/{expected:<5} drops={dropped:>3} inv/rep={i_mean:>7.1f}{flag}")

    _write(out_dir, "summary.csv", summary)
    _write(out_dir, "per_zone.csv", per_zone)
    _write(out_dir, "rtt_long.csv", rtt_long)
    _write(out_dir, "inversions_long.csv", inv_long)
    print(f"\n{'COMPLETE' if complete else 'INCOMPLETE — check flags above'} — "
          f"wrote 4 CSVs to {out_dir}")


def _write(out_dir: str, name: str, rows: List[dict]) -> None:
    if not rows:
        return
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {name}: {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate the Pi campaign into tidy CSVs")
    parser.add_argument("--results-dir", default="results-pi")
    parser.add_argument("--out-dir", default="results-consolidated")
    args = parser.parse_args()
    consolidate(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()

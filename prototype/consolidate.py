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
    comparisons.csv     one row per (scenario, scheduler): Welch's t-test of
                        mean alarm RTT against TRIAGE/4, with delta and p-value.

Stats reuse ``analyze._load_scheduler`` and ``inversions._count`` so there is one
implementation, not a second that can drift. Each row carries the manuscript
scenario ID (C3, R3) alongside the code key, so tables cite the paper's labels
directly (see ``docs/chat-reports/REFERENCE_Scenario_Crosswalk.md``).

Run from ``prototype/`` in the MAIN repository .venv, which provides
``assessment.metrics`` and scipy (paths to ``workloads/`` are relative, as in
analyze.py). This is a host-side step over data already pulled off the Pi, so
the isolated prototype venv is not the runtime here:

    ../.venv/bin/python consolidate.py --results-dir results-pi \
        --out-dir results-consolidated
"""

import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, List

import numpy as np

from triage4 import BandClassifier

# The canonical Welch implementation, shared with the simulation so hardware and
# simulation p-values are produced by the same code. This is why consolidate.py
# runs in the MAIN repository .venv, as generate_schedules.py does: it is a
# host-side step over data already pulled off the Pi, so the isolated prototype
# venv (which never imports assessment) is not the runtime here.
from assessment.metrics.compute import compare_schedulers

from analyze import BAND_ALARM, SCHEDULERS, _load_scheduler, _mean_ci, _read_csv
from inversions import _count, _schedule

# Code key -> manuscript scenario ID. Extend here if more scenarios are run on
# hardware; the crosswalk reference is authoritative for the full mapping.
MANUSCRIPT_ID = {
    "c3_multi_zone_emergency": "C3",
    "r1_alarm_flood_attack": "R1",
    "r2_alarm_malfunction_surge": "R2",
    "r3_legit_extreme_emergency": "R3",
}

# Scheduler each arm is tested against in comparisons.csv. TRIAGE/4 is the
# reference: every hardware claim is a claim about it relative to something else.
REFERENCE_SCHEDULER = "triage4"

# Band boundaries used by the broker config (zones 0-1 HIGH, 2-3 STANDARD, 4-5
# BACKGROUND). Only needed to reclassify shards when a broker CSV is absent.
_CLASSIFIER = BandClassifier(high_zone_max=1, standard_zone_max=3)

# Aggregate overhead p50/p99 (us) and broker core utilisation for the schedulers
# whose raw broker CSVs were cleared before the campaign was pulled off the Pi
# (fifo/strict/triage4 ran in an earlier session). These are the values already
# measured and reported in RESEARCH_Stage3b_Hardware_Results.md; the raw
# per-message overhead for these three is not recoverable, but the R2.1 table
# values are. Cells with a broker CSV present ignore this and use measured data.
_OVERHEAD_REPORTED = {
    ("fifo", "c3_multi_zone_emergency"): (1.31, 4.86),
    ("strict", "c3_multi_zone_emergency"): (4.65, 10.12),
    ("triage4", "c3_multi_zone_emergency"): (14.94, 43.88),
    ("fifo", "r3_legit_extreme_emergency"): (1.31, 2.11),
    ("strict", "r3_legit_extreme_emergency"): (4.48, 7.78),
    ("triage4", "r3_legit_extreme_emergency"): (13.68, 26.46),
}
_CPU_REPORTED = 0.001  # reported broker core utilisation for the earlier session


def _msg_meta(scenario: str) -> Dict[str, tuple]:
    """msg_id -> (zone_priority, is_alarm) from the committed schedule."""
    messages = json.load(open(f"workloads/{scenario}.json"))["messages"]
    return {m["msg_id"]: (m["zone_priority"], bool(m["is_alarm"])) for m in messages}


def _load_from_shards(results_dir: str, scheduler: str, scenario: str,
                      meta: Dict[str, tuple]) -> List[dict]:
    """Rebuild a cell's delivered records from RTT shards alone.

    Used when the broker CSV was lost: band comes from the schedule
    (zone -> band, alarm overrides) instead of the broker's own labelling, which
    is exactly what the broker would have recorded. Overhead and drops are not in
    the shards, so they are left absent here and filled from reported aggregates.
    """
    prefix = f"rtt_{scheduler}_{scenario}_"
    records = []
    for shard in glob.glob(os.path.join(results_dir, prefix + "*.csv")):
        zone = os.path.basename(shard)[len(prefix):].rsplit("_rep", 1)[0]
        for row in _read_csv(shard):
            zone_priority, is_alarm = meta[row["msg_id"]]
            records.append({
                "msg_id": row["msg_id"], "rep": row["rep"],
                "band": _CLASSIFIER.classify(zone_priority, is_alarm),
                "dropped": 0, "rtt_ms": float(row["rtt_ms"]), "zone": zone,
            })
    return records


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
    # (scenario, scheduler) -> per-rep mean alarm RTT. The rep is the unit of
    # analysis for the significance test: messages inside one rep share a queue
    # and are not independent, so pooling them would overstate the sample size.
    per_rep_alarm: Dict[tuple, List[float]] = {}
    complete = True

    for scenario in scenarios:
        mid = MANUSCRIPT_ID.get(scenario, scenario)
        arrival, alarms = _schedule(scenario)
        meta = _msg_meta(scenario)
        n_msgs = len(arrival)
        for scheduler in SCHEDULERS:
            has_broker = os.path.exists(
                os.path.join(results_dir, f"broker_{scheduler}_{scenario}.csv"))
            if has_broker:
                records = _load_scheduler(results_dir, scheduler, scenario)
            else:
                records = _load_from_shards(results_dir, scheduler, scenario, meta)
            if not records:
                continue
            delivered = [r for r in records if "rtt_ms" in r]
            reps = sorted({int(r["rep"]) for r in records})

            # --- aggregate RTT + overhead + drops (one summary row) ---
            alarm = np.array([r["rtt_ms"] for r in delivered if r["band"] == BAND_ALARM])
            routine = np.array([r["rtt_ms"] for r in delivered if r["band"] != BAND_ALARM])
            dropped = sum(r["dropped"] for r in records)
            a_mean, a_ci = _mean_ci(alarm)
            r_mean, r_ci = _mean_ci(routine)

            per_rep_alarm[(scenario, scheduler)] = [
                float(np.mean([r["rtt_ms"] for r in delivered
                               if r["band"] == BAND_ALARM and int(r["rep"]) == rep]))
                for rep in reps
                if any(r["band"] == BAND_ALARM and int(r["rep"]) == rep for r in delivered)
            ]

            # Overhead is measured only when the broker CSV survived; otherwise use
            # the aggregate already reported for that earlier session (raw
            # per-message overhead for those schedulers is not recoverable).
            if has_broker:
                overhead = np.array([(r["enqueue_ns"] + r["select_ns"]) / 1000.0 for r in records])
                p50 = float(np.percentile(overhead, 50))
                p99 = float(np.percentile(overhead, 99))
                overhead_source, cpu_util = "measured", cpu.get((scheduler, scenario), float("nan"))
            else:
                p50, p99 = _OVERHEAD_REPORTED.get((scheduler, scenario), (float("nan"), float("nan")))
                overhead_source, cpu_util = "reported", _CPU_REPORTED

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
                "overhead_p50_us": p50, "overhead_p99_us": p99,
                "overhead_source": overhead_source,
                "dropped": dropped,
                "inversions_per_rep_mean": i_mean, "inversions_per_rep_ci95": i_ci,
                "worst_alarm_overtaken": worst,
                "cpu_core_util_mean": cpu_util,
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
            n_records = len(records)
            expected = n_msgs * len(reps)
            flag = ""
            if n_records != expected:
                flag, complete = f"  !! {n_records} records != {expected}", False
            print(f"{scheduler:<8} {scenario:<28} reps={len(reps):>2} "
                  f"msgs={n_records:>5}/{expected:<5} drops={dropped:>3} "
                  f"inv/rep={i_mean:>7.1f}  overhead={overhead_source}{flag}")

    comparisons = _compare(scenarios, per_rep_alarm)

    _write(out_dir, "summary.csv", summary)
    _write(out_dir, "per_zone.csv", per_zone)
    _write(out_dir, "rtt_long.csv", rtt_long)
    _write(out_dir, "inversions_long.csv", inv_long)
    _write(out_dir, "comparisons.csv", comparisons)
    print(f"\n{'COMPLETE' if complete else 'INCOMPLETE — check flags above'} — "
          f"wrote 5 CSVs to {out_dir}")


def _compare(scenarios: List[str], per_rep_alarm: Dict[tuple, List[float]]) -> List[dict]:
    """Welch's t-test on mean alarm RTT, TRIAGE/4 against every other arm.

    Reported per scenario so the paper can state each latency claim with a
    p-value instead of leaving the reader to infer significance from overlapping
    error bars. Welch rather than Student because the arms have plainly unequal
    variance: a scheduler that queues alarms behind telemetry is both slower and
    far more variable than one that does not.
    """
    rows = []
    for scenario in scenarios:
        ref = per_rep_alarm.get((scenario, REFERENCE_SCHEDULER))
        if not ref or len(ref) < 2:
            continue
        for scheduler in SCHEDULERS:
            if scheduler == REFERENCE_SCHEDULER:
                continue
            other = per_rep_alarm.get((scenario, scheduler))
            if not other or len(other) < 2:
                continue
            comp = compare_schedulers(
                REFERENCE_SCHEDULER, scheduler, "alarm_rtt_ms", ref, other)
            rows.append({
                "manuscript_id": MANUSCRIPT_ID.get(scenario, scenario),
                "scenario": scenario,
                "reference": REFERENCE_SCHEDULER,
                "scheduler": scheduler,
                "reference_mean_ms": comp.mean_a,
                "scheduler_mean_ms": comp.mean_b,
                "delta_pct": comp.delta_pct,
                "t_statistic": comp.t_statistic,
                "p_value": comp.p_value,
                "significance": comp.significance_marker(),
                "n_reps_reference": len(ref),
                "n_reps_scheduler": len(other),
            })
            # delta_pct is the other arm relative to TRIAGE/4: positive means
            # that arm is slower. Spelled out because "a vs b +200%" reads either way.
            print(f"{'welch':<8} {scenario:<28} {scheduler:<17} "
                  f"{comp.delta_pct:>+9.1f}% vs {REFERENCE_SCHEDULER}  "
                  f"p={comp.p_value:<10.3g} {comp.significance_marker()}")
    return rows


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

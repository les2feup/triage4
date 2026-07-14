#!/usr/bin/env bash
# Pull the client-side results into the Pi's results/ directory, then verify the
# campaign is complete.
#
# The RTT shards live on the zone devices (each agent measures its own single-clock
# RTT and writes locally) and the delivery-order traces live on the observer; only
# the broker overhead and CPU samples are already here. Analysis needs all of it in
# one place.
#
# Verification is the point of the second half: a shard missing because a device was
# unreachable looks exactly like a shard missing because a zone stayed silent, and
# analyze.py would happily average over the gap. So the expected counts are derived
# from the schedules and the broker CSVs and checked before anything is analysed.
#
# Usage (on the Pi, from prototype/):
#   cp pi_hosts.conf.example pi_hosts.conf && $EDITOR pi_hosts.conf
#   ./collect_results.sh
#
# Env knobs: HOSTS (config path), REMOTE_DIR (results dir on the clients), RESULTS.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
HOSTS=${HOSTS:-pi_hosts.conf}
REMOTE_DIR=${REMOTE_DIR:-Developer/triage4/prototype/results}
RESULTS=${RESULTS:-results}
mkdir -p "$RESULTS"

if [ ! -f "$HOSTS" ]; then
  echo "no host list at $HOSTS — copy pi_hosts.conf.example and edit it." >&2
  exit 1
fi

failed=()
while read -r name target; do
  case "$name" in ''|'#'*) continue ;; esac
  printf '%-9s %-24s ' "$name" "$target"
  # Both patterns are requested from every device: a zone device holds rtt_*.csv and
  # the observer holds observed_*.csv, and rsync tolerates a pattern matching nothing.
  if rsync -q --ignore-missing-args \
        "$target:$REMOTE_DIR/rtt_*.csv" "$target:$REMOTE_DIR/observed_*.csv" \
        "$RESULTS/" 2>/dev/null; then
    echo "ok"
  else
    echo "FAILED"
    failed+=("$name ($target)")
  fi
done <"$HOSTS"

if [ ${#failed[@]} -gt 0 ]; then
  echo >&2
  echo "could not collect from: ${failed[*]}" >&2
  echo "fix these before analysing — a missing device is a hole in the data, not a gap you can average over." >&2
  exit 1
fi

echo
$PY - "$RESULTS" <<'EOF'
"""Check the collected campaign is complete before any of it is believed."""
import collections, csv, glob, json, os, re, sys

results = sys.argv[1]
ok = True

for broker_csv in sorted(glob.glob(os.path.join(results, "broker_*.csv"))):
    stem = os.path.basename(broker_csv)[len("broker_"):-len(".csv")]
    scheduler, scenario = stem.split("_", 1)

    with open(broker_csv) as handle:
        rows = list(csv.DictReader(handle))
    reps = sorted({int(r["rep"]) for r in rows})
    messages = json.load(open(f"workloads/{scenario}.json"))["messages"]
    n_msgs = len(messages)

    # Every rep must carry the whole workload: a short rep means the broker served
    # only part of it, which no amount of averaging recovers.
    per_rep = collections.Counter(r["rep"] for r in rows)
    short = {rep: n for rep, n in per_rep.items() if n != n_msgs}

    shards = glob.glob(os.path.join(results, f"rtt_{scheduler}_{scenario}_*.csv"))
    zones = {re.search(r"_([^_]+)_rep\d+\.csv$", s).group(1) for s in shards}
    # Expected zones come from the SCHEDULE, never from the shards present: deriving
    # them from what was collected is circular, and an unreachable device would
    # silently shrink the expectation to match its own absence. ("all" is the host
    # topology, where one process replays every zone.)
    if zones == {"all"}:
        expected_zones = {"all"}
    else:
        expected_zones = {str(m["zone_priority"]) for m in messages}
    expected_shards = len(expected_zones) * len(reps)

    print(f"{scheduler:<8} {scenario:<28} "
          f"reps={len(reps):>2}  broker_rows={len(rows):>5}/{n_msgs * len(reps):<5} "
          f"zones={len(zones)}/{len(expected_zones)}  shards={len(shards):>3}/{expected_shards}")
    if short:
        print(f"  !! reps with wrong row count (expected {n_msgs}): {dict(short)}")
        ok = False
    missing_zones = expected_zones - zones
    if missing_zones:
        print(f"  !! no RTT shards from zone(s): {sorted(missing_zones)}")
        ok = False
    if len(shards) != expected_shards:
        print(f"  !! expected {expected_shards} RTT shards, found {len(shards)}")
        ok = False
    if reps != list(range(len(reps))):
        print(f"  !! non-contiguous reps: {reps}")
        ok = False

observed = glob.glob(os.path.join(results, "observed_*.csv"))
print(f"\nobserver traces: {len(observed)}")
if not observed:
    print("  !! no observer traces — the independent delivery-order evidence is missing")
    ok = False

print("\nCOMPLETE — safe to analyse." if ok else "\nINCOMPLETE — do not analyse until resolved.")
sys.exit(0 if ok else 1)
EOF

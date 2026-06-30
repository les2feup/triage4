#!/usr/bin/env bash
# One-command host (loopback) run of the full matrix:
#   {fifo, strict, triage4} x {c3, r3} x REPS reps.
#
# For each cell it starts the broker (one scheduler per process), replays the
# pre-generated schedule through a single zone_client (all zones, single-clock
# RTT on loopback), then runs analyze.py per scenario. AAP is enabled only for
# triage4 (required for the R3 no-shed confirmation).
#
# Env knobs: PORT, REPS, SCHEDULERS, SCENARIOS. C (egress rate) per scenario is
# set so the offered load saturates the egress (rho > 1) and a real queue builds.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
HOST=127.0.0.1
PORT=${PORT:-1885}
REPS=${REPS:-1}
SCHEDULERS=${SCHEDULERS:-"fifo strict triage4"}
SCENARIOS=${SCENARIOS:-"c3_multi_zone_emergency r3_legit_extreme_emergency"}
RESULTS=results
mkdir -p "$RESULTS"

declare -A SCHED_FILE=(
  [c3_multi_zone_emergency]=workloads/c3_multi_zone_emergency.json
  [r3_legit_extreme_emergency]=workloads/r3_legit_extreme_emergency.json
)
# Egress rate C (msg/s): below each scenario's offered rate so rho > 1.
declare -A RATE_C=(
  [c3_multi_zone_emergency]=5
  [r3_legit_extreme_emergency]=8
)

# Fresh run: clear prior CSVs so append-mode broker files do not mix runs.
rm -f "$RESULTS"/broker_*.csv "$RESULTS"/rtt_*.csv "$RESULTS"/summary_*.csv

wait_port() {
  for _ in $(seq 1 50); do
    $PY -c "import socket;s=socket.socket();s.settimeout(0.2);s.connect(('$HOST',$PORT));s.close()" 2>/dev/null && return 0
    sleep 0.1
  done
  return 1
}

for scenario in $SCENARIOS; do
  schedule=${SCHED_FILE[$scenario]}
  C=${RATE_C[$scenario]}
  for sched in $SCHEDULERS; do
    aap=""
    [ "$sched" = "triage4" ] || aap="--no-aap"
    for rep in $(seq 0 $((REPS - 1))); do
      $PY -m broker.server --host "$HOST" --port "$PORT" --scheduler "$sched" \
          --rate-c "$C" --scenario "$scenario" --rep "$rep" \
          --results-dir "$RESULTS" $aap >"$RESULTS/.broker_${sched}_${scenario}_${rep}.log" 2>&1 &
      bpid=$!
      if ! wait_port; then
        echo "broker failed to start" >&2
        cat "$RESULTS/.broker_${sched}_${scenario}_${rep}.log" >&2
        kill "$bpid" 2>/dev/null || true
        exit 1
      fi
      $PY -m clients.zone_client --schedule "$schedule" --host "$HOST" --port "$PORT" \
          --scheduler "$sched" --scenario "$scenario" --rep "$rep" --out-dir "$RESULTS"
      kill -TERM "$bpid" 2>/dev/null || true   # broker flushes overhead CSV on SIGTERM
      wait "$bpid" 2>/dev/null || true
    done
  done
  $PY analyze.py --results-dir "$RESULTS" --scenario "$scenario"
done

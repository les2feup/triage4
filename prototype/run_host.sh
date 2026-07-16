#!/usr/bin/env bash
# One-command host (loopback) run of the full matrix:
#   six schedulers x {c3, hw_flood, r3} x REPS reps, plus the t4-nosourcelimit
#   ablation on the AAP scenarios (hw_flood, r3).
#
# For each cell it starts the broker (one scheduler per process), replays the
# pre-generated schedule through a single zone_client (all zones, single-clock
# RTT on loopback), then runs analyze.py per scenario. AAP is enabled for the
# TRIAGE/4 arms only (see AAP_SCHEDULERS): R3 needs it for the no-shed
# confirmation, hw_flood for the containment comparison against t4-nosourcelimit.
#
# This is the loopback developer run: one client replays every zone, so it is
# not the six-device testbed and its RTTs are not the reported hardware numbers.
# It exists to check the wiring end to end before the Pi campaign.
#
# Env knobs: PORT, REPS, SCHEDULERS, SCENARIOS. C (egress rate) per scenario is
# set so the offered load saturates the egress (rho > 1) and a real queue builds.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
HOST=127.0.0.1
PORT=${PORT:-1885}
REPS=${REPS:-1}
SCHEDULERS=${SCHEDULERS:-"fifo strict wfq drr tbp triage4"}
ABLATION=${ABLATION:-"t4-nosourcelimit"}
SCENARIOS=${SCENARIOS:-"c3_multi_zone_emergency hw_flood_attack r3_legit_extreme_emergency"}
# Scenarios the ablation arm joins; mirrors ABLATION_SCENARIOS in consolidate.py.
ABLATION_SCENARIOS="hw_flood_attack r3_legit_extreme_emergency"
# Arms that run with Adaptive Alarm Protection enabled; mirrors AAP_SCHEDULERS
# in broker/config.py. Everything else is launched with --no-aap.
AAP_SCHEDULERS="triage4 t4-nosourcelimit"
RESULTS=results
mkdir -p "$RESULTS"

declare -A SCHED_FILE=(
  [c3_multi_zone_emergency]=workloads/c3_multi_zone_emergency.json
  [hw_flood_attack]=workloads/hw_flood_attack.json
  [r3_legit_extreme_emergency]=workloads/r3_legit_extreme_emergency.json
)
# Egress rate C (msg/s): below each scenario's offered rate so rho > 1. Each
# value holds rho near 1.4, so no scenario is saturated harder than another.
declare -A RATE_C=(
  [c3_multi_zone_emergency]=5
  [hw_flood_attack]=23
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
  # The ablation arm only joins the scenarios in ABLATION_SCENARIOS.
  arms="$SCHEDULERS"
  case " $ABLATION_SCENARIOS " in *" $scenario "*) arms="$arms $ABLATION" ;; esac
  for sched in $arms; do
    # AAP stays on for both TRIAGE/4 arms. t4-nosourcelimit must keep it: the
    # ablation removes the per-source layer only, so --no-aap here would strip
    # the backstop too and compare against a scheduler that never sheds at all.
    aap="--no-aap"
    case " $AAP_SCHEDULERS " in *" $sched "*) aap="" ;; esac
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

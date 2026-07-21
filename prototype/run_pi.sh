#!/usr/bin/env bash
# One-command Pi run of the full matrix: six schedulers x {c3, hw_flood, r3} x REPS,
# plus the t4-nosourcelimit ablation on the AAP scenarios (hw_flood, r3).
#
# Runs ON THE PI (the broker). The six zone devices each run a long-lived
#   python -m clients.zone_agent --zone <0..5> --host <pi>
# and stay idle until this script announces a cell on t4ctl/go. Per cell it:
#   start broker (one scheduler per process) -> GO -> wait out the replay+drain
#   -> SIGTERM (broker flushes its overhead CSV) -> sample broker CPU.
# The agents write their own RTT CSVs locally; collect them afterwards (see
# docs/PI_TESTBED_GUIDE.md) and run analyze.py on the merged results.
#
# The CPU sample is the R2.1 sanity guard: if the broker approaches one full
# core, the Pi itself has become the bottleneck and the measured overhead is
# reporting CPU contention rather than scheduling cost. Re-tune C if so.
#
# Env knobs: PORT, REPS, SCHEDULERS, SCENARIOS, DRAIN_C3/DRAIN_HW/DRAIN_R3 (per-
# scenario drain seconds), ZONES, RESULTS, and
# RATE_C_C3 / RATE_C_HW / RATE_C_R3 (override for the unsaturated network
# baseline, e.g. RATE_C_C3=1000).
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
BIND=0.0.0.0
PORT=${PORT:-1883}
REPS=${REPS:-30}
# Baselines and TRIAGE/4 run on every scenario. The t4-nosourcelimit ablation
# runs only on the AAP scenarios (hw_flood, r3), where the per-source layer it
# removes actually engages; on C3 it would just duplicate triage4. This mirrors
# _expected_cells() in consolidate.py, which is what checks the campaign is
# whole. Override SCHEDULERS/ABLATION to run a subset, but a partial run will
# fail consolidation until every expected cell is present.
SCHEDULERS=${SCHEDULERS:-"fifo strict wfq drr tbp triage4"}
ABLATION=${ABLATION:-"t4-nosourcelimit"}
SCENARIOS=${SCENARIOS:-"c3_multi_zone_emergency hw_flood_attack r3_legit_extreme_emergency"}
# Scenarios the ablation arm joins; mirrors ABLATION_SCENARIOS in consolidate.py.
# The flood (where it should shed legitimate alarms the per-source layer spares)
# and R3 (the control, where neither arm sheds). C3 never engages AAP.
ABLATION_SCENARIOS="hw_flood_attack r3_legit_extreme_emergency"
# Arms that run with Adaptive Alarm Protection enabled; mirrors AAP_SCHEDULERS
# in broker/config.py. Everything else is launched with --no-aap.
AAP_SCHEDULERS="triage4 t4-nosourcelimit"
# Drain per scenario (s): how long to keep the broker up past the last arrival.
# Most scenarios empty their queue in seconds, so 30 s is ample. HW-Flood is the
# exception because TBP is non-work-conserving: it holds the 526-message background
# band (zones 4-5) behind a 5 token/s bucket and delivers it over ~100 s. It never
# drops these, only delays them, so a shorter window would cut that tail off and the
# late deliveries would vanish from every recorder at once. The zone agents' and
# observer's --drain must cover the same tail (see docs/PI_TESTBED_GUIDE.md).
declare -A DRAIN=(
  [c3_multi_zone_emergency]=${DRAIN_C3:-30}
  [hw_flood_attack]=${DRAIN_HW:-100}
  [r3_legit_extreme_emergency]=${DRAIN_R3:-30}
)
ZONES=${ZONES:-6}          # zone agents that must be ready before each cell fires
RESULTS=${RESULTS:-results}
mkdir -p "$RESULTS"

# Egress rate C (msg/s): below each scenario's offered rate so rho > 1. Re-tune
# on the Pi if the CPU sample shows the broker saturating a core.
declare -A RATE_C=(
  [c3_multi_zone_emergency]=${RATE_C_C3:-5}
  [hw_flood_attack]=${RATE_C_HW:-23}
  [r3_legit_extreme_emergency]=${RATE_C_R3:-8}
)

# Only the cells this invocation is about to produce are cleared, so a campaign
# can be extended one scheduler at a time (SCHEDULERS=wfq) without destroying the
# arms already collected. Broker CSVs are append-mode, so a stale file from a
# previous run of the SAME cell would otherwise double up.
for scenario in $SCENARIOS; do
  arms="$SCHEDULERS"
  case " $ABLATION_SCENARIOS " in *" $scenario "*) arms="$arms $ABLATION" ;; esac
  for sched in $arms; do
    rm -f "$RESULTS/broker_${sched}_${scenario}.csv"
  done
done

# Wall-clock span of a scenario: its last arrival offset, seconds (rounded up).
span() {
  $PY -c "import json,math;print(math.ceil(json.load(open('workloads/$1.json'))['messages'][-1]['t']))"
}

# Broker CPU seconds consumed so far (utime + stime from /proc/<pid>/stat).
cpu_seconds() {
  awk -v hz="$(getconf CLK_TCK)" '{print ($14 + $15) / hz}' "/proc/$1/stat"
}

port_open() {
  $PY -c "import socket;s=socket.socket();s.settimeout(0.2);s.connect(('127.0.0.1',$PORT));s.close()" 2>/dev/null
}

# Wait for OUR broker, not merely for an open port. A stale broker or a system
# mosquitto squatting on $PORT would make every cell's broker die on bind while
# the port still answers -- the agents would then replay into a foreign broker and
# the campaign would look healthy while producing nothing. So liveness of $bpid is
# checked on every attempt, and a dead pid fails the cell immediately.
wait_broker() {
  local bpid=$1
  for _ in $(seq 1 50); do
    kill -0 "$bpid" 2>/dev/null || return 1
    port_open && return 0
    sleep 0.1
  done
  return 1
}

# The port must be free before the campaign starts: if something already holds it,
# every broker we launch dies on bind and nothing downstream notices.
if port_open; then
  echo "port $PORT is already in use -- stop the process holding it before running." >&2
  echo "  ss -lptn 'sport = :$PORT'" >&2
  echo "  systemctl is-active mosquitto" >&2
  exit 1
fi

# Append: a later invocation adding a scheduler must not discard the CPU samples
# of the arms already run. The scheduler column disambiguates the rows.
if [ ! -f "$RESULTS/cpu.csv" ]; then
  echo "rep,scheduler,scenario,cpu_seconds,wall_seconds,core_utilisation" >"$RESULTS/cpu.csv"
fi

for scenario in $SCENARIOS; do
  C=${RATE_C[$scenario]}
  cell_seconds=$(( $(span "$scenario") + ${DRAIN[$scenario]} ))
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
      cell="${sched}:${scenario}:${rep}"
      log="$RESULTS/.broker_${sched}_${scenario}_${rep}.log"

      $PY -m broker.server --host "$BIND" --port "$PORT" --scheduler "$sched" \
          --rate-c "$C" --scenario "$scenario" --rep "$rep" \
          --results-dir "$RESULTS" $aap >"$log" 2>&1 &
      bpid=$!
      if ! wait_broker "$bpid"; then
        echo "broker failed to start for cell $cell" >&2
        cat "$log" >&2
        kill "$bpid" 2>/dev/null || true
        exit 1
      fi

      cpu_before=$(cpu_seconds "$bpid")
      wall_before=$SECONDS
      # Blocks until all ZONES agents have reconnected to this cell's broker and
      # resubscribed; exits non-zero (aborting the campaign) if one never does.
      $PY -m clients.go --host 127.0.0.1 --port "$PORT" --cell "$cell" --expect "$ZONES"
      sleep "$cell_seconds"
      # A broker that died mid-cell served only part of the workload, so the cell
      # is void: fail loudly rather than recording a partial result as if it were
      # a whole one.
      if ! kill -0 "$bpid" 2>/dev/null; then
        echo "broker died during cell $cell -- results are void" >&2
        cat "$log" >&2
        exit 1
      fi
      cpu_used=$($PY -c "print($(cpu_seconds "$bpid") - $cpu_before)")
      wall_used=$((SECONDS - wall_before))

      kill -TERM "$bpid" 2>/dev/null || true   # SIGTERM, never SIGINT (see guide)
      wait "$bpid" 2>/dev/null || true
      $PY -c "print(f'$rep,$sched,$scenario,{$cpu_used:.3f},$wall_used,{$cpu_used/$wall_used:.3f}')" \
          >>"$RESULTS/cpu.csv"
    done
  done
done

echo "broker overhead -> $RESULTS/broker_*.csv"
echo "broker CPU      -> $RESULTS/cpu.csv"
echo "now collect the zone agents' rtt_*.csv into $RESULTS, then run:"
for scenario in $SCENARIOS; do
  echo "  $PY analyze.py --results-dir $RESULTS --scenario $scenario"
done

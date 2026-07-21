# Raspberry Pi Testbed Guide

Reproduces the TRIAGE/4 broker prototype on real hardware: a Raspberry Pi hosting
the MQTT 5.0 broker, and six client devices, one per geographic priority. This is
the evidence for reviewer **R1.1** (the scheduler works on a real broker, not only
in simulation) and **R2.1** (its per-message cost on gateway-class hardware).

Fields marked *(measure)* are filled in from your own run — they are results, not
configuration inputs.

## 1. Topology

```
  zone agent 0 ─┐
  zone agent 1 ─┤
  zone agent 2 ─┼── WiFi (5 GHz) ──> [ AP / router ] ──Ethernet──> [ Raspberry Pi ]
  zone agent 3 ─┤                          │                        broker + dispatcher
  zone agent 4 ─┤                          │
  zone agent 5 ─┘                     Ethernet
                                           │
                                    [ control centre ]
                                    passive observer
```

**The Pi is not the access point.** It is wired to the AP by Ethernet. This is
deliberate: R2.1's headline is the broker's per-message scheduling overhead, and if
the Pi also ran `hostapd` plus the softirq path for every 802.11 frame from six
clients, the measured `enqueue`/`select_next` timings would absorb CPU contention
that has nothing to do with TRIAGE/4. Wiring the Pi keeps its CPU doing one job,
and still leaves exactly one wireless hop between a client and the broker.

If you must make the Pi the AP, pin the broker to an isolated core, move IRQ
affinity off that core, and additionally run the matrix over Ethernet as a control
so the AP's contribution to overhead can be subtracted rather than assumed away.

**Six devices, one per `zone_priority` (0–5).** All three scenarios' messages carry
a `zone_priority` in 0–5; that is what the scheduler bands on and what the return
topic `t4out/<zone>` is keyed by, so each device owns exactly one return topic and
receives only its own traffic. (In R3 the ten *logical* zones fold onto these six
priorities via `zone_id % 6`; in HW-Flood the attacker is the single device on zone
5 and the five legitimate sources are one device each on zones 0–4. The six-device
split follows the priorities the scheduler actually sees.)

## 2. Why no clock synchronisation is needed

Each agent publishes its own messages and subscribes to its own return topic, so
`t_send` and `t_recv` for any message are stamped by the **same process's**
`time.monotonic_ns()`. RTT is therefore a single-clock measurement and **NTP/PTP is
not required** — clocks are never compared across devices. The broker's own clock
is used only for scheduling and never enters an RTT.

## 2b. The schedules — committed, distributed by git, never regenerated per device

Every device replays the **same** fixed arrival schedule: one committed JSON per
scenario under `workloads/` — `c3_multi_zone_emergency.json`, `hw_flood_attack.json`,
`r3_legit_extreme_emergency.json`. These *are* the workload; the broker and clients
only replay them, and the consolidator joins every RTT and every observed delivery
back to them on `msg_id`.

**A `msg_id` names a position in the schedule, not a fixed message.** Regenerating a
scenario with different jitter reshuffles which message owns which id, so results
recorded against one schedule join cleanly against a differently-jittered copy and
silently describe *different* messages — the exact accident that once made a baseline
look like it beat TRIAGE/4. To make that impossible, every RTT shard and every
observer trace carries a 12-character fingerprint of the schedule it ran against, and
`consolidate.py` refuses to consolidate any shard whose fingerprint does not match the
JSON on disk. A stale or unfingerprinted shard aborts the run rather than averaging
into it.

The one invariant this rests on: **every device holds byte-identical schedule JSON.**
Git gives you that for free — clone or `git pull` the *same commit* on the Pi, all six
zone devices, and the observer, and the fingerprints agree automatically. This is the
whole distribution mechanism; there is no copy step and no per-device generation.

**Do not run `generate_schedules.py` on the broker or the clients.** It is a
maintainer/build tool, not a runtime step:

- It imports `assessment.workloads`, which exists only in the **main repository**
  venv. The prototype venv the broker and clients use installs `triage4` + `paho-mqtt`
  and nothing else, so it *cannot* run the generator at all.
- Its output is deterministic and already committed. Regenerating reproduces the
  committed JSON to the same fingerprint (verified under numpy 2.x), so running it
  gains nothing and risks one device's numpy emitting a hairsbreadth-different float
  that desynchronises its fingerprint.

You touch the generator only to deliberately **change a scenario**, and then the
lifecycle is: regenerate in the main venv → commit the JSON → `git pull` everywhere.
Never regenerate on a single device and never hand-edit a JSON — either desynchronises
that device and fails consolidation by design.

```bash
# Maintainer only, from the repo root, in the MAIN venv (the one with assessment/):
.venv/bin/python -m prototype.workloads.generate_schedules
git add prototype/workloads/*.json && git commit    # then git pull on every device
```

So the pre-campaign checklist for schedules is simply: **`git pull` on the Pi, all six
zone devices, and the observer, and confirm all are on the same commit.** Nothing is
generated at run time.

## 3. Pi setup (the broker)

1. **Image:** Raspberry Pi OS Lite 64-bit, SSH and avahi enabled (headless). Record
   the model and clock speed: an overhead number is only meaningful next to the
   hardware it was measured on. *(measure: model, RAM, kernel)*
2. **Network:** Ethernet to the AP. Static lease, or reach it as `<pi-host>.local`.
   Open TCP **1883**.
3. **Runtime** (Python 3.11+):

   ```bash
   git clone <repo> && cd triage4/prototype
   python -m venv .venv
   .venv/bin/pip install -r requirements.txt     # triage4==1.2.0 + paho-mqtt
   ```

   The Pi installs the **published** `triage4==1.2.0` wheel from PyPI — the same
   artifact the host runs used. Nothing is cross-compiled, and nothing is imported
   from the repo's `src/`, so the code path under measurement is the one a user
   would actually deploy.

## 4. Client setup (each of the six zone devices)

Same clone and venv as above (any Linux/macOS host: laptops, Pi Zeros, spare Pis).
Then on device *z* (one of 0…5):

```bash
.venv/bin/python -m clients.zone_agent --zone <z> --host <pi-host>.local --drain 120
```

`--drain 120` is sized for HW-Flood: under TBP a zone-4 background message can wait
~100 s behind the flood before it is delivered, and the agent must still be
listening on `t4out/<z>` when its echo arrives or the RTT is lost. The agent stops
early once every message it sent has returned, so the long drain costs nothing on
the cells that finish quickly (see §4b for why HW-Flood is the exception).

Each agent connects, subscribes to `t4ctl/go` and `t4out/<z>`, and idles. It stays
running for the **whole campaign** — do not restart it between cells. While idle it
heartbeats its zone id on `t4ctl/ready`, which is how the coordinator proves all six
devices are present before starting a cell.

### Heterogeneous client devices

The zone devices do not have to be identical, and in practice they will not be. A
client only publishes, subscribes and timestamps — **it never imports `triage4`**
(the scheduler runs in the broker), so a client device needs only `paho-mqtt`:

```bash
.venv/bin/pip install -r requirements-client.txt     # paho-mqtt only
```

**A device stuck on 2.4 GHz** (e.g. a Pi Zero W, whose radio is 2.4 GHz only) is
acceptable. Run the AP dual-band and bridge both radios to the same LAN. The bands
are separate channels, so the 2.4 GHz client does not steal airtime from the 5 GHz
ones. It carries a worse network term (congestion, single-stream 802.11n), but that
term is **common-mode across schedulers** — the same device runs fifo, strict and
triage4 — so it cannot change the ordering result. It only inflates that zone's
absolute RTT, and since a fixed additive term is relatively larger against
TRIAGE/4's ~200 ms alarms than against FIFO's ~4000 ms, it biases *against* the
claim being made. That is the safe direction.

Two things follow:

1. **Assign the weakest device to zone 5.** In C3 all six alarms originate in zones
   2 and 4, so a zone-5 device contributes **no alarm samples at all** and the C3
   alarm headline is untouched by it. In R3 every zone has alarms, but zones 4–5
   carry the fewest (3 of 30). Zone 5 is also the lowest geographic priority, so its
   messages queue longest (~10 s) and a ~20 ms radio offset is noise against that.
   Decide this before collecting data and state it in the writeup.
2. **Measure the offset, don't assert it.** `analyze.py --per-zone` breaks RTT down
   by the device that recorded it. Read it against the unsaturated baseline pass,
   where the queueing term is absent and what remains *is* the network term.
   *(measure: per-zone idle RTT.)*

**Power-save matters most on the weakest device.** `brcmfmac` enables it by default
and it adds tens of milliseconds of jitter — far more than the band itself does.

**A device pinned to an old Python** (e.g. a Jetson Nano on JetPack's Python 3.6)
cannot run `paho-mqtt` 2.x, which needs ≥ 3.7. Install a modern interpreter beside
the system one rather than downgrading the client code:

```bash
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh && conda create -n t4 python=3.11 && conda activate t4
pip install -r requirements-client.txt
```

Backporting the clients to `paho-mqtt` 1.6.1 is possible but forks the client code
path (different callback signatures) and risks silent divergence from the validated
one. Not worth it.

A Jetson Nano is in fact the **best host for the observer**: it has gigabit
Ethernet, and the observer must be wired, off the Pi, and off any transmitter.

## 4b. Control-centre observer (one wired device)

The consumer of the system: it subscribes to all six return topics and records the
order in which the broker actually delivered messages.

```bash
.venv/bin/python -m clients.observer --host <pi-host>.local --zones 6 --drain 120
```

It is **passive** — deliberately not a latency instrument. RTT stays measured at
the senders, where `t_send` and `t_recv` share one clock; taking `t_recv` here
instead would compare clocks across devices and force NTP/PTP for no gain, since
the scheduler's entire effect is upstream of delivery.

What it buys is independence. Each zone agent sees only its own zone, so the
*interleaved* egress order across zones is otherwise known only from the broker's
own CSV — self-reported. The observer is an unmodified MQTT subscriber recording
the same thing from the wire. Combined with the committed schedule's arrival
times, it yields the direct R1.1 artifact: the count of **routine-over-alarm
inversions** (a routine message that arrived *after* an alarm but was delivered
*before* it). On the loopback rehearsal of C3 this was **48 for Strict and 0 for
TRIAGE/4**. *(measure: your own values.)*

Run it on a **wired** device: not on the Pi (it would spend the broker's CPU) and
not on a zone device (it would perturb that zone's timing).

It heartbeats on `t4ctl/ready` like the agents, so include it in the expected
count — run the campaign with `ZONES=7 ./run_pi.sh`. Set `--drain` to at least the
scenario span plus the time the slowest scheduler needs to empty its queue, since
the observer has no completion signal of its own and simply collects for that long.
For most cells that is span + 30 s (C3 ≈ 18 + 30, R3 ≈ 30 + 30).

**HW-Flood is the exception, because of TBP.** TBP is the one non-work-conserving
scheduler: it holds the 526-message background band (zones 4–5) behind a 5 token/s
bucket and delivers it over ~100 s. It never drops these messages, it only delays
them. A 50 s window cuts that tail off, and the late deliveries then disappear from
every recorder at once — broker CSV, RTT shards, and this trace — because a message
delivered after the window is recorded nowhere. Size HW-Flood at **span + ~100 s ≈
120 s** so TBP finishes; every other scheduler delivers within seconds and the
recorders idle out the remainder. The same 120 s applies to the zone agents'
`--drain` and to the broker window (`DRAIN` in `run_pi.sh`); all three must cover
the tail together. The broker now warns on shutdown if any message is still queued,
so a window that is still too short is caught in the cell's log, not in analysis.

Like the agents, the observer needs the committed `workloads/` JSONs present (the
same `git pull` — the default `--schedules-dir workloads` finds them). It does **not**
replay them: it reads each scenario only to stamp the schedule fingerprint into its
trace, so `consolidate.py` can reject a stale delivery-order trace exactly as it
rejects a stale RTT shard (see §2b). If `workloads/` is missing on the observer host,
it fails at the first cell rather than recording unfingerprinted traces.

Disable WiFi power-save on every wireless device:

```bash
sudo iw dev wlan0 set power_save off
```

Power-save adds tens of milliseconds of jitter and is the most common cause of noisy
RTT on Pi-class WiFi.

## 5. Running the campaign

On the Pi, once all six agents are idling:

```bash
ZONES=7 ./run_pi.sh
```

(`ZONES` is the number of participants the coordinator waits for: 6 agents + the
observer. Drop it to `6` if you run without the observer.)

The matrix is **20 cells × 30 reps = 600 broker launches**:

- six schedulers `{fifo, strict, wfq, drr, tbp, triage4}` on **all three** scenarios
  `{C3, HW-Flood, R3}` — 18 cells;
- the `t4-nosourcelimit` ablation on the two AAP scenarios `{HW-Flood, R3}` only —
  2 cells. It strips TRIAGE/4's per-source layer but keeps the band-global backstop,
  so on HW-Flood it sheds legitimate alarms the per-source layer spares, and on R3
  (the control) it sheds nothing. On C3 it would just duplicate `triage4`, so it does
  not run there.

`run_pi.sh`, `_expected_cells()` in `consolidate.py`, and this matrix are kept in
step; a partial run fails consolidation (§6) until every one of the 20 cells is
present. For each cell the script starts a broker (one scheduler per process), waits
until all six agents are ready, announces the cell on `t4ctl/go`, waits out the replay
plus drain, then stops the broker with **SIGTERM** so it flushes its overhead CSV.

Knobs: `PORT REPS SCHEDULERS SCENARIOS DRAIN ZONES RESULTS RATE_C_C3 RATE_C_HW
RATE_C_R3`. Run the **whole matrix in one invocation** — a campaign split across
several runs is what lost data last time. If you must extend one arm, override
`SCHEDULERS`/`SCENARIOS`, but the run is not consolidatable until every expected cell
exists.

**Adaptive Alarm Protection is enabled for both TRIAGE/4 arms** — `triage4` and
`t4-nosourcelimit` — and the five baselines run with `--no-aap`. The ablation *must*
keep AAP on: `--no-aap` there would strip the backstop too and compare against a
scheduler that never sheds at all, which is not the comparison being made. Record this
alongside any result.

### Choosing the egress rate C

`C` (msg/s) is the saturation knob, independent of the workload. It is set below each
scenario's offered rate so a real queue builds (ρ > 1): the host runs used **C3 C=5**,
**HW-Flood C=23**, and **R3 C=8**. Re-check it on the Pi, because the job of `C` is to
saturate the
*egress*, not the *CPU*. `run_pi.sh` writes `results/cpu.csv` with the broker's
`core_utilisation` per cell for exactly this reason:

- utilisation well below 1.0 → the Pi has headroom, and the overhead figure is
  measuring scheduling cost. *(measure: report the value)*
- utilisation approaching 1.0 → the Pi itself is the bottleneck, and the overhead
  figure is contaminated by CPU contention. **Lower `C` and rerun.**

### Network baseline

Run one unsaturated pass with a very high `C`, so the egress never queues:

```bash
RATE_C_C3=1000 RATE_C_HW=1000 RATE_C_R3=1000 REPS=5 RESULTS=results_baseline ./run_pi.sh
```

This quantifies the fixed network + broker term. *(measure: idle RTT and jitter.)*
Under saturation the queueing term dominates it by orders of magnitude, which is what
makes the scheduler comparison robust to WiFi noise.

## 6. Collecting and analysing

The agents write their RTT shards locally and the observer writes its traces
locally; only the broker overhead and CPU samples are already on the Pi. Pull the
rest in and verify the campaign is complete:

```bash
cp pi_hosts.conf.example pi_hosts.conf && $EDITOR pi_hosts.conf   # once: zone -> ssh target
./collect_results.sh
```

`collect_results.sh` runs **on the Pi in the prototype venv**. It rsyncs every
device's results into `results/` and then checks the campaign against the schedules:
each rep carries the whole workload, every zone in the schedule produced a shard for
every rep, reps are contiguous, and the observer traces exist. It **exits non-zero and
refuses to bless the data** if any of that fails — a device that was unreachable at
collection time leaves a hole that looks exactly like a zone that stayed silent, and
the analysis would average over it without complaint.

Analysis is two tools with different jobs:

**`consolidate.py` — the authoritative pass.** It produces the tidy CSVs the
manuscript reports (`summary.csv`, `comparisons.csv` with Welch's t-test p-values,
`per_zone.csv`, `rtt_long.csv`, `inversions_long.csv`) and enforces both integrity
guards: it verifies the **schedule fingerprint** of every RTT shard *and* every
observer trace against the JSON on disk (§2b), and it checks the campaign fills the
whole **20-cell matrix** for the expected reps. Either failure aborts with a non-zero
exit, so a stale or partial campaign cannot be consolidated unnoticed. It needs the
**main-repo venv** (it imports `assessment.metrics` and SciPy for the Welch test),
which the prototype venv does not have — so run it from `prototype/` with the main venv
and the repo root on `PYTHONPATH`:

```bash
# from prototype/, using the MAIN venv (the one with assessment/ + scipy):
PYTHONPATH=.. ../.venv/bin/python consolidate.py \
    --results-dir results --out-dir results-consolidated --expect-reps 30
```

**`analyze.py` — the per-scenario spot check.** Runs in the prototype venv, joins
broker overhead to client RTT on `(rep, msg_id)`, and prints alarm vs routine RTT
(mean ±95% CI), overhead p50/p99, and drops for one scenario. Use it for a quick look
on the Pi before pulling everything to the analysis host:

```bash
.venv/bin/python analyze.py --results-dir results --scenario c3_multi_zone_emergency --per-zone
.venv/bin/python analyze.py --results-dir results --scenario hw_flood_attack --per-zone
.venv/bin/python analyze.py --results-dir results --scenario r3_legit_extreme_emergency --per-zone
```

**Integrity checks before believing a run** (`consolidate.py` enforces the first two;
the fingerprint guard makes them fail loudly rather than silently):

- 6 zones × 30 reps = **180 RTT shards** per (scheduler, scenario).
- Broker CSV rows per rep = the scenario's message count: **C3 126**, **HW-Flood 630**
  (530 alarms), **R3 330** — so `126 × reps` (C3), `630 × reps` (HW-Flood), `330 ×
  reps` (R3), plus header.
- `dropped` is 0 everywhere except where AAP legitimately sheds: on **HW-Flood**,
  `triage4` sheds the attacker while sparing the five legitimate sources, and
  `t4-nosourcelimit` sheds indiscriminately — that contrast is the point of the cell.
  **R3 + triage4 dropping zero alarms** is the on-hardware confirmation of no
  false-positive shedding (**R1.3**).

## 7. Troubleshooting

**A cell aborts with "only N/6 zone agents ready".** A device is down, off-network, or
its agent crashed. The campaign stops rather than silently producing a cell with a
missing zone — fix the device and restart. This is intentional: a hole in one zone
would bias that cell's RTT distribution.

**Never stop the broker with SIGINT from a script.** A process backgrounded from a
non-interactive shell inherits `SIGINT = SIG_IGN`, and Python honours the inherited
disposition, so `kill -INT` is a no-op and the broker hangs. `run_pi.sh` uses
`kill -TERM`, and the broker installs explicit `SIGINT`/`SIGTERM` handlers.

**Broker will not exit / `run_pi.sh` hangs in `wait`.** The zone agents hold their TCP
connections open across cells. Since Python 3.12, `Server.wait_closed()` blocks until
every connection handler returns, which deadlocks shutdown; the broker therefore calls
`server.close()` without awaiting `wait_closed()`. Preserve this if you modify the
shutdown path.

**Noisy or bimodal RTT.** WiFi power-save (§4), a busy 2.4 GHz channel, or another
client on the SSID. Use a dedicated 5 GHz SSID on a fixed channel.

**Overhead p99 much worse than the host's.** Expected to a degree on a slower core,
but check `cpu.csv` first: if `core_utilisation` is near 1.0 you are measuring
contention, not the scheduler.

## 8. Reproducibility log (fill in per campaign)

| Field | Value |
| --- | --- |
| Pi model / kernel / Python | *(measure)* |
| `triage4` wheel | `1.2.0` (from PyPI, on the Pi) |
| Scenarios | `c3_multi_zone_emergency`, `hw_flood_attack`, `r3_legit_extreme_emergency` |
| Schedules | committed under `workloads/`, base seed 999; git commit *(record)*; fingerprints C3 `e04b977a7019`, HW-Flood `23a941029272`, R3 `1f991c3012d8` |
| Schedulers | `fifo`, `strict`, `wfq`, `drr`, `tbp`, `triage4`; ablation `t4-nosourcelimit` on HW-Flood + R3 |
| AAP | on for `triage4` and `t4-nosourcelimit`; `--no-aap` for the five baselines |
| Matrix | 20 cells × 30 reps = 600 broker launches |
| Egress rate C | C3 = *(measure)*, HW-Flood = *(measure)*, R3 = *(measure)* |
| Reps | 30 |
| Broker core utilisation | *(measure)* |
| Network baseline RTT | *(measure)* |
| Client devices / radios | *(record: which zone, which device, 2.4 or 5 GHz)* |
| Per-zone network offset | *(measure: `analyze.py --per-zone` on the baseline pass)* |

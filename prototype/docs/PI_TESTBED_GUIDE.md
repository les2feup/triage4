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

**Six devices, one per `zone_priority` (0–5).** Both scenarios' messages carry a
`zone_priority` in 0–5; that is what the scheduler bands on and what the return
topic `t4out/<zone>` is keyed by, so each device owns exactly one return topic and
receives only its own traffic. (In R3 the ten *logical* zones fold onto these six
priorities via `zone_id % 6`; the six-device split follows the priorities the
scheduler actually sees.)

## 2. Why no clock synchronisation is needed

Each agent publishes its own messages and subscribes to its own return topic, so
`t_send` and `t_recv` for any message are stamped by the **same process's**
`time.monotonic_ns()`. RTT is therefore a single-clock measurement and **NTP/PTP is
not required** — clocks are never compared across devices. The broker's own clock
is used only for scheduling and never enters an RTT.

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
   .venv/bin/pip install -r requirements.txt     # triage4==1.1.0 + paho-mqtt
   ```

   The Pi installs the **published** `triage4==1.1.0` wheel from PyPI — the same
   artifact the host runs used. Nothing is cross-compiled, and nothing is imported
   from the repo's `src/`, so the code path under measurement is the one a user
   would actually deploy.

## 4. Client setup (each of the six zone devices)

Same clone and venv as above (any Linux/macOS host: laptops, Pi Zeros, spare Pis).
Then on device *z* (one of 0…5):

```bash
.venv/bin/python -m clients.zone_agent --zone <z> --host <pi-host>.local
```

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
.venv/bin/python -m clients.observer --host <pi-host>.local --zones 6 --drain 60
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
scenario span plus the campaign's drain (C3 ≈ 18 s + 30 s, R3 ≈ 30 s + 30 s), since
the observer has no completion signal of its own and simply collects for that long.

Disable WiFi power-save on every wireless device:

```bash
sudo iw dev wlan0 set power_save off
```

Power-save adds tens of milliseconds of jitter and is the most common cause of noisy
RTT on Pi-class WiFi.

## 5. Running the campaign

On the Pi, once all six agents are idling:

```bash
ZONES=7 ./run_pi.sh              # {fifo, strict, triage4} x {C3, R3} x 30 reps
```

(`ZONES` is the number of participants the coordinator waits for: 6 agents + the
observer. Drop it to `6` if you run without the observer.)

For each of the 180 cells the script starts a broker (one scheduler per process),
waits until all six agents are ready, announces the cell on `t4ctl/go`, waits out the
replay plus drain, then stops the broker with **SIGTERM** so it flushes its overhead
CSV.

Knobs: `PORT REPS SCHEDULERS SCENARIOS DRAIN ZONES RESULTS RATE_C_C3 RATE_C_R3`.

**Adaptive Alarm Protection is enabled only for `triage4`**; the baselines run with
`--no-aap`. Record this alongside any result.

### Choosing the egress rate C

`C` (msg/s) is the saturation knob, independent of the workload. It is set below each
scenario's offered rate so a real queue builds (ρ > 1): the host runs used **C3 C=5**
and **R3 C=8**. Re-check it on the Pi, because the job of `C` is to saturate the
*egress*, not the *CPU*. `run_pi.sh` writes `results/cpu.csv` with the broker's
`core_utilisation` per cell for exactly this reason:

- utilisation well below 1.0 → the Pi has headroom, and the overhead figure is
  measuring scheduling cost. *(measure: report the value)*
- utilisation approaching 1.0 → the Pi itself is the bottleneck, and the overhead
  figure is contaminated by CPU contention. **Lower `C` and rerun.**

### Network baseline

Run one unsaturated pass with a very high `C`, so the egress never queues:

```bash
RATE_C_C3=1000 RATE_C_R3=1000 REPS=5 RESULTS=results_baseline ./run_pi.sh
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

`collect_results.sh` rsyncs every device's results into `results/` and then checks
the campaign against the schedules: each rep carries the whole workload, every zone
in the schedule produced a shard for every rep, reps are contiguous, and the
observer traces exist. It **exits non-zero and refuses to bless the data** if any
of that fails — a device that was unreachable at collection time leaves a hole that
looks exactly like a zone that stayed silent, and `analyze.py` would average over
it without complaint.

Then analyse:

```bash
.venv/bin/python analyze.py --results-dir results --scenario c3_multi_zone_emergency --per-zone
.venv/bin/python analyze.py --results-dir results --scenario r3_legit_extreme_emergency --per-zone
```

`analyze.py` joins broker overhead to client RTT on `(rep, msg_id)` and reports alarm
vs routine RTT (mean ±95% CI), overhead p50/p99, and drops.

**Integrity checks before believing a run:**

- 6 zones × 30 reps = **180 RTT shards** per (scheduler, scenario).
- Broker CSV rows = `126 × reps` (C3) and `330 × reps` (R3), plus header.
- `dropped` is 0 everywhere except where AAP legitimately sheds. R3 + triage4
  dropping **zero** alarms is the on-hardware confirmation of no false-positive
  shedding (**R1.3**).

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
| Scenarios | `c3_multi_zone_emergency`, `r3_legit_extreme_emergency` |
| Schedules | pre-generated, base seed 999, committed under `workloads/` |
| Schedulers | `fifo`, `strict`, `triage4` |
| AAP | on for `triage4` only |
| Egress rate C | C3 = *(measure)*, R3 = *(measure)* |
| Reps | 30 |
| Broker core utilisation | *(measure)* |
| Network baseline RTT | *(measure)* |
| Client devices / radios | *(record: which zone, which device, 2.4 or 5 GHz)* |
| Per-zone network offset | *(measure: `analyze.py --per-zone` on the baseline pass)* |

# Stage 3b — Hardware Results (Raspberry Pi MQTT testbed)

Answers reviewers **R1.1** (validation on a real broker, not only simulation), **R2.1**
(deployment overhead on IoT gateway hardware), and confirms **R1.3** (AAP does not shed
legitimate alarms) on physical hardware.

Campaign: 2026-07-14. `{fifo, strict, triage4} × {C3, R3} × 30 reps` = 180 cells.
Zero message loss and zero unexpected drops in every cell.

---

## 1. Testbed

| Role | Device | Link |
| --- | --- | --- |
| Broker | Raspberry Pi 5 (`rpi5-b-hailo`), Python 3.13 | **Ethernet** to the AP |
| Zones 0–4 | client devices, one per geographic priority | WiFi 5 GHz |
| Zone 5 | Raspberry Pi Zero W | WiFi **2.4 GHz** (radio has no 5 GHz) |
| Control centre | Windows host, passive observer | Ethernet |

The Pi is **not** the access point. Running `hostapd` plus the 802.11 softirq path on the
same CPU would have contaminated the R2.1 overhead measurement, which is the headline
number; wiring the Pi keeps one wireless hop between client and broker while leaving the
broker's CPU doing exactly one job.

**Six devices, one per `zone_priority` (0–5)** — the value the scheduler bands on and the
key of the return topic `t4out/<zone>`. Each device therefore owns exactly one return
topic and receives only its own traffic.

**Latency is single-clock RTT.** Each agent publishes its own messages and subscribes to
its own return topic, so `t_send` and `t_recv` are stamped by the same process's
`monotonic_ns()`. No NTP/PTP is required and no clock is ever compared across devices.

**Saturation** is created by the broker's egress rate `C` (C3: 5 msg/s, R3: 8 msg/s), set
below the offered rate so ρ > 1 and a real queue builds. `C` is independent of
`TRIAGE4Config.service_rate`, which the online dispatcher ignores.

**AAP is enabled for `triage4` only**; the baselines run with `--no-aap`.

The published `triage4==1.1.0` wheel from PyPI is what runs on the Pi — the same artifact
a deployer would install. Nothing is cross-compiled and nothing is imported from `src/`.

---

## 2. R2.1 — Scheduling overhead on the Pi

Per-message cost of `enqueue` + `select_next`, timed with `perf_counter_ns()` inside the
broker.

| Scenario | Scheduler | p50 (µs) | p99 (µs) |
| --- | --- | ---: | ---: |
| C3 | fifo | 1.31 | 4.86 |
| C3 | strict | 4.65 | 10.12 |
| C3 | **triage4** | **14.94** | **43.88** |
| R3 | fifo | 1.31 | 2.11 |
| R3 | strict | 4.48 | 7.78 |
| R3 | **triage4** | **13.68** | **26.46** |

**Broker core utilisation: 0.001** (0.1% of one core), measured per cell from
`/proc/<pid>/stat`. The Pi is nowhere near being the bottleneck, so these timings measure
scheduling cost rather than CPU contention — the assumption R2.1 depends on, verified
rather than asserted.

TRIAGE/4 costs roughly **10× FIFO and 3× Strict**, and in absolute terms **~14 µs per
message**. Against a 200 ms service quantum (C = 5 msg/s) that is 0.007% of the egress
budget.

> **Note:** overhead measured on the Pi is *lower* than the earlier desktop-host figures
> (triage4 p50 17–21 µs). This is not an anomaly. On the host, the broker, six clients and
> the observer shared one machine and preempted each other inside the timed section; on the
> Pi the broker has the box to itself. The hardware measurement is the cleaner of the two.

---

## 3. R1.1 — Alarm latency on a real broker

RTT in ms, mean ± 95% CI, 30 reps.

| Scenario | Scheduler | Alarm RTT | Routine RTT | Dropped |
| --- | --- | ---: | ---: | ---: |
| C3 | fifo | 3869.01 ± 174.76 | 3508.86 ± 56.48 | 0 |
| C3 | strict | 2931.77 ± 399.68 | 3546.85 ± 173.06 | 0 |
| C3 | **triage4** | **168.59 ± 14.32** | 3687.47 ± 146.34 | 0 |
| R3 | fifo | 4457.91 ± 203.59 | 6309.85 ± 65.07 | 0 |
| R3 | strict | 4800.03 ± 563.95 | 6247.04 ± 194.30 | 0 |
| R3 | **triage4** | **587.93 ± 22.04** | 7122.67 ± 195.99 | 0 |

TRIAGE/4 cuts alarm RTT by **17–23×** versus both baselines, with non-overlapping CIs.
Routine RTT is modestly worse (3.5 s → 3.7 s; 6.3 s → 7.1 s) — the intended trade: latency
is moved from alarms to routine telemetry under saturation.

The aggregates reproduce the N=30 loopback campaign (fifo 4212/4426, strict 2985/4797,
triage4 216/564 ms) to within a few percent, on entirely different hardware and over real
WiFi.

### 3.1 The per-zone table is the result

RTT by the zone device that recorded it (ms, 30 reps). Band mapping:
**zones 0–1 → HIGH, 2–3 → STANDARD, 4–5 → BACKGROUND**; alarms from any zone → ALARM.

**R3 — alarm RTT by zone**

| Zone | fifo | strict | **triage4** |
| ---: | ---: | ---: | ---: |
| 0 | 4822.53 | 145.32 | 658.46 |
| 1 | 4159.54 | 442.51 | 533.12 |
| 2 | 4346.36 | 1124.96 | 647.71 |
| 3 | 4501.88 | 2397.71 | 762.89 |
| 4 | 4303.50 | 12984.85 | 284.08 |
| 5 | 4614.93 | **26794.47** | **390.87** |

This is the paper's thesis in one column. Under Strict, **a critical alarm from the
lowest-priority zone takes 26.8 seconds** — a 185× spread across zones, because geographic
priority is allowed to outrank semantic urgency. Under TRIAGE/4, alarm latency is
**essentially flat in geography** (284–763 ms): **68× faster** for the worst-case zone.

Meanwhile TRIAGE/4's *routine* traffic still respects geography — R3 routine RTT is
220/316 ms (HIGH), 946/1059 ms (STANDARD), 19923/20272 ms (BACKGROUND). The four plateaus
emerge from RTTs measured independently on six devices; nothing in the measurement path
knows the band mapping. Both halves of the design — semantic override for alarms,
geographic ordering for telemetry — are visible in a single table.

Strict instead produces a clean six-level geographic gradient (94 → 168 → 289 → 986 →
13157 → 22788 ms routine), and FIFO produces no gradient at all. Three schedulers, three
qualitatively distinct signatures, each the one its algorithm predicts.

---

## 4. R1.1 — Priority inversions observed on the wire

A **priority inversion** is a routine message that *arrived after* an alarm yet was
*delivered before* it. The delivery order comes from `clients/observer.py`, an unmodified
MQTT subscriber on a separate host; arrival times come from the committed schedule. **The
broker is not asked to vouch for itself.**

Per rep, mean ± 95% CI, 30 reps.

| Scenario | Scheduler | Inversions / rep | Worst alarm overtaken by |
| --- | --- | ---: | ---: |
| C3 | fifo | 0.0 ± 0.0 | 0 msgs |
| C3 | strict | 48.0 ± 0.0 | 24 msgs |
| C3 | **triage4** | **0.0 ± 0.0** | **0 msgs** |
| R3 | fifo | 4.3 ± 0.7 | 2 msgs |
| R3 | strict | 823.9 ± 1.8 | **249 msgs** |
| R3 | **triage4** | **0.0 ± 0.0** | **0 msgs** |

Under Strict in R3, a single alarm was overtaken by **249 routine messages** out of 330.
TRIAGE/4 records **zero inversions in all 60 reps of both scenarios** — a categorical
result, not a statistical one.

### 4.1 Reading FIFO correctly (important)

**FIFO's ~0 inversions is not evidence that FIFO is good, and this must not be presented as
a ranking.** FIFO delivers in arrival order, so it is *structurally incapable* of inverting:
nothing that arrived later can be delivered earlier. Its R3 value of 4.3 measures the
**network's own reordering** — six devices publishing concurrently, so a message scheduled
at t=10.00 can reach the broker after one scheduled at t=10.01. That figure is a useful
**noise floor**, not a scheduling property.

The two metrics are therefore complementary and must be read together:

- **Inversions** show that Strict *actively inverts* priority; TRIAGE/4 never does.
- **Alarm RTT** shows that FIFO, while it never inverts, is still hopeless (4458 ms),
  because *not inverting is not the same as prioritising*.

TRIAGE/4 is the only scheduler that does both. Note also that TRIAGE/4 reaches 0.0 despite
the network demonstrably reordering publishes (FIFO's 4.3 proves it occurs): the semantic
override corrects network-induced reordering as well as scheduler-induced.

---

## 5. R1.3 — AAP does not shed legitimate alarms

R3 (`legit_extreme_emergency`) is a genuine mass-casualty surge, not an attack. With AAP
enabled, TRIAGE/4 dropped **0 alarms across all 30 reps** on hardware — no false-positive
shedding, confirming on real hardware what the simulation showed.

---

## 6. The 2.4 GHz client is not a confound

Zone 5 is a Pi Zero W whose radio has no 5 GHz. R3 supplies a **matched control**: zones 4
and 5 have identical workload shape (53 messages, 3 alarms) and differ *only* in radio.
Under FIFO, which treats every zone alike:

- zone 4 (5 GHz): **6357.72 ± 159.61 ms**
- zone 5 (2.4 GHz): **6392.57 ± 158.97 ms**

A 35 ms difference against a ±160 ms CI — **statistically indistinguishable from zero**. In
C3, zone 5 (4093.81 ms) is not even the slowest zone under FIFO (zone 0: 4413.42 ms).

Independently of the measurement, the offset could not have changed the conclusion: it is
common-mode (the same device runs all three schedulers) and additive, so it inflates
TRIAGE/4's ~200 ms alarms *relatively more* than FIFO's ~4000 ms — it biases **against** the
claim being made. Zone 5 was chosen for the weakest device before data collection precisely
because C3 sources all its alarms from zones 2 and 4, so the device contributes no C3 alarm
samples at all.

---

## 7. Limitations

- **Absolute RTT is not comparable to the DES.** RTT adds fixed network and broker terms
  the simulation does not model. The claim is the *relative ordering across schedulers*,
  which reproduces.
- **The hardware set is a trio** (FIFO, Strict, TRIAGE/4), framed as a feasibility
  confirmation. The WFQ/DRR/TBP breadth comparison remains simulation-only (§IV-C).
- **Routine RTT degrades** under TRIAGE/4 (+5% C3, +13% R3). This is the design's intended
  trade under saturation, not an incidental cost.
- **An unsaturated network baseline pass has not yet been run** (`RATE_C_*=1000`), so the
  fixed network term is bounded by the FIFO control above rather than measured directly.

---

## 8. Reproducing

```bash
# on each zone device (0-5)
.venv/bin/python -m clients.zone_agent --zone <z> --host <pi-ip>
# on the control centre
.venv/bin/python -m clients.observer --host <pi-ip> --zones 6 --drain 60
# on the Pi
ZONES=7 ./run_pi.sh                    # 3 schedulers x 2 scenarios x 30 reps
./collect_results.sh                   # pull shards + verify completeness
.venv/bin/python analyze.py --results-dir results --scenario <scenario> --per-zone
.venv/bin/python inversions.py --results-dir results --scenario <scenario>
```

| Field | Value |
| --- | --- |
| Broker | Raspberry Pi 5, Python 3.13, `triage4==1.1.0` from PyPI |
| Scenarios | `c3_multi_zone_emergency` (126 msg, 6 alarms), `r3_legit_extreme_emergency` (330 msg, 30 alarms) |
| Schedules | pre-generated, base seed 999, committed under `prototype/workloads/` |
| Egress rate C | C3 = 5 msg/s, R3 = 8 msg/s |
| Reps | 30 |
| AAP | enabled for `triage4` only |
| Broker core utilisation | 0.001 |
| Drops | 0 in all 180 cells |

---

## 9. Headline

> On a Raspberry Pi 5 running a real MQTT 5.0 broker under saturation, TRIAGE/4 reduces
> alarm round-trip latency from 4.5 s (FIFO) and 4.8 s (strict geographic priority) to
> **588 ms** — and for the lowest-priority zone, from **26.8 s to 391 ms (68×)** — while an
> independent MQTT subscriber observes **zero priority inversions in 60 of 60 reps**, versus
> 824 per rep under strict priority. The cost is **~14 µs of scheduling per message** at
> 0.1% of one core, with no alarms shed during a legitimate mass-casualty surge.

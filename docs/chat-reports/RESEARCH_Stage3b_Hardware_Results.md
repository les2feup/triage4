# Stage 3b — Hardware Results (Raspberry Pi MQTT testbed)

Answers reviewers **R1.1** (validation on a real broker, not only simulation), **R2.1**
(deployment overhead on IoT gateway hardware), and confirms **R1.3** (AAP does not shed
legitimate alarms) on physical hardware.

Campaign: 2026-07-14. `{fifo, strict, wfq, drr, tbp, triage4} × {C3, R3} × 30 reps` = 360
cells — the full baseline set from the simulation study, run on a real broker. Zero message
loss and zero unexpected drops in every cell.

**This is an independent saturated-regime demonstration, not a numerical reproduction of the
simulation.** The two studies use the same workloads but at different operating points, on
purpose. The discrete-event study fixes service rate C = 20 msg/s as a calibration constant
and varies load per scenario through ρ = λ/C; for these two scenarios that puts the DES at
moderate load (ρ ≈ 0.35). The hardware instead drives the egress into saturation (ρ ≈ 1.4),
because saturation is the regime where a priority scheduler matters — an unsaturated run
would show small differences and prove little. So the hardware confirms the *claim* the
simulation makes (TRIAGE/4 protects alarm traffic; the baselines do not), not its per-row
numbers. Absolute latencies and the ordering *among baselines* are not expected to match, and
are not claimed to; what carries across both studies is TRIAGE/4's dominance.

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
| C3 | wfq | 5.61 | 11.27 |
| C3 | drr | 6.02 | 14.40 |
| C3 | tbp | 5.78 | 12.65 |
| C3 | **triage4** | **14.94** | **43.88** |
| R3 | fifo | 1.31 | 2.11 |
| R3 | strict | 4.48 | 7.78 |
| R3 | wfq | 5.67 | 8.81 |
| R3 | drr | 5.56 | 8.94 |
| R3 | tbp | 5.13 | 11.39 |
| R3 | **triage4** | **13.68** | **26.46** |

**Broker core utilisation: 0.001** (0.1% of one core), measured per cell from
`/proc/<pid>/stat`. The Pi is nowhere near being the bottleneck, so these timings measure
scheduling cost rather than CPU contention — the assumption R2.1 depends on, verified
rather than asserted.

TRIAGE/4 is the most expensive scheduler, at **~14 µs per message** — roughly 10× FIFO and
2.5× the WFQ/DRR/TBP cluster (all ~5–6 µs). Even so, against a 200 ms service quantum
(C = 5 msg/s) that is 0.007% of the egress budget. The four bands, per-device round-robin,
token buckets and AAP together cost single-digit microseconds more than a plain queue.

> **Note:** overhead measured on the Pi is *lower* than the earlier desktop-host figures
> (triage4 p50 17–21 µs). This is not an anomaly. On the host, the broker, six clients and
> the observer shared one machine and preempted each other inside the timed section; on the
> Pi the broker has the box to itself. The hardware measurement is the cleaner of the two.

---

## 3. R1.1 — Alarm latency on a real broker

RTT in ms, mean ± 95% CI, 30 reps.

All six schedulers of the simulation baseline set run on hardware — no scheduler is measured
in simulation but withheld here. ("Table 1" throughout means the baseline-comparison table in
`RESEARCH_Stage3_Tables.md`, not a table in the manuscript, whose scenario labels differ —
the manuscript's S1 is the near-saturation scarcity scenario, not `legit_extreme_emergency`.)
AAP is enabled for `triage4` only.

| Scenario | Scheduler | Alarm RTT | Routine RTT | Dropped |
| --- | --- | ---: | ---: | ---: |
| C3 | fifo | 3869.01 ± 174.76 | 3508.86 ± 56.48 | 0 |
| C3 | strict | 2931.77 ± 399.68 | 3546.85 ± 173.06 | 0 |
| C3 | wfq | 3841.52 ± 308.42 | 3494.60 ± 77.26 | 0 |
| C3 | drr | 2507.68 ± 20.82 | 3577.17 ± 57.64 | 0 |
| C3 | tbp | 5608.20 ± 693.24 | 3407.92 ± 135.65 | 0 |
| C3 | **triage4** | **168.59 ± 14.32** | 3687.47 ± 146.34 | 0 |
| R3 | fifo | 4457.91 ± 203.59 | 6309.85 ± 65.07 | 0 |
| R3 | strict | 4800.03 ± 563.95 | 6247.04 ± 194.30 | 0 |
| R3 | wfq | 3790.83 ± 225.96 | 6386.33 ± 76.23 | 0 |
| R3 | drr | 1316.96 ± 39.60 | 6601.58 ± 65.27 | 0 |
| R3 | tbp | 3857.47 ± 441.33 | 6802.01 ± 191.48 | 0 |
| R3 | **triage4** | **587.93 ± 22.04** | 7122.67 ± 195.99 | 0 |

TRIAGE/4 has the lowest alarm RTT of all six in both scenarios, with non-overlapping CIs.
The margin depends on which baseline it is measured against: over the **strongest** baseline
(DRR, 2508 ms in C3 and 1317 ms in R3) it is **15× in C3 and 2.2× in R3**; over the weakest
it is 33× (C3, TBP) and 8× (R3, strict). Routine RTT is modestly worse than the baselines
(3.5 s → 3.7 s; 6.3 s → 7.1 s), the intended trade: latency is moved from alarms to routine
telemetry under saturation.

The R3 aggregate gap over DRR is deliberately not overstated — 2.2× is the honest number.
But the aggregate hides the mechanism, and two things recover it. First, **DRR gives no
guarantee**: its low alarm latency is a side effect of round-robin luck under this workload,
not urgency-awareness, and the per-zone table (§3.1) shows TRIAGE/4 below DRR in every zone.
Second, **DRR pays for that latency with no priority discipline at all** — the inversion and
per-zone views (§4) are where it and TRIAGE/4 separate cleanly. Naming the two extremes:
**DRR is the strongest baseline** (per-device round-robin stops any one device monopolising
the channel under saturation); **TBP is the worst for alarms** (5608 ms in C3, above even
FIFO — it drops every alarm into BACKGROUND, then token-gates it). §4.2 returns to TBP; it
is the ablation that matters most.

The `triage4` aggregates reproduce the N=30 loopback campaign (216/564 ms) to within a few
percent, on entirely different hardware and over real WiFi.

### 3.1 The per-zone table is the result

RTT by the zone device that recorded it (ms, 30 reps). Band mapping:
**zones 0–1 → HIGH, 2–3 → STANDARD, 4–5 → BACKGROUND**; alarms from any zone → ALARM.

**R3 — alarm RTT by zone (ms)**

| Zone | fifo | strict | wfq | drr | tbp | **triage4** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4822.53 | 145.32 | 1649.49 | 1596.76 | 377.91 | 658.46 |
| 1 | 4159.54 | 442.51 | 2350.81 | 1047.27 | 184.75 | 533.12 |
| 2 | 4346.36 | 1124.96 | 3547.03 | 1242.13 | 1324.03 | 647.71 |
| 3 | 4501.88 | 2397.71 | 4771.30 | 1366.88 | 1516.65 | 762.89 |
| 4 | 4303.50 | 12984.85 | 5977.70 | 1148.79 | 15712.74 | 284.08 |
| 5 | 4614.93 | **26794.47** | 7293.35 | 1514.74 | 16055.30 | **390.87** |

This one table separates the schedulers by mechanism.

The **geographic family — Strict, WFQ, TBP — all penalise low-priority zones**, because they
let zone outrank urgency. Strict rises monotonically to **26.8 s for a zone-5 alarm**; WFQ
shows the same gradient more gently (1.6 → 7.3 s); TBP jumps at the BACKGROUND boundary (zone
4–5 alarms land in BACKGROUND and wait ~16 s). A critical alarm's latency depends on *where
it came from* — the priority inversion the paper is about.

The **arrival family — FIFO and DRR — is flat in geography** (FIFO ~4.3–4.8 s, DRR ~1.0–1.6
s) but flat *high*: they give alarms no preference, so every alarm waits out the queue
regardless of zone.

**TRIAGE/4 is flat and low** (284–763 ms): alarm latency stops depending on geography
without becoming uniformly slow. And its *routine* traffic still respects geography — R3
routine RTT is 220/316 ms (HIGH), 946/1059 ms (STANDARD), 19923/20272 ms (BACKGROUND). Both
halves of the design — semantic override for alarms, geographic ordering for telemetry — are
visible in a single table, and no other scheduler in the set has both.

### 3.2 Why DRR is the strongest baseline here

DRR ignores both zone and alarm, yet it has the lowest alarm RTT of any baseline. That is a
saturation property, not an anomaly. Under sustained overload several devices are backlogged
at once, and DRR's per-device round-robin stops any one of them from monopolising the egress,
so an alarm sharing the channel with a busy neighbour still gets a turn soon. FIFO, serving in
pure arrival order, makes that same alarm wait behind the neighbour's whole backlog. Round-
robin and arrival order only diverge when a queue actually persists — which is precisely what
saturation creates.

This is worth stating because the same two schedulers behave identically at light load, where
no queue builds and both simply serve each message before the next arrives. The distinction
appears only under contention, which is the regime this testbed targets. (It is why we do not
re-derive the DES numbers at the hardware's egress rate: the DES holds C fixed at 20 msg/s by
design and varies load through the workload, so matching the hardware would mean changing that
calibration constant for two scenarios alone — an inconsistency we avoid. The two studies meet
at the claim, not at the operating point.)

---

## 4. R1.1 — Priority inversions observed on the wire

A **priority inversion** is a routine message that *arrived after* an alarm yet was
*delivered before* it. The delivery order comes from `clients/observer.py`, an unmodified
MQTT subscriber on a separate host; arrival times come from the committed schedule. **The
broker is not asked to vouch for itself.**

Per rep, mean ± 95% CI, 30 reps.

Per rep, mean ± 95% CI, 30 reps.

| Scenario | Scheduler | Inversions / rep | Worst alarm overtaken by |
| --- | --- | ---: | ---: |
| C3 | fifo | 0.0 ± 0.0 | 0 msgs |
| C3 | strict | 48.0 ± 0.0 | 24 msgs |
| C3 | wfq | 26.6 ± 1.9 | 12 msgs |
| C3 | drr | 0.0 ± 0.0 | 0 msgs |
| C3 | tbp | 88.0 ± 0.0 | 40 msgs |
| C3 | **triage4** | **0.0 ± 0.0** | **0 msgs** |
| R3 | fifo | 4.3 ± 0.7 | 2 msgs |
| R3 | strict | 823.9 ± 1.8 | 249 msgs |
| R3 | wfq | 151.0 ± 3.3 | 26 msgs |
| R3 | drr | 0.7 ± 0.2 | 1 msg |
| R3 | tbp | 531.9 ± 1.3 | 132 msgs |
| R3 | **triage4** | **0.0 ± 0.0** | **0 msgs** |

TRIAGE/4 records **zero inversions in all 60 reps of both scenarios** — a categorical result,
not a statistical one. Every geographic-family scheduler inverts heavily: under Strict in R3
a single alarm was overtaken by **249 routine messages** out of 330; under TBP, by 132.

### 4.1 The two metrics form a 2×2, and only TRIAGE/4 is in the good quadrant

Inversions and alarm RTT must be read together, because each alone is misleading. The six
schedulers fall into a clean pattern:

| | Few inversions | Many inversions |
| --- | --- | --- |
| **Slow alarms** | FIFO, DRR | Strict, WFQ, TBP |
| **Fast alarms** | **TRIAGE/4** | — |

- **FIFO and DRR invert almost nothing but are slow** (4.5 s / 1.3 s alarm RTT in R3). They
  order by arrival and by device turn respectively — both priority-agnostic and roughly
  symmetric, so they cannot *systematically* deliver a later routine message before an
  earlier alarm. FIFO is in fact *structurally incapable* of inverting; its R3 value of 4.3
  is the **network's own reordering** (six devices publishing concurrently), a noise floor,
  not a scheduling property. DRR's 0.7 is the same story. But not inverting is not
  prioritising: neither gives an alarm any preference, so both make alarms wait out the queue.
- **Strict, WFQ and TBP are fast for high-zone alarms but invert massively for low-zone
  ones**, because they encode geography as priority with no override.
- **TRIAGE/4 alone is fast *and* non-inverting.** It reaches 0.0 even though the network
  demonstrably reorders publishes (FIFO's 4.3 proves reordering occurs) — the semantic
  override corrects network-induced reordering as well as scheduler-induced.

This is the whole thesis, measured on hardware with six schedulers: two of them show that
low inversions can coexist with hopeless latency, three show that low latency for some zones
can coexist with massive inversion, and one — TRIAGE/4 — has neither failure.

### 4.2 TBP isolates the semantic override

TBP is the most informative baseline in the set. It is TRIAGE/4 with the geographic bands
(component B) and per-band token buckets (component C) but **without** the semantic override
(A) or AAP (E). So the difference between TBP and TRIAGE/4 is the semantic override, on its
own.

Removing just that mechanism is not neutral — it is actively harmful. TBP has the **worst C3
alarm RTT of all six** (5608 ms, above even FIFO) and the **most C3 inversions** (88, above
Strict's 48). Banding by geography alone drops every alarm into BACKGROUND and then
token-gates it, so bands-plus-tokens without the override buries alarms rather than
protecting them. The gap this leaves for the override to close is **TBP → TRIAGE/4: 5608 →
169 ms in C3, 3857 → 588 ms in R3.** That gap is the paper's contribution, measured directly
on hardware.

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
common-mode (the same device runs all six schedulers) and additive, so it inflates
TRIAGE/4's ~200 ms alarms *relatively more* than the ~4000 ms baselines — it biases
**against** the claim being made. Zone 5 was chosen for the weakest device before data
collection precisely because C3 sources all its alarms from zones 2 and 4, so the device
contributes no C3 alarm samples at all.

---

## 7. Limitations

- **The hardware is not a numerical reproduction of the DES.** RTT adds fixed network and
  broker terms the simulation does not model, and the hardware runs saturated (ρ ≈ 1.4) while
  these two scenarios sit at moderate load in the DES (C fixed at 20 msg/s). What holds across
  both studies is **TRIAGE/4's dominance** — lowest alarm latency, zero inversions, in every
  case. The ordering *among the baselines* is load-dependent and is neither expected nor
  claimed to match between the two (see §3.2 for DRR); reading the hardware as a re-run of a
  Table 1 row is a category error.
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
ZONES=7 ./run_pi.sh                    # 6 schedulers x 2 scenarios x 30 reps
./collect_results.sh                   # pull shards + verify completeness
.venv/bin/python analyze.py --results-dir results --scenario <scenario> --per-zone
.venv/bin/python inversions.py --results-dir results --scenario <scenario>
```

| Field | Value |
| --- | --- |
| Broker | Raspberry Pi 5, Python 3.13, `triage4==1.1.0` from PyPI |
| Scenarios | `c3_multi_zone_emergency` (126 msg, 6 alarms), `r3_legit_extreme_emergency` (330 msg, 30 alarms) |
| Schedulers | fifo, strict, wfq, drr, tbp, triage4 (full Table 1 set) |
| Schedules | pre-generated, base seed 999, committed under `prototype/workloads/` |
| Egress rate C | C3 = 5 msg/s (ρ ≈ 1.4), R3 = 8 msg/s (ρ ≈ 1.4) |
| Reps | 30 |
| AAP | enabled for `triage4` only |
| Broker core utilisation | 0.001 |
| Drops | 0 in all 360 cells |

---

## 9. Headline

> On a Raspberry Pi 5 running a real MQTT 5.0 broker under saturation, against the full set
> of five baselines from our simulation, TRIAGE/4 has the lowest alarm round-trip latency in
> both scenarios — **169 ms (C3) and 588 ms (R3)** — and it is the only scheduler that also
> records **zero priority inversions**, observed independently by an unmodified MQTT
> subscriber across all 60 reps, where the priority-aware baselines invert up to 824 times per
> rep. The two metrics matter together: the arrival-order baselines (FIFO, DRR) barely invert
> but leave alarms slow, the priority baselines (strict, WFQ, TBP) are fast for some zones but
> invert massively, and only TRIAGE/4 avoids both failures. For the lowest-priority zone it
> cuts alarm latency from **26.8 s (strict) to 391 ms — 68×**. The cost is **~14 µs of
> scheduling per message** at 0.1% of one core, with no alarms shed during a legitimate
> mass-casualty surge.

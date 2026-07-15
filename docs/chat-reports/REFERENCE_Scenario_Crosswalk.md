# Scenario Label Crosswalk (authoritative)

One place to resolve the three naming systems that drifted apart during the
project, so writing does not mix them. **The manuscript (`writing/manuscript/access.tex`)
is authoritative.** Internal reports and display labels conform to it; the code
keys never change (they are load-bearing in the CLI, tests, and benchmark
registry — see the CLAUDE.md scenario rule).

Three systems exist:

1. **Manuscript ID** — `C1–C3`, `R1–R3`, `S1–S3`, `V1–V3`. What the paper cites.
2. **Code key** — the snake_case generator name in `assessment/workloads/scenarios.py`.
   Immutable.
3. **Legacy internal label** — the display names used in older reports (notably
   `RESEARCH_Stage3_Tables.md`), which predate the manuscript's final scheme and
   diverge from it.

## Fixed-scenario suite (manuscript authoritative)

Signatures are the generator output at `seed=999` and were used to verify each
row against the manuscript scenario tables (`tab:core-scenarios`,
`tab:aap-scenarios`) and results text — the mapping is checked, not assumed.

| Manuscript ID | Manuscript name | Code key | Signature (dur / msg / alarm) | Legacy label in `RESEARCH_Stage3_Tables.md` |
| --- | --- | --- | --- | --- |
| **C1** | Burst Inversion | `alarm_under_burst` | 5 s / 81 / 1 | C1 — Alarm Under Burst |
| **C2** | Device Monopolization | `device_monopolization` | 60 s / 528 / 0 | C2 — Device Monopolization |
| **C3** | Multi-Zone Emergency | `multi_zone_emergency` | 18 s / 126 / 6 | C3 — Multi-Zone Emergency |
| **R1** | Alarm Flood Attack | `alarm_flood_attack` | 10 s / 250 / 200 | R1 — Alarm Flood Attack |
| **R2** | Malfunction Surge | `alarm_malfunction_surge` | 20 s / — / 394 | R2 — Alarm Malfunction Surge |
| **R3** | Legitimate Extreme Emergency | `legit_extreme_emergency` | 30 s / 330 / 30 | **S1 — Legit Extreme Emergency** ⚠ |
| **S1** | Near-Saturation Scarcity | `alarm_load_near_saturation_constrained` | 30 s / 564 / 28 (ρ≈0.95) | **S2 — Near-Saturation Constrained** ⚠ |

The two ⚠ rows are the drift that caused the confusion. They are confirmed by
the numbers, not by name similarity:

- Internal **S1** (Legit Extreme) → manuscript **R3**: both report a **0 % alarm
  drop rate** and describe a legitimate extreme emergency used as the
  false-positive check. (Manuscript §results: "R3 … TRIAGE/4 drops no alarms".)
- Internal **S2** (Near-Saturation) → manuscript **S1**: both are the
  token-scarcity scenario where component C binds; the ablation drops BACKGROUND
  latency **7.506 s → 1.186 s** in both. (Manuscript §ablation: "component C is
  visible only in S1".)

## Supplemental scenarios (stress sweeps / variants)

Not part of the fixed suite; used for sweeps and sensitivity. Manuscript IDs:

| Manuscript ID | Manuscript name | Code key |
| --- | --- | --- |
| V1–V3 | phase-based variants (60 s) | `*_phased` (e.g. `alarm_under_burst_phased`) |
| S2 | Device-Count Scalability (10–500 devices) | stress sweep (`stress_benchmark`) |
| S3 | Alarm-Rate Surge (1 %–50 % alarm fraction) | `alarm_rate_sweep_*` |

**Unresolved — need author decision.** These legacy rows in
`RESEARCH_Stage3_Tables.md` have no clean manuscript target and are left labelled
as-is until confirmed:

- `C1b — Alarm Under Burst (Phased)` (`alarm_under_burst_phased`) — a phase-based
  variant; maps to the manuscript's V-series, but which of V1–V3 is unstated.
- `C2b — Device Monopolization Sweep` (`device_monopolization_sweep`) — a rate
  sweep within monopolization; not the same as the manuscript's S2 device-count
  sweep.
- `C3b — Multi-Zone Emergency Cascade` (`multi_zone_emergency_cascade`) — no
  manuscript ID at all; appears to be internal-only.

## Hardware prototype (Stage 3b)

`RESEARCH_Stage3b_Hardware_Results.md` and `prototype/` use **C3**
(`multi_zone_emergency`) and **R3** (`legit_extreme_emergency`). Both already
match the manuscript — no relabelling needed. The prototype's file names and
schedules use the code keys, which are stable.

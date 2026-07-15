> **Correction (2026-06-25):** T4 values for R1 and R2 in Tables 1 and 2 were regenerated after fixing a one-line bug in `adaptive_token_bucket.py` (commit 84f1dc1). The `activate()` method incorrectly initialized the alarm bucket to `burst_capacity` (30) instead of `budget` (15) on first activation, giving 15 extra admission tokens to the flood in the first period. The correct behavior starts the bucket at `budget` (one refill period worth of tokens), since the bucket has no accumulated credit when it first activates. Original (buggy) values: R1 T4 latency = 240.17 ± 34.31 ms, drop rate = 10.00%; R2 T4 latency = 187.81 ± 25.71 ms, drop rate = 16.68%. Ablation variants T4-FIFOInBand and T4-NoTokens (which also use AAP) carry the same correction. T4-NoSemantic and T4-NoAAP are unaffected.

## Table 1 — Baseline Comparison (TRIAGE/4 vs. Classical Schedulers)

> **Scenario labels (2026-07-15):** aligned to the manuscript IDs. Two rows were
> relabelled from the pre-manuscript scheme — `S1 — Legit Extreme Emergency` →
> **R3 — Legitimate Extreme Emergency**, and `S2 — Near-Saturation Constrained` →
> **S1 — Near-Saturation Scarcity** — confirmed by matching drop rates and
> BACKGROUND-latency values against the manuscript, not by name. The `C1b`/`C2b`/
> `C3b` variants are supplemental and have no manuscript core-table target; they
> are left as-is pending author confirmation. Only display labels changed; all
> values and code keys are unchanged. See `REFERENCE_Scenario_Crosswalk.md`.

Metrics: alarm mean latency (ms, mean ± 95% CI half-width), Jain latency fairness index (per device, mean ± CI), BACKGROUND mean latency (ms, mean ± CI), alarm drop rate (%, mean ± CI). n = 50 runs per scenario, base-seed = 999. WFQ: pure geographic (is_alarm ignored in dispatch). TBP: pure geographic bands + token buckets (is_alarm ignored in dispatch; no A component).

### Alarm Mean Latency (ms)

| Scenario | TRIAGE/4 | Strict | FIFO | WFQ | DRR | TBP |
|---|---|---|---|---|---|---|
| C1 — Alarm Under Burst | 29.28 ± 10.98 | 525.00 ± 191.12 | 169.93 ± 37.61 | 29.28 ± 10.98 | 132.76 ± 24.49 | 525.00 ± 191.12 |
| C1b — Alarm Under Burst (Phased) | 29.04 ± 10.23 | 695.49 ± 245.13 | 139.53 ± 32.52 | 34.71 ± 14.69 | 99.32 ± 18.53 | 695.49 ± 245.13 |
| C2 — Device Monopolization | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C2b — Device Monopolization Sweep | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C3 — Multi-Zone Emergency | 27.75 ± 5.88 | 161.70 ± 13.83 | 228.24 ± 16.68 | 78.01 ± 10.12 | 228.24 ± 16.68 | 193.76 ± 14.72 |
| C3b — Multi-Zone Emergency Cascade | 11.50 ± 2.99 | 11.50 ± 2.99 | 11.50 ± 2.99 | 11.50 ± 2.99 | 11.50 ± 2.99 | 11.50 ± 2.99 |
| R1 — Alarm Flood Attack | 137.92 ± 15.33 | 1700.64 ± 132.61 | 1406.50 ± 108.02 | 1700.64 ± 132.61 | 1688.34 ± 132.47 | 14688.37 ± 2.80 |
| R2 — Alarm Malfunction Surge | 132.09 ± 13.89 | 567.36 ± 78.92 | 567.36 ± 78.92 | 567.36 ± 78.92 | 567.36 ± 78.92 | 617.10 ± 97.55 |
| R3 — Legitimate Extreme Emergency | 206.31 ± 16.72 | 311.40 ± 26.37 | 245.02 ± 19.14 | 244.54 ± 18.25 | 245.02 ± 19.14 | 292.18 ± 24.55 |
| S1 — Near-Saturation Scarcity | 118.17 ± 7.58 | 293.52 ± 18.88 | 440.16 ± 48.15 | 122.33 ± 7.00 | 359.62 ± 26.57 | 288.11 ± 18.30 |

### Device Jain Fairness

| Scenario | TRIAGE/4 | Strict | FIFO | WFQ | DRR | TBP |
|---|---|---|---|---|---|---|
| C1 — Alarm Under Burst | 0.709 ± 0.039 | 0.683 ± 0.054 | 0.827 ± 0.028 | 0.775 ± 0.036 | 0.848 ± 0.028 | 0.683 ± 0.054 |
| C1b — Alarm Under Burst (Phased) | 0.501 ± 0.013 | 0.361 ± 0.049 | 0.513 ± 0.016 | 0.546 ± 0.021 | 0.523 ± 0.012 | 0.361 ± 0.049 |
| C2 — Device Monopolization | 0.838 ± 0.029 | 0.648 ± 0.009 | 0.648 ± 0.009 | 0.838 ± 0.029 | 0.650 ± 0.009 | 0.648 ± 0.009 |
| C2b — Device Monopolization Sweep | 0.635 ± 0.017 | 0.934 ± 0.015 | 0.934 ± 0.015 | 0.635 ± 0.017 | 0.942 ± 0.015 | 0.934 ± 0.015 |
| C3 — Multi-Zone Emergency | 0.646 ± 0.008 | 0.702 ± 0.008 | 0.733 ± 0.008 | 0.667 ± 0.009 | 0.733 ± 0.008 | 0.716 ± 0.007 |
| C3b — Multi-Zone Emergency Cascade | 0.586 ± 0.006 | 0.586 ± 0.006 | 0.586 ± 0.006 | 0.586 ± 0.006 | 0.586 ± 0.006 | 0.586 ± 0.006 |
| R1 — Alarm Flood Attack | 0.693 ± 0.023 | 0.327 ± 0.014 | 0.997 ± 0.000 | 0.327 ± 0.014 | 0.391 ± 0.018 | 0.178 ± 0.000 |
| R2 — Alarm Malfunction Surge | 0.741 ± 0.028 | 0.557 ± 0.035 | 0.984 ± 0.003 | 0.614 ± 0.023 | 0.563 ± 0.047 | 0.652 ± 0.017 |
| R3 — Legitimate Extreme Emergency | 0.546 ± 0.013 | 0.399 ± 0.010 | 0.541 ± 0.010 | 0.468 ± 0.011 | 0.541 ± 0.010 | 0.421 ± 0.011 |
| S1 — Near-Saturation Scarcity | 0.203 ± 0.001 | 0.474 ± 0.032 | 0.935 ± 0.009 | 0.503 ± 0.019 | 0.921 ± 0.007 | 0.517 ± 0.029 |

### BACKGROUND Mean Wait (ms)

| Scenario | TRIAGE/4 | Strict | FIFO | WFQ | DRR | TBP |
|---|---|---|---|---|---|---|
| C1 — Alarm Under Burst | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C1b — Alarm Under Burst (Phased) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C2 — Device Monopolization | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C2b — Device Monopolization Sweep | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C3 — Multi-Zone Emergency | 487.62 ± 11.77 | 485.00 ± 11.73 | 477.43 ± 11.62 | 487.62 ± 11.77 | 477.43 ± 11.62 | 482.60 ± 11.70 |
| C3b — Multi-Zone Emergency Cascade | 472.93 ± 14.10 | 472.93 ± 14.10 | 472.93 ± 14.10 | 472.93 ± 14.10 | 472.93 ± 14.10 | 472.93 ± 14.10 |
| R1 — Alarm Flood Attack | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| R2 — Alarm Malfunction Surge | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| R3 — Legitimate Extreme Emergency | 63.19 ± 7.77 | 58.53 ± 7.34 | 41.47 ± 4.50 | 54.25 ± 6.09 | 41.47 ± 4.50 | 61.80 ± 7.76 |
| S1 — Near-Saturation Scarcity | 7506.11 ± 152.80 | 1185.82 ± 162.60 | 478.24 ± 42.54 | 1014.23 ± 107.70 | 483.47 ± 44.22 | 1256.63 ± 180.56 |

### Alarm Drop Rate (%)

| Scenario | TRIAGE/4 | Strict | FIFO | WFQ | DRR | TBP |
|---|---|---|---|---|---|---|
| C1 — Alarm Under Burst | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C1b — Alarm Under Burst (Phased) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C2 — Device Monopolization | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C2b — Device Monopolization Sweep | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C3 — Multi-Zone Emergency | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C3b — Multi-Zone Emergency Cascade | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| R1 — Alarm Flood Attack | 17.50 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| R2 — Alarm Malfunction Surge | 20.36 ± 0.40 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| R3 — Legitimate Extreme Emergency | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| S1 — Near-Saturation Scarcity | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

## Table 2 — Ablation Study (Leave-One-Out)

Scenarios: C1 (alarm_under_burst), C2 (device_monopolization), R1 (alarm_flood_attack), R2 (alarm_malfunction_surge), S2 (alarm_load_near_saturation_constrained). Same metric set and CI conventions as Table 1.

### Alarm Mean Latency (ms)

| Scenario | TRIAGE/4 | T4-NoSemantic | T4-FIFOInBand | T4-NoTokens | T4-NoAAP |
|---|---|---|---|---|---|
| C1 — Alarm Under Burst | 29.28 ± 10.98 | 525.00 ± 191.12 | 29.28 ± 10.98 | 29.28 ± 10.98 | 29.28 ± 10.98 |
| C2 — Device Monopolization | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| R1 — Alarm Flood Attack | 137.92 ± 15.33 | 14688.37 ± 2.80 | 137.92 ± 15.33 | 137.92 ± 15.33 | 366.74 ± 56.41 |
| R2 — Alarm Malfunction Surge | 132.09 ± 13.89 | 617.10 ± 97.55 | 132.09 ± 13.89 | 132.09 ± 13.89 | 567.36 ± 78.92 |
| S1 — Near-Saturation Scarcity | 118.17 ± 7.58 | 212.78 ± 16.81 | 118.17 ± 7.58 | 122.33 ± 7.00 | 118.17 ± 7.58 |

### Device Jain Fairness

| Scenario | TRIAGE/4 | T4-NoSemantic | T4-FIFOInBand | T4-NoTokens | T4-NoAAP |
|---|---|---|---|---|---|
| C1 — Alarm Under Burst | 0.709 ± 0.039 | 0.683 ± 0.054 | 0.709 ± 0.039 | 0.709 ± 0.039 | 0.709 ± 0.039 |
| C2 — Device Monopolization | 0.838 ± 0.029 | 0.838 ± 0.029 | 0.648 ± 0.009 | 0.838 ± 0.029 | 0.838 ± 0.029 |
| R1 — Alarm Flood Attack | 0.693 ± 0.023 | 0.178 ± 0.000 | 0.693 ± 0.023 | 0.693 ± 0.023 | 0.818 ± 0.010 |
| R2 — Alarm Malfunction Surge | 0.741 ± 0.028 | 0.609 ± 0.019 | 0.741 ± 0.028 | 0.741 ± 0.028 | 0.412 ± 0.038 |
| S1 — Near-Saturation Scarcity | 0.203 ± 0.001 | 0.211 ± 0.002 | 0.203 ± 0.001 | 0.459 ± 0.026 | 0.203 ± 0.001 |

### BACKGROUND Mean Wait (ms)

| Scenario | TRIAGE/4 | T4-NoSemantic | T4-FIFOInBand | T4-NoTokens | T4-NoAAP |
|---|---|---|---|---|---|
| C1 — Alarm Under Burst | 0.00 ± 0.00 | 525.00 ± 191.12 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C2 — Device Monopolization | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| R1 — Alarm Flood Attack | 0.00 ± 0.00 | 14688.37 ± 2.80 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| R2 — Alarm Malfunction Surge | 0.00 ± 0.00 | 2132.99 ± 385.72 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| S1 — Near-Saturation Scarcity | 7506.11 ± 152.80 | 7506.11 ± 152.80 | 7506.11 ± 152.80 | 1185.82 ± 162.60 | 7506.11 ± 152.80 |

### Alarm Drop Rate (%)

| Scenario | TRIAGE/4 | T4-NoSemantic | T4-FIFOInBand | T4-NoTokens | T4-NoAAP |
|---|---|---|---|---|---|
| C1 — Alarm Under Burst | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| C2 — Device Monopolization | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| R1 — Alarm Flood Attack | 17.50 ± 0.00 | 0.00 ± 0.00 | 17.50 ± 0.00 | 17.50 ± 0.00 | 0.00 ± 0.00 |
| R2 — Alarm Malfunction Surge | 20.36 ± 0.40 | 0.00 ± 0.00 | 20.36 ± 0.40 | 20.36 ± 0.40 | 0.00 ± 0.00 |
| S1 — Near-Saturation Scarcity | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

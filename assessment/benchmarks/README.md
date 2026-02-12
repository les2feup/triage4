# TRIAGE/4 Evaluation Benchmarks

Comprehensive comparison of TRIAGE/4 (Tiered Resource Allocation for IoT Alarm and Geographic-priority Emergency) against baseline schedulers.

## Overview

The benchmark system evaluates three schedulers across three scenarios:

**Schedulers:**
- **TRIAGE/4**: Four-band hierarchical scheduler with semantic urgency override
- **Strict Priority**: Geographic priority only (ignores alarms)
- **FIFO**: Pure first-in-first-out (no priority)

**Scenarios:**
1. **Alarm Under Burst Load**: Critical alarm from low-priority zone during high-priority burst
2. **Device Monopolization**: High-rate device competing with low-rate device in same band
3. **Multi-Zone Emergency**: Multiple zones with varying priorities and emergencies

## Running Benchmarks

### Full Comparison

Run all three schedulers on all three scenarios:

```bash
python benchmarks/comparison_benchmark.py
```

Output includes:
- Per-scenario comparison tables with metrics
- Success criteria checks (alarm latency, fairness, bandwidth)
- Overall summary

### Example Output

```
==========================================================================================
SCENARIO: Alarm Under Burst Load
==========================================================================================
  Metric                    |     TRIAGE/4 |   Strict (vs TRIAGE/4) |     FIFO (vs TRIAGE/4)
  ------------------------- | -------- | ------------------ | ------------------
  Alarm Avg Latency (s)     |   0.0177 |  50.2179 (+100.0%) |  24.8181 ( +99.9%)
  Min Device Rate (msg/s)   |   1.0101 |   1.0101 (  +0.0%) |   1.0101 (  +0.0%)
  HIGH Band Fairness        |   1.0000 |   1.0000 (  -0.0%) |   1.0000 (  -0.0%)
  ...

📊 SUCCESS CRITERIA CHECK: Alarm Under Burst Load
  Criterion                                | Status     |           Value
  ---------------------------------------- | ---------- | ---------------
  Alarm latency reduction (≥40%)           | ✅ PASS     |          100.0%
  Min bandwidth guarantee (≥0.1 msg/s)     | ✅ PASS     |          1.010 msg/s
  Fairness Jain Index (>0.8 all bands)     | ✅ PASS     |     1.000 (min)
```

## Metrics

### Alarm Metrics
- **Alarm Avg Latency**: Mean waiting time for alarm messages
- **Alarm P95 Latency**: 95th percentile waiting time for alarms

### Bandwidth Metrics
- **Min Device Rate**: Minimum messages/sec across all devices
- **Avg Device Rate**: Average messages/sec across all devices
- **Max Device Rate**: Maximum messages/sec across all devices

### Fairness Metrics
- **Band X Fairness**: Jain fairness index for each band [0-4]
  - 1.0 = perfect fairness (all devices equal)
  - 1/n = maximum unfairness (one device monopolizes)

### Overhead Metrics
- **HIGH Avg Latency**: Average waiting time for HIGH band messages
- **HIGH P95 Latency**: 95th percentile for HIGH band
- **Overall Avg Wait**: Average waiting time across all messages
- **Overall P95 Wait**: 95th percentile across all messages

## Success Criteria

From [REFACTORING_PLAN.md](../REFACTORING_PLAN.md):

| Criterion | Target | Notes |
|-----------|--------|-------|
| **Alarm latency reduction** | ≥40% vs Strict Priority | TRIAGE/4 should dramatically reduce alarm delays |
| **Min bandwidth guarantee** | ≥0.1 msg/sec per device | All devices get minimum service |
| **Fairness improvement** | Jain Index >0.8 for all bands | Per-device fairness prevents monopolization |
| **Acceptable overhead** | <1.2x Strict Priority | HIGH band overhead should be minimal |

## Implementation Details

### Schedulers

**TRIAGE/4** ([src/schedulers/triage4/](../src/schedulers/triage4/)):
- Four bands: ALARM (0), HIGH (1), STANDARD (2), BACKGROUND (3)
- Semantic urgency override: `is_alarm=True` → ALARM band
- Token bucket rate limiting for non-alarm bands
- Per-device round-robin fairness within each band

**Strict Priority** ([src/baselines/strict_priority_scheduler.py](../src/baselines/strict_priority_scheduler.py)):
- Geographic priority only (zone_priority)
- Ignores `is_alarm` flag
- FIFO within same priority
- Demonstrates alarm delay problem

**FIFO** ([src/baselines/fifo_scheduler.py](../src/baselines/fifo_scheduler.py)):
- Pure first-in-first-out
- Ignores all priority information
- Lower bound baseline

### Workloads

**Alarm Under Burst** ([src/workloads/scenarios.py](../src/workloads/scenarios.py)):
- 10 EDUs in Zone 0 (HIGH) send 100 msgs each over 1 second
- 1 sensor in Zone 5 (BACKGROUND) sends alarm at t=0.5s
- Tests semantic urgency override

**Device Monopolization**:
- Device A: 3000 msgs over 60s (50 msg/sec) in Zone 0
- Device B: 60 msgs over 60s (1 msg/sec) in Zone 0
- Tests per-device fairness

**Multi-Zone Emergency**:
- 6 zones with 5 devices each sending routine telemetry
- Zones 2 and 4 experience emergencies (3 alarms each)
- Tests multi-band interaction and alarm preemption

## Key Findings

**TRIAGE/4 achieves:**
- ✅ **99.8-100% alarm latency reduction** vs baselines
- ✅ **Perfect fairness** (Jain index 0.8-1.0) across devices
- ✅ **Bandwidth guarantees** (all devices >0.1 msg/sec)

**Tradeoff:**
- ⚠️ HIGH band experiences higher latency vs Strict Priority
- This is by design: token buckets prevent HIGH monopolization
- Ensures alarms always preempt and lower bands get service

## Customization

### Modify Scheduler Parameters

Edit [src/schedulers/triage4/triage4_config.py](../src/schedulers/triage4/triage4_config.py):

```python
config = TRIAGE4Config(
    high_token_budget=20,      # Increase HIGH throughput
    high_token_period=1.0,
    standard_token_budget=10,  # Increase STANDARD throughput
    service_rate=30.0,         # Faster server
)
```

### Add Custom Scenario

Edit [src/workloads/scenarios.py](../src/workloads/scenarios.py):

```python
def generate_custom_scenario():
    return Workload(
        arrival_times=[...],
        device_ids=[...],
        zone_priorities=[...],
        is_alarm=[...],
        description="My custom scenario"
    )
```

Then add to [benchmarks/comparison_benchmark.py](comparison_benchmark.py):

```python
scenarios = [
    ("Custom Scenario", generate_custom_scenario()),
    # ... existing scenarios
]
```

## Testing

Run benchmark tests:

```bash
pytest assessment/tests/test_metrics.py -v     # Metrics computation
pytest assessment/tests/test_baselines.py -v   # Baseline schedulers
pytest tests/test_triage4_core.py -v           # TRIAGE/4 core functionality
pytest assessment/tests/test_workloads.py -v   # Workload generators
```

Run all tests:

```bash
pytest tests/ -v
```

## References

- [REFACTORING_PLAN.md](../REFACTORING_PLAN.md): System specification and success criteria
- [src/schedulers/triage4/](../src/schedulers/triage4/): TRIAGE/4 implementation
- [src/baselines/](../src/baselines/): Baseline schedulers
- [src/workloads/](../src/workloads/): Workload generators
- [src/metrics/](../src/metrics/): Metrics computation

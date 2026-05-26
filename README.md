# TRIAGE/4

![LES2 Banner](https://github.com/les2feup/guidelines/raw/main/src/figures/les2banner.png)

[![PyPI version](https://img.shields.io/pypi/v/triage4.svg)](https://pypi.org/project/triage4/)
[![Python versions](https://img.shields.io/pypi/pyversions/triage4.svg)](https://pypi.org/project/triage4/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TRIAGE/4** (*Tiered Resource Allocation for IoT Alarm and Geographic-priority Emergency*) is a four-band message scheduler designed for distributed IoT emergency monitoring networks.

It resolves the classic **priority inversion** conflict between geographic zone priorities (e.g., routine telemetry from high-priority zones delaying critical emergency alarms from low-priority zones) while preventing channel monopolization by malicious or malfunctioning devices.

---

## Key Features

1. **Semantic & Geographic Prioritization**: Divides message traffic into 4 hierarchical bands:
   * **Band 0 (ALARM)**: Critical emergency alarms. Served with strict priority.
   * **Band 1 (HIGH)**: High-priority zone telemetry (token-constrained).
   * **Band 2 (STANDARD)**: Standard zone telemetry (token-constrained).
   * **Band 3 (BACKGROUND)**: Low-priority zone telemetry (token-constrained).
2. **Adaptive Alarm Protection**: Uses rate monitoring and token buckets to detect and mitigate alarm floods (DoS attacks or sensor failures) by dropping excessive traffic in Band 0 while maintaining clean service for legitimate alarms.
3. **Per-Device Fair Queuing**: Employs round-robin queueing inside bands to prevent a single high-frequency device from starving other sensors in the same band.

---

## Installation

Install the core runtime package from PyPI:
```bash
pip install triage4
```

To run the research benchmarks or test suites locally, install with development dependencies:
```bash
pip install "triage4[dev,research]"
```

---

## Quick Start

```python
from triage4 import TRIAGE4Scheduler, TRIAGE4Config

# Configure the scheduler
config = TRIAGE4Config(
    service_rate=20.0,            # Mean service rate (messages/second)
    high_zone_max=1,              # Zone priority threshold for HIGH band (0-1)
    standard_zone_max=3,          # Zone priority threshold for STANDARD band (2-3)
    high_token_budget=10,         # Maximum tokens per refill window for HIGH band
    enable_alarm_protection=True, # Protect against alarm storms
)

# Initialize the scheduler
scheduler = TRIAGE4Scheduler(config, scheduler_seed=42)

# Define a message workload
arrival_times = [0.0, 0.1, 0.15, 0.2]
device_ids = ["sensor_1", "sensor_2", "sensor_1", "sensor_3"]
zone_priorities = [0, 5, 0, 2]  # Lower number = higher priority
is_alarm = [False, True, False, False]

# Execute discrete-event simulation
result = scheduler.schedule(
    arrival_times=arrival_times,
    device_ids=device_ids,
    zone_priorities=zone_priorities,
    is_alarm=is_alarm
)

# Inspect outcomes
print("Waiting times:", result.waiting_times)
print("E2E latency:", result.e2e_times)
print("Assigned bands:", result.priorities)  # Mapped to Band 0, 1, 2, or 3
print("Alarms dropped:", result.metadata.get("alarm_dropped", 0))
```

---

## Project Structure

* `src/triage4/`: Contains the core scheduling logic and algorithms (packaged for distribution).
* `assessment/`: Contains baselines, scenarios, workloads, and performance benchmarking scripts (excluded from wheel packaging).
* `tests/`: Integration and unit tests.

---

## Development and Testing

Run the test suite using `pytest`:
```bash
pytest
```

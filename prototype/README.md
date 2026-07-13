# TRIAGE/4 Broker Prototype

A minimal MQTT 5.0 broker that runs an online TRIAGE/4 egress dispatcher, built
to answer reviewers **R1.1** (real-broker validation) and **R2.1** (deployment
overhead). It confirms on real hardware that the scheduling behaviour
characterised in simulation holds end-to-end over a network.

The authoritative design — decisions, parity contracts, and the run matrix —
lives in [`../docs/chat-reports/FEATURE_Stage3b_Broker_Prototype_Plan.md`](../docs/chat-reports/FEATURE_Stage3b_Broker_Prototype_Plan.md).
This README is the operational quick-start; the plan is the why.

## Isolation

This directory is self-contained and installs the **published** `triage4==1.1.0`
package from PyPI — it never imports the repository's local `src/`. That keeps
the prototype auditable as a deployer's view of the package.

```bash
python -m venv prototype/.venv
prototype/.venv/bin/pip install -r prototype/requirements.txt
```

`prototype/.venv/` and `prototype/results/` are gitignored.

## Layout

```
broker/      minimal MQTT5 broker + online dispatchers (FIFO/Strict baselines + TRIAGE/4)
clients/     zone_client (host: one process, all zones) and zone_agent (Pi: one device per zone)
workloads/   pre-generated C3/R3 arrival schedules (identical to the simulation)
results/     gitignored CSV/plots
analyze.py   RTT (alarm vs routine) + R2.1 overhead stats
run_host.sh  one-command loopback run
run_pi.sh    one-command Pi run (drives the six zone agents via the GO control topic)
```

## Running

Loopback (one process replays every zone):

```bash
./run_host.sh                     # {fifo, strict, triage4} x {C3, R3}
```

Pi testbed (six devices, one per geographic priority) — see
`docs/PI_TESTBED_GUIDE.md`. Start a long-lived agent on each zone device, then
drive the campaign from the Pi:

```bash
.venv/bin/python -m clients.zone_agent --zone <0..5> --host <pi-host>.local   # each device
./run_pi.sh                                                                    # on the Pi
```

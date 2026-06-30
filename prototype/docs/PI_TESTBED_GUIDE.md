# Raspberry Pi Testbed Guide — PLACEHOLDER

> This file is a scaffold. It is filled in at deployment time (plan **Part F**),
> once the broker and clients are validated on the host. Do not treat the
> section stubs below as final instructions.

The deployment setup is specified in plan Part F
(`../../docs/chat-reports/FEATURE_Stage3b_Broker_Prototype_Plan.md`, §8):
Pi as broker, each zone as one client device on a private 5 GHz WiFi with WiFi
power-save disabled; single-clock RTT (no NTP) because every zone-client
receives back its own messages.

## To be written at deployment

- [ ] Pi OS image, Python version, and `prototype/.venv` setup on the Pi
- [ ] Broker static IP / hostname on the private WiFi network
- [ ] WiFi AP configuration (5 GHz, fixed channel, power-save off)
- [ ] Per-zone client host setup and zone→device assignment
- [ ] Coordinated start procedure (shared start timestamp)
- [ ] `run_pi.sh` invocation and result collection
- [ ] Measured network baseline (idle RTT, jitter) for context

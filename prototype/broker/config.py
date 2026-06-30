"""
Broker run configuration — STUB (implemented in Part D).

Selects the egress discipline and builds the matching online dispatcher for a
single run (one scheduler per process, no cross-run token/AAP contamination).

Planned responsibilities (plan §6, "Scheduler selection"):
    - Choose one of {fifo, strict, triage4} via CLI/env.
    - For triage4: build the SAME ``TRIAGE4Config`` the C3/R3 simulation used
      (AAP on per open item O4) and a ``triage4.Triage4EgressDispatcher`` so the
      token/AAP behaviour is a faithful port of the simulated configuration.
    - Expose the broker egress rate ``C`` (the saturation knob), which is
      INDEPENDENT of ``TRIAGE4Config.service_rate`` (the online dispatcher
      ignores service_rate; the transmitter paces at C).
"""

"""
Result analysis — STUB (implemented in Part E).

Merges the client-side RTT CSVs with the broker-side overhead CSV on msg_id and
produces the two reviewer-facing artifacts:

    R1.1 — RTT distributions split alarm vs routine, per scheduler
           (FIFO / Strict / TRIAGE/4), per scenario (C3 / R3). Confirms on
           hardware that TRIAGE/4 keeps alarm RTT low where Strict inverts.
    R2.1 — per-message scheduling overhead (enqueue_ns + select_ns) reported
           WITH active-device count (not a flat constant, per the concurrency
           note in plan §6).

Reads from results/, writes plots/tables back to results/.
"""

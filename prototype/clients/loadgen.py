"""
Per-zone load generator — STUB (implemented in Part D/E).

In-process helper (not an entry point). Replays ONE zone's slice of a
pre-generated schedule, publishing each message to the fixed ingest topic
``t4/in`` with user properties zone/device/alarm and a payload
``b"<msg_id>:<t_send>"`` where ``t_send`` is this client's ``time.monotonic_ns()``
(D11). Paced to the schedule's arrival times.

Lives inside ``zone_client.py``'s single process so publish and receive share
one monotonic clock (single-clock RTT, no NTP).
"""

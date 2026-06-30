"""
Per-zone receiver — STUB (implemented in Part D/E).

In-process helper (not an entry point). Subscribes to this zone's return topic
``t4out/<zone>`` (exact match, no wildcard) and, on each delivered message,
computes ``RTT = time.monotonic_ns() - t_send`` using THIS process's clock — the
same clock loadgen stamped t_send with — then writes a per-message row
(msg_id, zone, device, alarm, t_send, t_recv, rtt_ns) to a client-side CSV.

The client-side RTT CSV is merged with the broker-side overhead CSV on msg_id
during analysis (plan §6/§7).
"""

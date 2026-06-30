"""
Asyncio MQTT 5.0 broker — STUB (implemented in Part D, after M0).

A single-event-loop broker: ingest -> dispatcher.enqueue -> rate-limited
transmitter -> dispatcher.select_next -> deliver. One asyncio loop means
enqueue and select_next never run concurrently, so no locks are needed.

Planned data path (plan §6):
    Ingest: read user properties (zone/device/alarm) off the fixed topic
        ``t4/in``; compute relative ``now = monotonic() - t0`` (D10) for the
        dispatcher ONLY; assign a globally-unique zone-prefixed msg_id; call
        ``admitted = dispatcher.enqueue(handle, device_id, zone, is_alarm, now)``
        wrapped in perf_counter_ns() for enqueue overhead. If admitted is False
        (AAP shed), record the drop broker-side and do not deliver.
    Transmit: a coroutine paced at egress rate C. On each tick, relative now,
        ``msg = dispatcher.select_next(now)`` (timed for select overhead); if
        not None, republish the payload UNTOUCHED on ``t4out/<zone>`` and bank
        the slot (next_t += 1/C); if None, do NOT bank (next_t = now + 1/C) to
        avoid an idle catch-up burst (D12).
    Clock safety: the payload carries the CLIENT's t_send (monotonic_ns, D11)
        and is forwarded byte-for-byte; the broker's relative ``now`` is a
        separate clock used only for scheduling/refill and never enters RTT.

Overhead CSV (R2.1 evidence): per message accumulate enqueue_ns, select_ns,
band, dropped, active_devices; flush to
``results/broker_<scheduler>_<scenario>.csv`` at shutdown. Reported WITH the
active-device count, not as a flat constant.
"""

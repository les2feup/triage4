"""
Zone client ENTRY POINT — STUB (implemented in Part D/E).

Runs ``loadgen`` and ``receiver`` together in ONE process for a single zone.
Co-locating publish and receive in one process is what makes RTT single-clock:
``t_send`` (stamped at publish) and ``t_recv`` (read at delivery) come from the
same ``time.monotonic_ns()``, so no NTP or cross-host clock sync is needed
(D7/D11).

Planned CLI: --zone, --schedule <json>, --broker-host, --broker-port,
--start-at <shared epoch> (coordinated start across zone clients), --out <csv>.
One process per zone; on the Pi testbed each zone client runs on its own device.
"""

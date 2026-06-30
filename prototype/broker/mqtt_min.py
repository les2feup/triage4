"""
Minimal MQTT 5.0 wire codec — STUB (implemented in Part D, after M0).

This module will parse and emit the small MQTT 5.0 subset the prototype needs.
It is intentionally empty until the **M0 interop smoke test** (plan §6) proves a
paho-mqtt v5 client can CONNECT/SUBSCRIBE/PUBLISH/receive against the bare
broker — M0 retires the hand-rolled v5 framing risk before any scheduling code
is added.

Planned subset (plan §6, "MQTT 5.0 subset"):
    - Fixed header + the v5 PROPERTY field (property-length varint + identifiers).
      Only 0x26 UserProperty (UTF-8 string pairs) is interpreted.
    - CONNECT (protocol level 5) -> CONNACK (v5, empty properties)
    - SUBSCRIBE -> SUBACK (register topic -> writer)
    - PUBLISH (QoS 0 ingest; read user properties zone/device/alarm)
    - PINGREQ -> PINGRESP
    - Ignored: QoS 1/2, retained, wildcards, will, all other property ids.

Target ~250-350 lines once implemented (full v5 framing on CONNACK/SUBACK/
PUBLISH is the real cost). No third-party deps — pure bytes.
"""

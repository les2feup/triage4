"""
Per-zone load generator (in-process helper, not an entry point).

Replays a list of schedule entries by publishing each to the fixed ingest topic
``t4/in`` at its real-time offset, attaching zone/device/alarm as MQTT 5.0 user
properties and stamping ``t_send = time.monotonic_ns()`` into the payload
``<msg_id>:<t_send>:<filler>``. Pacing is relative to a shared start instant so
the replay reproduces the schedule's arrival pattern.

Every payload is padded to a fixed size so the broker and the network path carry
a representative, constant-size frame rather than a few-byte string — the RTT
then reflects transferring a realistic message, and no scheduler is advantaged by
a shorter payload. The filler is constant, so the payload is fixed apart from the
identifying header the receiver needs to match the return and compute RTT.

Lives in the same process as the receiver (zone_client) so publish and receive
share one monotonic clock — single-clock RTT, no NTP (D11).
"""

import time
from typing import List

import paho.mqtt.client as mqtt
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties

INGEST_TOPIC = "t4/in"

# Fixed on-the-wire payload size (bytes), a representative IoT telemetry/alarm
# frame. The header ``<msg_id>:<t_send_ns>:`` is well under this, and the rest is
# constant filler.
PAYLOAD_SIZE = 500
_FILLER = b"\x00"


def replay(client: mqtt.Client, messages: List[dict], start_monotonic: float) -> None:
    """Publish each message at ``start_monotonic + entry['t']`` (seconds).

    ``messages`` must be sorted by ``t``. Blocks until the last message is
    published; network IO runs in paho's background loop thread.
    """
    for entry in messages:
        target = start_monotonic + entry["t"]
        remaining = target - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

        props = Properties(PacketTypes.PUBLISH)
        props.UserProperty = [
            ("zone", str(entry["zone_priority"])),
            ("device", entry["device_id"]),
            ("alarm", "1" if entry["is_alarm"] else "0"),
        ]
        # Header carries the id and send timestamp; trailing colon terminates
        # t_send so the receiver parses it cleanly regardless of the filler.
        header = f"{entry['msg_id']}:{time.monotonic_ns()}:".encode()
        payload = header + _FILLER * max(0, PAYLOAD_SIZE - len(header))
        client.publish(INGEST_TOPIC, payload=payload, qos=0, properties=props)

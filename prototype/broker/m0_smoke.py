"""
M0 interop smoke test (plan §6).

Drives the bare broker with a real paho-mqtt v5 client to prove the hand-rolled
MQTT 5.0 framing works end to end: CONNECT, SUBSCRIBE, PUBLISH carrying User
Properties, and receipt of the forwarded message with those properties and the
opaque payload intact. Exits 0 on success, 1 on failure.

This retires the largest implementation risk (v5 framing) before any scheduling
code is wired into the broker.

Usage: python -m broker.m0_smoke [host] [port]   (run from prototype/)
"""

import sys
import threading

import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties

TOPIC = "t4/m0"
SENT_PROPS = [("zone", "2"), ("device", "sensor_1"), ("alarm", "1")]
PAYLOAD = b"42:123456789"

_done = threading.Event()
_result: dict = {}


def on_connect(client, userdata, connect_flags, reason_code, properties):
    client.subscribe(TOPIC, qos=0)


def on_subscribe(client, userdata, mid, reason_code_list, properties):
    props = Properties(PacketTypes.PUBLISH)
    props.UserProperty = SENT_PROPS  # the zone/device/alarm metadata channel
    client.publish(TOPIC, payload=PAYLOAD, qos=0, properties=props)


def on_message(client, userdata, message):
    _result["topic"] = message.topic
    _result["payload"] = message.payload
    _result["props"] = getattr(message.properties, "UserProperty", [])
    _done.set()


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 1884

    client = mqtt.Client(CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message

    client.connect(host, port, keepalive=30)
    client.loop_start()
    received = _done.wait(timeout=5.0)
    client.loop_stop()
    client.disconnect()

    if not received:
        print("M0 FAIL: no message received within 5s", file=sys.stderr)
        sys.exit(1)

    if _result["payload"] != PAYLOAD:
        print(f"M0 FAIL: payload mismatch: {_result['payload']!r}", file=sys.stderr)
        sys.exit(1)

    got = dict(_result["props"])
    if got != dict(SENT_PROPS):
        print(f"M0 FAIL: user properties mismatch: {got}", file=sys.stderr)
        sys.exit(1)

    print(f"M0 PASS: topic={_result['topic']} payload={_result['payload']!r} "
          f"user_props={_result['props']}")
    sys.exit(0)


if __name__ == "__main__":
    main()

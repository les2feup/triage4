"""
Part D integration smoke test: scheduler wiring end to end.

Publishes two near-simultaneous messages in the SAME zone — a routine telemetry
(``r``) then an alarm (``a``) — to the ingest topic, subscribes to the zone's
return topic, and prints the delivery order. With a slow egress rate both
messages queue before the transmitter dispatches, so the order reveals the
scheduling policy:

    triage4 -> a,r   (semantic override promotes the alarm above routine)
    strict  -> r,a   (same zone => FIFO; alarm flag ignored)
    fifo    -> r,a   (single FIFO queue)

Exits 0 once both messages are received (the runner asserts the order);
exits 1 on timeout. Usage: python -m broker.d_smoke [host] [port]
"""

import sys
import threading
import time

import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties

ZONE = 5
RETURN_TOPIC = f"t4out/{ZONE}"
INGEST_TOPIC = "t4/in"

_done = threading.Event()
_order: list = []


def _publish(client, msg_id: str, is_alarm: bool) -> None:
    props = Properties(PacketTypes.PUBLISH)
    props.UserProperty = [("zone", str(ZONE)), ("device", "sensor_1"),
                          ("alarm", "1" if is_alarm else "0")]
    payload = f"{msg_id}:{time.monotonic_ns()}".encode()
    client.publish(INGEST_TOPIC, payload=payload, qos=0, properties=props)


def on_connect(client, userdata, connect_flags, reason_code, properties):
    client.subscribe(RETURN_TOPIC, qos=0)


def on_subscribe(client, userdata, mid, reason_code_list, properties):
    _publish(client, "r", is_alarm=False)  # routine first
    _publish(client, "a", is_alarm=True)   # alarm second


def on_message(client, userdata, message):
    _order.append(message.payload.split(b":", 1)[0].decode())
    if len(_order) >= 2:
        _done.set()


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 1883

    client = mqtt.Client(CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.connect(host, port, keepalive=30)
    client.loop_start()
    received = _done.wait(timeout=8.0)
    client.loop_stop()
    client.disconnect()

    if not received:
        print(f"D FAIL: received {_order} (expected 2 messages)", file=sys.stderr)
        sys.exit(1)
    print(f"DELIVER_ORDER {','.join(_order)}")
    sys.exit(0)


if __name__ == "__main__":
    main()

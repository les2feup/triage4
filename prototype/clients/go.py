"""
Run-start coordinator (entry point, executed on the broker side).

Publishes one GO message on ``t4ctl/go`` carrying the cell key
``<scheduler>:<scenario>:<rep>``. Every zone agent starts its replay clock on
receipt, so start skew across the client devices is one WiFi hop (plan F.3).

The cell key makes the agents stateless: they take scheduler, scenario and rep
straight from GO rather than iterating a matrix of their own, so a device that
restarts mid-campaign cannot silently desynchronise.

GO is gated on readiness, not on a settle delay. The broker restarts once per
cell, and an agent that has not finished reconnecting and resubscribing when GO
fires would miss the cell and leave a hole in the results. So this waits until
every expected zone has been heard from on ``t4ctl/ready``, and fails loudly if
one never appears — a missing zone must abort the cell, not corrupt it.

Usage (from prototype/):
    python -m clients.go --host pi.local --cell triage4:c3_multi_zone_emergency:0
"""

import argparse
import sys
import threading
import time

import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion

GO_TOPIC = "t4ctl/go"
READY_TOPIC = "t4ctl/ready"


def main() -> None:
    parser = argparse.ArgumentParser(description="publish the run-start GO")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--cell", required=True,
                        help="cell key <scheduler>:<scenario>:<rep>")
    parser.add_argument("--expect", type=int, default=6,
                        help="zone agents that must be ready before GO")
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    args = parser.parse_args()

    ready: set = set()
    all_ready = threading.Event()

    def on_ready(client, userdata, message) -> None:
        ready.add(message.payload.decode())
        if len(ready) >= args.expect:
            all_ready.set()

    client = mqtt.Client(CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    client.on_connect = lambda c, *_: c.subscribe([(READY_TOPIC, 0)])
    client.message_callback_add(READY_TOPIC, on_ready)
    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()

    if not all_ready.wait(timeout=args.ready_timeout):
        client.loop_stop()
        client.disconnect()
        missing = args.expect - len(ready)
        sys.exit(f"only {len(ready)}/{args.expect} zone agents ready "
                 f"({missing} missing: saw {sorted(ready)}) — aborting cell")

    client.publish(GO_TOPIC, payload=args.cell.encode(), qos=0)
    # paho publishes from its loop thread; let the QoS-0 write reach the socket
    # before tearing the connection down.
    time.sleep(0.2)
    client.loop_stop()
    client.disconnect()
    print(f"GO {args.cell} (zones {sorted(ready)})", flush=True)


if __name__ == "__main__":
    main()

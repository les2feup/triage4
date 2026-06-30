"""
Zone client ENTRY POINT — single-clock RTT measurement.

Runs the load generator and the receiver in ONE process so each message's
``t_send`` (at publish) and ``t_recv`` (at delivery) come from the same
``time.monotonic_ns()`` — single-clock RTT, no NTP (D7/D11).

On the host (loopback) one client replays the whole schedule and subscribes to
every return topic. On the Pi testbed, ``--zones`` restricts a client to a
subset of geographic priorities (one client per device); the receiver still
records RTT only for the msg_ids it sent, so overlapping return-topic
subscriptions remain single-clock.

Writes ``rtt_<scheduler>_<scenario>_<zonespec>.csv`` with one row per delivered
message: msg_id, t_send_ns, t_recv_ns, rtt_ms. ``analyze.py`` joins it with the
broker-side overhead CSV on msg_id.

Usage (from prototype/):
    python -m clients.zone_client --schedule workloads/c3_multi_zone_emergency.json \
        --scheduler triage4 --scenario c3_multi_zone_emergency --out-dir results
"""

import argparse
import csv
import json
import os
import threading
import time
from typing import List, Optional, Set

import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion

from . import loadgen
from .receiver import RttReceiver


def _select_messages(schedule: dict, zones: Optional[Set[int]]) -> List[dict]:
    """Return the schedule messages this client owns (all, or a zone subset)."""
    messages = schedule["messages"]
    if zones is None:
        return messages
    return [m for m in messages if m["zone_priority"] in zones]


def main() -> None:
    parser = argparse.ArgumentParser(description="TRIAGE/4 prototype zone client")
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--zones", default="all",
                        help="comma-separated zone priorities this client owns, or 'all'")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--drain", type=float, default=30.0,
                        help="seconds to wait for returns after the last arrival")
    args = parser.parse_args()

    with open(args.schedule) as handle:
        schedule = json.load(handle)

    zones = None if args.zones == "all" else {int(z) for z in args.zones.split(",")}
    messages = _select_messages(schedule, zones)
    if not messages:
        raise SystemExit(f"no messages for zones {args.zones} in {args.schedule}")

    expected_ids = {m["msg_id"] for m in messages}
    return_topics = sorted({f"t4out/{m['zone_priority']}" for m in messages})
    receiver = RttReceiver(expected_ids)

    client = mqtt.Client(CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    subscribed = threading.Event()

    def on_connect(c, userdata, flags, reason_code, properties):
        c.subscribe([(topic, 0) for topic in return_topics])

    def on_subscribe(c, userdata, mid, reason_code_list, properties):
        subscribed.set()

    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = receiver.on_message

    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()

    if not subscribed.wait(timeout=10.0):
        client.loop_stop()
        raise SystemExit("subscription to return topics timed out")

    # Replay the schedule, then wait for the egress queue to drain back to us.
    start = time.monotonic()
    loadgen.replay(client, messages, start)
    last_arrival = messages[-1]["t"]
    deadline = start + last_arrival + args.drain
    receiver.complete.wait(timeout=max(0.0, deadline - time.monotonic()))

    client.loop_stop()
    client.disconnect()

    os.makedirs(args.out_dir, exist_ok=True)
    zonespec = "all" if zones is None else "-".join(str(z) for z in sorted(zones))
    path = os.path.join(
        args.out_dir,
        f"rtt_{args.scheduler}_{args.scenario}_{zonespec}_rep{args.rep}.csv")
    rows = [dict(rep=args.rep, **record) for record in receiver.records.values()]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["rep", "msg_id", "t_send_ns", "t_recv_ns", "rtt_ms"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"zone_client[{args.scheduler}/{args.scenario}/{zonespec}]: "
          f"sent {len(expected_ids)}, received {receiver.received_count} -> "
          f"{os.path.relpath(path)}", flush=True)


if __name__ == "__main__":
    main()

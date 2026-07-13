"""
Zone agent ENTRY POINT — one long-lived process per client device (Pi testbed).

Owns exactly one geographic priority: it publishes that zone's share of the
schedule and subscribes to that zone's return topic ``t4out/<zone>``, so
``t_send`` and ``t_recv`` are stamped by the same ``time.monotonic_ns()`` —
single-clock RTT, no NTP (D11), exactly as in the host-side ``zone_client``.

Unlike ``zone_client`` (one process, one cell, whole schedule — used for the
loopback runs), the agent is campaign-driven and stateless: it stays connected
and waits on the control topic. Each GO announces a cell key
``<scheduler>:<scenario>:<rep>``; the agent replays that scenario, drains, writes
one CSV named after the cell, and goes back to waiting. Nothing about the
campaign matrix is encoded here, so a device that restarts mid-campaign rejoins
at whatever cell the broker announces next instead of drifting out of step.

Usage (on each zone device, from prototype/):
    python -m clients.zone_agent --zone 3 --host pi.local
"""

import argparse
import csv
import json
import os
import threading
import time
from typing import List, Optional

import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion

from . import loadgen
from .receiver import RttReceiver

GO_TOPIC = "t4ctl/go"
READY_TOPIC = "t4ctl/ready"
# The broker restarts once per cell, so an idle agent republishes its readiness
# rather than announcing it once: the coordinator can then gate GO on all zones
# being reconnected and resubscribed, instead of racing a fixed settle delay.
HEARTBEAT_SECONDS = 1.0


class ZoneAgent:
    """A single-zone device: waits for GO, replays its share, records RTT."""

    def __init__(self, zone: int, schedules_dir: str, out_dir: str,
                 drain: float) -> None:
        self.zone = zone
        self.schedules_dir = schedules_dir
        self.out_dir = out_dir
        self.drain = drain
        self.receiver: Optional[RttReceiver] = None
        self.cell: Optional[str] = None
        self.go = threading.Event()
        self.idle = threading.Event()  # heartbeat only between cells

    # -- MQTT callbacks ------------------------------------------------------

    def on_go(self, client, userdata, message) -> None:
        self.cell = message.payload.decode()
        self.go.set()

    def on_message(self, client, userdata, message) -> None:
        # Return-topic deliveries; ignored between cells (no active receiver).
        if self.receiver is not None:
            self.receiver.on_message(client, userdata, message)

    # -- campaign loop -------------------------------------------------------

    def load_messages(self, scenario: str) -> List[dict]:
        """This zone's share of a scenario, in arrival order."""
        path = os.path.join(self.schedules_dir, f"{scenario}.json")
        with open(path) as handle:
            schedule = json.load(handle)
        return [m for m in schedule["messages"]
                if m["zone_priority"] == self.zone]

    def run_cell(self, client: mqtt.Client, cell: str) -> None:
        """Replay one announced cell and write its RTT records."""
        scheduler, scenario, rep = cell.split(":")
        messages = self.load_messages(scenario)
        if not messages:
            print(f"zone {self.zone}: no messages in {scenario}, skipping",
                  flush=True)
            return

        self.receiver = RttReceiver({m["msg_id"] for m in messages})
        start = time.monotonic()
        loadgen.replay(client, messages, start)
        deadline = start + messages[-1]["t"] + self.drain
        self.receiver.complete.wait(timeout=max(0.0, deadline - time.monotonic()))

        self.write_csv(scheduler, scenario, int(rep))
        print(f"zone {self.zone} [{cell}]: sent {len(messages)}, "
              f"received {self.receiver.received_count}", flush=True)
        self.receiver = None

    def write_csv(self, scheduler: str, scenario: str, rep: int) -> None:
        """One CSV per cell; ``analyze.py`` joins these on (rep, msg_id)."""
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(
            self.out_dir,
            f"rtt_{scheduler}_{scenario}_{self.zone}_rep{rep}.csv")
        rows = [dict(rep=rep, **record)
                for record in self.receiver.records.values()]
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["rep", "msg_id", "t_send_ns", "t_recv_ns", "rtt_ms"])
            writer.writeheader()
            writer.writerows(rows)

    def heartbeat(self, client: mqtt.Client) -> None:
        """Announce this zone on t4ctl/ready whenever it is idle."""
        while True:
            if self.idle.is_set():
                client.publish(READY_TOPIC, payload=str(self.zone).encode(), qos=0)
            time.sleep(HEARTBEAT_SECONDS)

    def serve(self, host: str, port: int) -> None:
        client = mqtt.Client(CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
        # Re-subscribing from on_connect (not once after connect) is what makes the
        # agent survive the broker restart between cells: paho's loop thread
        # reconnects automatically and this restores both subscriptions.
        client.on_connect = lambda c, *_: c.subscribe(
            [(GO_TOPIC, 0), (f"t4out/{self.zone}", 0)])
        client.message_callback_add(GO_TOPIC, self.on_go)
        client.on_message = self.on_message
        client.reconnect_delay_set(min_delay=1, max_delay=2)

        # The broker may not be up yet (or may be between cells): keep trying.
        while True:
            try:
                client.connect(host, port, keepalive=60)
                break
            except OSError:
                time.sleep(1.0)

        client.loop_start()
        threading.Thread(target=self.heartbeat, args=(client,), daemon=True).start()
        print(f"zone agent {self.zone} ready on {host}:{port}", flush=True)

        try:
            while True:
                self.idle.set()
                self.go.wait()
                self.go.clear()
                self.idle.clear()
                self.run_cell(client, self.cell)
        except KeyboardInterrupt:
            pass
        finally:
            client.loop_stop()
            client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="TRIAGE/4 prototype zone agent")
    parser.add_argument("--zone", type=int, required=True,
                        help="the geographic priority this device owns (0-5)")
    parser.add_argument("--host", default="pi.local")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--schedules-dir", default="workloads")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--drain", type=float, default=30.0,
                        help="seconds to wait for returns after the last arrival")
    args = parser.parse_args()

    agent = ZoneAgent(args.zone, args.schedules_dir, args.out_dir, args.drain)
    agent.serve(args.host, args.port)


if __name__ == "__main__":
    main()

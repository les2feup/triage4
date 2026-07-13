"""
Control-centre observer ENTRY POINT — passive, one wired device.

Subscribes to every return topic and records the order in which the broker
actually delivered messages. This is the deployment's message *consumer* (the
operator-facing control centre in an emergency-monitoring system), and it doubles
as third-party evidence for R1.1: each zone agent only ever sees its own zone, so
the interleaved egress order across zones is otherwise known only from the
broker's own CSV — i.e. self-reported. Here an unmodified MQTT subscriber
observes it independently.

It is deliberately NOT a latency instrument. RTT stays measured at the senders
(``zone_agent``), where t_send and t_recv share one clock; taking t_recv here
instead would compare clocks across devices and force NTP/PTP. The timestamps
below are this host's own and are only meaningful as *ordering* and as intervals
within one cell.

Run it on a wired device — not on the Pi (it would spend the broker's CPU) and
not on a zone device (it would perturb that zone's timing). Like the agents it is
long-lived and GO-driven, and it heartbeats on t4ctl/ready, so include it in the
coordinator's expected count (``ZONES=7 ./run_pi.sh``).

Usage:
    python -m clients.observer --host <pi-host>.local --zones 6
"""

import argparse
import csv
import os
import threading
import time
from typing import List, Optional

import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion

GO_TOPIC = "t4ctl/go"
READY_TOPIC = "t4ctl/ready"
HEARTBEAT_SECONDS = 1.0


class Observer:
    """Records the global delivery order the broker produces, per cell."""

    def __init__(self, zones: int, out_dir: str, drain: float) -> None:
        self.zones = zones
        self.out_dir = out_dir
        self.drain = drain
        self.records: Optional[List[dict]] = None
        self.cell: Optional[str] = None
        self.go = threading.Event()
        self.idle = threading.Event()

    def on_go(self, client, userdata, message) -> None:
        self.cell = message.payload.decode()
        self.go.set()

    def on_message(self, client, userdata, message) -> None:
        # Deliveries are appended in arrival order; the row index IS the datum.
        if self.records is None:
            return  # between cells
        t_recv = time.monotonic_ns()
        msg_id, _, _ = message.payload.partition(b":")
        self.records.append({
            "seq": len(self.records),
            "topic": message.topic,
            "msg_id": msg_id.decode(),
            "t_recv_ns": t_recv,
        })

    def heartbeat(self, client: mqtt.Client) -> None:
        while True:
            if self.idle.is_set():
                client.publish(READY_TOPIC, payload=b"observer", qos=0)
            time.sleep(HEARTBEAT_SECONDS)

    def run_cell(self, cell: str) -> None:
        """Collect one announced cell, then write its delivery-order trace."""
        scheduler, scenario, rep = cell.split(":")
        self.records = []
        # No completion signal exists here (the observer does not know how many
        # messages a cell holds), so it simply collects for the cell's duration.
        time.sleep(self.drain)
        rows, self.records = self.records, None

        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(
            self.out_dir, f"observed_{scheduler}_{scenario}_rep{rep}.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["seq", "topic", "msg_id", "t_recv_ns"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"observer [{cell}]: {len(rows)} deliveries -> {path}", flush=True)

    def serve(self, host: str, port: int) -> None:
        topics = [(f"t4out/{z}", 0) for z in range(self.zones)]
        client = mqtt.Client(CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
        client.on_connect = lambda c, *_: c.subscribe([(GO_TOPIC, 0)] + topics)
        client.message_callback_add(GO_TOPIC, self.on_go)
        client.on_message = self.on_message
        client.reconnect_delay_set(min_delay=1, max_delay=2)

        while True:
            try:
                client.connect(host, port, keepalive=60)
                break
            except OSError:
                time.sleep(1.0)

        client.loop_start()
        threading.Thread(target=self.heartbeat, args=(client,), daemon=True).start()
        print(f"observer ready on {host}:{port} (zones 0-{self.zones - 1})",
              flush=True)

        try:
            while True:
                self.idle.set()
                self.go.wait()
                self.go.clear()
                self.idle.clear()
                self.run_cell(self.cell)
        except KeyboardInterrupt:
            pass
        finally:
            client.loop_stop()
            client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="TRIAGE/4 control-centre observer")
    parser.add_argument("--host", default="pi.local")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--zones", type=int, default=6,
                        help="number of return topics t4out/0..N-1 to observe")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--drain", type=float, default=60.0,
                        help="seconds to collect per cell (>= replay span + drain)")
    args = parser.parse_args()

    observer = Observer(args.zones, args.out_dir, args.drain)
    observer.serve(args.host, args.port)


if __name__ == "__main__":
    main()

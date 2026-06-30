"""
Asyncio MQTT 5.0 broker with an online TRIAGE/4 egress dispatcher.

Data path (plan §6):
    Ingest  — clients PUBLISH to the fixed topic ``t4/in`` with zone/device/alarm
              user properties and an opaque payload ``<msg_id>:<t_send_ns>``. The
              broker reads the properties, computes a relative clock
              ``now = monotonic() - t0`` (D10), and calls
              ``dispatcher.enqueue(handle, device, zone, is_alarm, now)`` timed
              with perf_counter_ns(). A False result is an AAP drop: recorded and
              not delivered (a dropped message never yields a t_recv, so drops
              must be counted broker-side).
    Egress  — a transmitter coroutine paced at rate ``C`` calls
              ``dispatcher.select_next(now)`` (timed) and republishes the payload
              UNTOUCHED on the per-zone return topic ``t4out/<zone>``. Busy ticks
              bank the slot; idle ticks do not (D12), avoiding a catch-up burst.

A single asyncio event loop serialises ingest and egress, so the dispatcher and
the bookkeeping need no locks. Clock safety: the payload carries the client's
t_send and is forwarded byte-for-byte; the broker's relative ``now`` is a
separate clock used only for scheduling and never enters RTT.

Overhead (R2.1 evidence) accumulates per delivered/dropped message and flushes
to ``results/broker_<scheduler>_<scenario>.csv`` at shutdown, reported with the
active-device count rather than as a flat constant.
"""

import argparse
import asyncio
import csv
import os
import signal
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from . import mqtt_min
from .config import SCHEDULERS, build_classifier, build_config, build_dispatcher

INGEST_TOPIC = "t4/in"


class Broker:
    """MQTT 5.0 broker that schedules egress through an online dispatcher."""

    def __init__(self, dispatcher, classifier, scheduler: str, scenario: str,
                 rep: int, egress_rate: float, results_dir: str) -> None:
        self.dispatcher = dispatcher
        self.classifier = classifier
        self.scheduler = scheduler
        self.scenario = scenario
        self.rep = rep
        self.egress_rate = egress_rate
        self.results_dir = results_dir

        # topic -> subscriber writers (exact match; no wildcards per D4).
        self._subscriptions: Dict[str, Set[asyncio.StreamWriter]] = {}
        # handle -> (zone, device, payload, msg_id, band, enqueue_ns) in-flight.
        self._registry: Dict[int, Tuple[int, str, bytes, str, int, int]] = {}
        # devices with at least one in-flight message (for active-device count).
        self._inflight: Counter = Counter()
        self._next_handle = 0
        self._t0 = 0.0
        # Accumulated overhead rows flushed to CSV at shutdown.
        self._overhead: List[dict] = []

    # -- connection handling -------------------------------------------------

    async def handle_client(self, reader: asyncio.StreamReader,
                            writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                ptype, flags, body = await mqtt_min.read_packet(reader)
                if ptype == mqtt_min.CONNECT:
                    writer.write(mqtt_min.build_connack())
                    await writer.drain()
                elif ptype == mqtt_min.SUBSCRIBE:
                    packet_id, filters = mqtt_min.parse_subscribe(body)
                    for topic in filters:
                        self._subscriptions.setdefault(topic, set()).add(writer)
                    writer.write(mqtt_min.build_suback(packet_id, len(filters)))
                    await writer.drain()
                elif ptype == mqtt_min.PUBLISH:
                    self._ingest(flags, body)
                elif ptype == mqtt_min.PINGREQ:
                    writer.write(mqtt_min.build_pingresp())
                    await writer.drain()
                elif ptype == mqtt_min.DISCONNECT:
                    break
        except asyncio.IncompleteReadError:
            pass  # client closed the connection
        finally:
            for subscribers in self._subscriptions.values():
                subscribers.discard(writer)
            writer.close()

    # -- ingest and egress ---------------------------------------------------

    def _ingest(self, flags: int, body: bytes) -> None:
        """Admit one client PUBLISH on t4/in into the dispatcher."""
        topic, user_props, _raw_props, payload = mqtt_min.parse_publish(flags, body)
        if topic != INGEST_TOPIC:
            return  # the prototype only schedules the fixed ingest topic
        props = dict(user_props)
        zone = int(props["zone"])
        device = props["device"]
        is_alarm = props["alarm"] == "1"
        msg_id = payload.split(b":", 1)[0].decode()
        band = self.classifier.classify(zone, is_alarm)

        handle = self._next_handle
        self._next_handle += 1
        now = time.monotonic() - self._t0

        start = time.perf_counter_ns()
        admitted = self.dispatcher.enqueue(handle, device, zone, is_alarm, now)
        enqueue_ns = time.perf_counter_ns() - start

        if not admitted:
            # AAP rate-shed: count the drop here; it is never delivered.
            self._overhead.append(self._row(msg_id, band, enqueue_ns, 0,
                                             len(self._inflight), dropped=1))
            return
        self._inflight[device] += 1
        self._registry[handle] = (zone, device, payload, msg_id, band, enqueue_ns)

    async def _transmit(self) -> None:
        """Egress coroutine paced at the configured egress rate ``C``."""
        period = 1.0 / self.egress_rate
        next_t = time.monotonic()
        while True:
            delay = next_t - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            now = time.monotonic() - self._t0

            start = time.perf_counter_ns()
            handle = self.dispatcher.select_next(now)
            select_ns = time.perf_counter_ns() - start

            if handle is None:
                next_t = time.monotonic() + period  # idle: do not bank (D12)
                continue

            zone, device, payload, msg_id, band, enqueue_ns = self._registry.pop(handle)
            # active_devices reflects contention at dispatch (includes this device).
            active_devices = len(self._inflight)
            await self._deliver(zone, payload)
            self._inflight[device] -= 1
            if self._inflight[device] == 0:
                del self._inflight[device]
            self._overhead.append(self._row(msg_id, band, enqueue_ns, select_ns,
                                            active_devices, dropped=0))
            next_t += period  # busy: bank the service slot

    async def _deliver(self, zone: int, payload: bytes) -> None:
        """Republish the opaque payload on the per-zone return topic."""
        # The return PUBLISH carries no user properties (clients match by the
        # msg_id inside the payload), keeping the no-wildcard routing minimal.
        packet = mqtt_min.build_publish(f"t4out/{zone}", b"", payload)
        for subscriber in list(self._subscriptions.get(f"t4out/{zone}", ())):
            subscriber.write(packet)
            await subscriber.drain()

    # -- bookkeeping / output ------------------------------------------------

    def _row(self, msg_id: str, band: int, enqueue_ns: int, select_ns: int,
             active_devices: int, dropped: int) -> dict:
        return {
            "msg_id": msg_id,
            "scheduler": self.scheduler,
            "scenario": self.scenario,
            "rep": self.rep,
            "band": band,
            "enqueue_ns": enqueue_ns,
            "select_ns": select_ns,
            "active_devices": active_devices,
            "dropped": dropped,
        }

    def write_overhead_csv(self) -> Optional[str]:
        """Append accumulated overhead rows; return the path written (or None).

        Append mode lets one broker process per rep accumulate into a single
        per-(scheduler, scenario) file; the ``rep`` column disambiguates rows.
        """
        if not self._overhead:
            return None
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(
            self.results_dir, f"broker_{self.scheduler}_{self.scenario}.csv")
        new_file = not os.path.exists(path)
        with open(path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self._overhead[0].keys()))
            if new_file:
                writer.writeheader()
            writer.writerows(self._overhead)
        return path

    async def serve(self, host: str, port: int) -> None:
        self._t0 = time.monotonic()
        server = await asyncio.start_server(self.handle_client, host, port)
        addr = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        print(f"broker[{self.scheduler}] listening on {addr}", flush=True)

        # Explicit signal handlers: a process backgrounded from a non-interactive
        # script inherits SIGINT=SIG_IGN, so the default KeyboardInterrupt path
        # never fires. add_signal_handler overrides that and lets the run scripts
        # stop the broker cleanly (flushing the overhead CSV) with SIGTERM/SIGINT.
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        transmitter = asyncio.create_task(self._transmit())
        try:
            async with server:
                await stop.wait()
        finally:
            transmitter.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description="TRIAGE/4 prototype MQTT broker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--scheduler", choices=SCHEDULERS, default="triage4")
    parser.add_argument("--rate-c", type=float, default=20.0,
                        help="broker egress rate C (msg/s) — the saturation knob")
    parser.add_argument("--scenario", default="adhoc")
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-aap", action="store_true",
                        help="disable Adaptive Alarm Protection for TRIAGE/4")
    args = parser.parse_args()

    config = build_config(enable_alarm_protection=not args.no_aap)
    dispatcher = build_dispatcher(args.scheduler, config)
    classifier = build_classifier(config)
    broker = Broker(dispatcher, classifier, args.scheduler, args.scenario,
                    args.rep, args.rate_c, args.results_dir)
    try:
        asyncio.run(broker.serve(args.host, args.port))
    except KeyboardInterrupt:
        pass
    finally:
        path = broker.write_overhead_csv()
        if path:
            print(f"overhead -> {path}", flush=True)


if __name__ == "__main__":
    main()

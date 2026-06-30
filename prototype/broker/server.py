"""
Asyncio MQTT 5.0 broker.

M0 stage (this file): a bare forwarding broker that proves the hand-rolled
MQTT 5.0 framing in ``mqtt_min`` interoperates with a real paho-mqtt v5 client —
CONNECT/CONNACK, SUBSCRIBE/SUBACK, PUBLISH (exact-topic forward), PINGREQ/
PINGRESP. There is no scheduling here yet; messages publish straight through to
matching subscribers.

A single asyncio event loop means client handlers never run truly concurrently
across an await-free critical section, so the routing table needs no locks.

Part D will insert the egress path at the marked seam: on PUBLISH, read the
zone/device/alarm user properties and call ``dispatcher.enqueue(...)`` with a
relative clock (D10) instead of forwarding immediately; a separate transmitter
coroutine paced at rate ``C`` will call ``dispatcher.select_next(...)`` and emit
on ``t4out/<zone>``.
"""

import argparse
import asyncio
from typing import Dict, Set

from . import mqtt_min


class Broker:
    """Minimal exact-topic MQTT 5.0 broker (M0 forwarding stage)."""

    def __init__(self) -> None:
        # topic filter -> set of subscriber writers (exact match only).
        self._subscriptions: Dict[str, Set[asyncio.StreamWriter]] = {}

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
                    await self._on_publish(flags, body)
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

    async def _on_publish(self, flags: int, body: bytes) -> None:
        # === Part D seam: dispatcher.enqueue(...) goes here ===
        # M0 forwards the PUBLISH verbatim to every exact-topic subscriber,
        # preserving properties (incl. user properties) and the opaque payload.
        topic, _user_props, raw_props, payload = mqtt_min.parse_publish(flags, body)
        packet = mqtt_min.build_publish(topic, raw_props, payload)
        for subscriber in list(self._subscriptions.get(topic, ())):
            subscriber.write(packet)
            await subscriber.drain()

    async def serve(self, host: str, port: int) -> None:
        server = await asyncio.start_server(self.handle_client, host, port)
        addr = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        print(f"broker listening on {addr}", flush=True)
        async with server:
            await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal MQTT 5.0 broker (M0)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1884)
    args = parser.parse_args()
    try:
        asyncio.run(Broker().serve(args.host, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

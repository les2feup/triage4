"""
Per-zone receiver (in-process helper, not an entry point).

Collects messages delivered on this client's return topic(s) and computes
``RTT = time.monotonic_ns() - t_send`` using THIS process's clock — the same
clock ``loadgen`` stamped ``t_send`` with — so RTT needs no cross-host sync.

A client may share a return topic with other zones (when several geographic
zones map to the same return key), so the receiver records RTT only for the
msg_ids it actually sent; foreign messages are ignored. This keeps RTT
single-clock even when subscriptions overlap.
"""

import threading
import time
from typing import Dict, Set


class RttReceiver:
    """Records single-clock RTT for a known set of sent msg_ids."""

    def __init__(self, expected_ids: Set[str]) -> None:
        self._expected = expected_ids
        self._records: Dict[str, dict] = {}
        self.complete = threading.Event()
        if not expected_ids:
            self.complete.set()

    def on_message(self, client, userdata, message) -> None:
        t_recv = time.monotonic_ns()
        # Payload is <msg_id>:<t_send_ns>:<filler>; take the first two fields and
        # ignore the constant padding after them.
        parts = message.payload.split(b":", 2)
        msg_id = parts[0].decode()
        if msg_id not in self._expected or msg_id in self._records:
            return  # foreign or duplicate delivery
        t_send_ns = int(parts[1])
        self._records[msg_id] = {
            "msg_id": msg_id,
            "t_send_ns": t_send_ns,
            "t_recv_ns": t_recv,
            "rtt_ms": (t_recv - t_send_ns) / 1e6,
        }
        if len(self._records) >= len(self._expected):
            self.complete.set()

    @property
    def records(self) -> Dict[str, dict]:
        return self._records

    @property
    def received_count(self) -> int:
        return len(self._records)

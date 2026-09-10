"""Regression tests for prototype broker subscriber isolation."""

import asyncio

import pytest

from prototype.broker.server import Broker


class FakeWriter:
    def __init__(self, *, fail=False, delay=0.0):
        self.fail = fail
        self.delay = delay
        self.writes = []
        self.closed = False

    def write(self, packet):
        if self.fail:
            raise ConnectionResetError("subscriber closed")
        self.writes.append(packet)

    async def drain(self):
        if self.fail:
            raise ConnectionResetError("subscriber closed")
        if self.delay:
            await asyncio.sleep(self.delay)

    def close(self):
        self.closed = True


class FailingDispatcher:
    def select_next(self, now):
        raise RuntimeError("internal dispatcher failure")


@pytest.fixture
def broker():
    return Broker(
        dispatcher=object(),
        classifier=object(),
        scheduler="triage4",
        scenario="adhoc",
        rep=0,
        egress_rate=20.0,
        results_dir="/tmp/triage4-test-results",
    )


def test_forward_isolates_failed_subscriber(broker):
    failed = FakeWriter(fail=True)
    healthy = FakeWriter()
    broker._subscriptions["t4out/0"] = {failed, healthy}

    asyncio.run(broker._forward("t4out/0", b"message"))

    assert len(healthy.writes) == 1
    assert failed.closed
    assert failed not in broker._subscriptions["t4out/0"]
    assert healthy in broker._subscriptions["t4out/0"]


def test_forward_times_out_slow_subscriber_and_continues(broker):
    broker.DRAIN_TIMEOUT_SECONDS = 0.01
    slow = FakeWriter(delay=0.1)
    healthy = FakeWriter()
    broker._subscriptions["t4out/0"] = {slow, healthy}

    asyncio.run(broker._forward("t4out/0", b"message"))

    assert len(healthy.writes) == 1
    assert slow.closed
    assert slow not in broker._subscriptions["t4out/0"]
    assert healthy in broker._subscriptions["t4out/0"]


def test_transmitter_failure_is_observable():
    broker = Broker(
        dispatcher=FailingDispatcher(),
        classifier=object(),
        scheduler="triage4",
        scenario="adhoc",
        rep=0,
        egress_rate=1000.0,
        results_dir="/tmp/triage4-test-results",
    )
    broker._t0 = 0.0

    async def run_transmitter():
        await broker._transmit()

    with pytest.raises(RuntimeError, match="internal dispatcher failure"):
        asyncio.run(run_transmitter())

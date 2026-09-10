"""Regression tests for prototype input and source-state limits."""

import asyncio

import pytest

from prototype.broker import mqtt_min
from triage4.source_rate_limiter import SourceRateLimiter


class Reader:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    async def readexactly(self, size):
        if not self.chunks:
            raise asyncio.IncompleteReadError(b"", size)
        chunk = self.chunks.pop(0)
        if len(chunk) != size:
            raise asyncio.IncompleteReadError(chunk, size)
        return chunk


@pytest.mark.parametrize("remaining_length", [b"\x80\x80\x41", b"\x80\x80\x80\x01"])
def test_read_packet_rejects_oversized_remaining_length(remaining_length):
    with pytest.raises(ValueError, match="maximum size|malformed"):
        chunks = [b"\x30"] + [bytes([byte]) for byte in remaining_length]
        asyncio.run(mqtt_min.read_packet(Reader(chunks)))


def test_decode_utf8_rejects_truncated_value():
    with pytest.raises(ValueError, match="truncated UTF-8"):
        mqtt_min.decode_utf8(b"\x00\x05x", 0)


def test_source_limiter_expires_inactive_sources():
    limiter = SourceRateLimiter(10.0, 5.0, 4.0, 3, 1, 1.0, 1)
    limiter.admit(0.0, "source-a")
    assert limiter.tracked_sources == 1
    limiter.admit(11.0, "source-b")
    assert limiter.tracked_sources == 1

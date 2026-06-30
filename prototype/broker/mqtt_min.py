"""
Minimal MQTT 5.0 wire codec.

Implements just the subset the prototype broker needs (plan §6): the fixed
header, the v5 property field, and the CONNECT/CONNACK, SUBSCRIBE/SUBACK,
PUBLISH (QoS 0), and PINGREQ/PINGRESP packets. QoS 1/2, retained messages,
wildcards, will, and unhandled property identifiers are ignored. No third-party
dependencies — pure bytes.

The property parser knows each identifier's wire type so it can skip properties
it does not interpret and still locate User Properties (0x26), which carry the
zone/device/alarm routing metadata. For forwarding, the raw property bytes are
preserved verbatim so a delivered PUBLISH reproduces the publisher's properties
exactly (the payload, carrying the client's t_send, is likewise untouched).
"""

import asyncio
from typing import Dict, List, Tuple

# MQTT control packet type codes (high nibble of the fixed header byte).
CONNECT = 1
CONNACK = 2
PUBLISH = 3
SUBSCRIBE = 8
SUBACK = 9
PINGREQ = 12
PINGRESP = 13
DISCONNECT = 14

# Wire type of each MQTT 5.0 property identifier, used to skip over properties
# the broker does not interpret while still parsing the ones it needs.
_PROP_TYPE: Dict[int, str] = {
    0x01: "byte", 0x02: "int4", 0x03: "utf8", 0x08: "utf8", 0x09: "bin",
    0x0B: "varint", 0x11: "int4", 0x12: "utf8", 0x13: "int2", 0x15: "utf8",
    0x16: "bin", 0x17: "byte", 0x18: "int4", 0x19: "byte", 0x1A: "utf8",
    0x1C: "utf8", 0x1F: "utf8", 0x21: "int2", 0x22: "int2", 0x23: "int2",
    0x24: "byte", 0x25: "byte", 0x26: "utf8pair", 0x27: "int4", 0x28: "byte",
    0x29: "byte", 0x2A: "byte",
}


def encode_varint(value: int) -> bytes:
    """Encode a Variable Byte Integer (remaining length / property length)."""
    out = bytearray()
    while True:
        byte = value % 128
        value //= 128
        if value:
            byte |= 0x80
        out.append(byte)
        if not value:
            return bytes(out)


def decode_varint(buf: bytes, offset: int) -> Tuple[int, int]:
    """Decode a Variable Byte Integer at ``offset``; return (value, new_offset)."""
    multiplier = 1
    value = 0
    while True:
        byte = buf[offset]
        offset += 1
        value += (byte & 0x7F) * multiplier
        if not (byte & 0x80):
            return value, offset
        multiplier *= 128
        if multiplier > 128 ** 3:
            raise ValueError("malformed variable byte integer")


def encode_utf8(text: str) -> bytes:
    """Encode an MQTT UTF-8 string (2-byte big-endian length prefix + bytes)."""
    raw = text.encode("utf-8")
    return len(raw).to_bytes(2, "big") + raw


def decode_utf8(buf: bytes, offset: int) -> Tuple[str, int]:
    """Decode a length-prefixed UTF-8 string; return (text, new_offset)."""
    length = int.from_bytes(buf[offset:offset + 2], "big")
    offset += 2
    text = buf[offset:offset + length].decode("utf-8")
    return text, offset + length


def _skip_property_value(buf: bytes, offset: int, wire: str) -> int:
    """Advance ``offset`` past one property value of the given wire type."""
    if wire == "byte":
        return offset + 1
    if wire == "int2":
        return offset + 2
    if wire == "int4":
        return offset + 4
    if wire == "varint":
        _, offset = decode_varint(buf, offset)
        return offset
    if wire in ("utf8", "bin"):
        length = int.from_bytes(buf[offset:offset + 2], "big")
        return offset + 2 + length
    raise ValueError(f"unknown property wire type {wire!r}")


def parse_user_properties(props: bytes) -> List[Tuple[str, str]]:
    """Extract all User Property (0x26) key/value pairs from a property block."""
    pairs: List[Tuple[str, str]] = []
    offset = 0
    while offset < len(props):
        identifier, offset = decode_varint(props, offset)
        if identifier == 0x26:
            key, offset = decode_utf8(props, offset)
            value, offset = decode_utf8(props, offset)
            pairs.append((key, value))
        else:
            wire = _PROP_TYPE.get(identifier)
            if wire is None:
                raise ValueError(f"unhandled property id 0x{identifier:02x}")
            offset = _skip_property_value(props, offset, wire)
    return pairs


def encode_user_properties(pairs: List[Tuple[str, str]]) -> bytes:
    """Encode User Property key/value pairs as a property block (no length prefix)."""
    out = bytearray()
    for key, value in pairs:
        out += b"\x26" + encode_utf8(key) + encode_utf8(value)
    return bytes(out)


async def read_packet(reader: asyncio.StreamReader) -> Tuple[int, int, bytes]:
    """Read one complete control packet; return (packet_type, flags, body bytes)."""
    first = (await reader.readexactly(1))[0]
    packet_type = first >> 4
    flags = first & 0x0F
    # Remaining Length is a varint read byte-by-byte off the stream.
    multiplier = 1
    length = 0
    while True:
        byte = (await reader.readexactly(1))[0]
        length += (byte & 0x7F) * multiplier
        if not (byte & 0x80):
            break
        multiplier *= 128
        if multiplier > 128 ** 3:
            raise ValueError("malformed remaining length")
    body = await reader.readexactly(length) if length else b""
    return packet_type, flags, body


def parse_subscribe(body: bytes) -> Tuple[int, List[str]]:
    """Parse SUBSCRIBE; return (packet_id, [topic_filter, ...])."""
    packet_id = int.from_bytes(body[0:2], "big")
    prop_len, offset = decode_varint(body, 2)
    offset += prop_len  # SUBSCRIBE properties are not used by this broker
    filters: List[str] = []
    while offset < len(body):
        topic, offset = decode_utf8(body, offset)
        offset += 1  # subscription options byte
        filters.append(topic)
    return packet_id, filters


def parse_publish(flags: int, body: bytes) -> Tuple[str, List[Tuple[str, str]], bytes, bytes]:
    """Parse a QoS-0 PUBLISH; return (topic, user_properties, raw_props, payload).

    ``raw_props`` is the property block verbatim so it can be re-emitted on
    delivery without re-encoding (preserves any properties the broker ignores).
    """
    qos = (flags >> 1) & 0x03
    topic, offset = decode_utf8(body, 0)
    if qos > 0:
        offset += 2  # packet identifier (present only for QoS > 0)
    prop_len, offset = decode_varint(body, offset)
    raw_props = body[offset:offset + prop_len]
    offset += prop_len
    payload = body[offset:]
    return topic, parse_user_properties(raw_props), raw_props, payload


def _packet(header_byte: int, body: bytes) -> bytes:
    """Frame a packet: header byte + remaining-length varint + body."""
    return bytes([header_byte]) + encode_varint(len(body)) + body


def build_connack() -> bytes:
    """CONNACK v5: session-present=0, reason=success(0x00), empty properties."""
    return _packet(0x20, bytes([0x00, 0x00, 0x00]))


def build_suback(packet_id: int, count: int) -> bytes:
    """SUBACK v5: echo packet id, empty properties, granted-QoS-0 per filter."""
    body = packet_id.to_bytes(2, "big") + b"\x00" + b"\x00" * count
    return _packet(0x90, body)


def build_publish(topic: str, raw_props: bytes, payload: bytes) -> bytes:
    """PUBLISH v5 QoS 0 with the given (already-encoded) property block."""
    variable_header = encode_utf8(topic) + encode_varint(len(raw_props)) + raw_props + payload
    return _packet(0x30, variable_header)


def build_pingresp() -> bytes:
    """PINGRESP: fixed header only."""
    return _packet(0xD0, b"")

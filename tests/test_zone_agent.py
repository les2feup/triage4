"""Regression tests for zone-agent control validation."""

import json

import pytest

pytest.importorskip("paho.mqtt.client")

from prototype.clients.zone_agent import ZoneAgent


class Message:
    def __init__(self, payload):
        self.payload = payload


def test_on_go_ignores_malformed_cell():
    agent = ZoneAgent(0, "/tmp/workloads", "/tmp/results", 1.0)

    agent.on_go(None, None, Message(b"triage4:broken"))

    assert agent.cell is None
    assert not agent.go.is_set()


def test_load_messages_rejects_path_traversal(tmp_path):
    agent = ZoneAgent(0, str(tmp_path / "workloads"), str(tmp_path / "results"), 1.0)

    with pytest.raises(ValueError, match="invalid path"):
        agent.load_messages("../outside")


def test_load_messages_requires_message_list(tmp_path):
    workloads = tmp_path / "workloads"
    workloads.mkdir()
    (workloads / "scenario.json").write_text(json.dumps({"messages": {}}))
    agent = ZoneAgent(0, str(workloads), str(tmp_path / "results"), 1.0)

    with pytest.raises(ValueError, match="messages"):
        agent.load_messages("scenario")

"""
Broker run configuration: scheduler selection and the TRIAGE/4 config.

Builds the online dispatcher for a single run (one scheduler per process, so
token/AAP state never leaks across runs). The three dispatchers share the
``enqueue/select_next/is_empty`` contract, so the broker treats them uniformly.

The broker egress rate ``C`` (the saturation knob) is configured separately on
the command line and is INDEPENDENT of ``TRIAGE4Config.service_rate`` — the
online dispatcher ignores service_rate; the transmitter paces delivery at ``C``.
"""

from typing import Union

from triage4 import (
    BandClassifier,
    TRIAGE4Config,
    Triage4EgressDispatcher,
)

from .dispatchers import (
    FifoEgressDispatcher,
    StrictEgressDispatcher,
    WfqEgressDispatcher,
)

SCHEDULERS = ("fifo", "strict", "wfq", "triage4")

Dispatcher = Union[
    FifoEgressDispatcher,
    StrictEgressDispatcher,
    WfqEgressDispatcher,
    Triage4EgressDispatcher,
]


def build_config(enable_alarm_protection: bool = True) -> TRIAGE4Config:
    """TRIAGE/4 configuration for the prototype runs.

    Part E pins the exact C3/R3 configuration used by the simulation; this
    default mirrors the package defaults with AAP toggled per the run (AAP is
    enabled for TRIAGE/4 on hardware, required for the R3 no-shed confirmation).
    """
    return TRIAGE4Config(enable_alarm_protection=enable_alarm_protection)


def build_dispatcher(name: str, config: TRIAGE4Config) -> Dispatcher:
    """Construct the online dispatcher selected by ``name``."""
    if name == "fifo":
        return FifoEgressDispatcher()
    if name == "strict":
        return StrictEgressDispatcher()
    if name == "wfq":
        return WfqEgressDispatcher()
    if name == "triage4":
        return Triage4EgressDispatcher(config)
    raise ValueError(f"unknown scheduler {name!r}; choose from {SCHEDULERS}")


def build_classifier(config: TRIAGE4Config) -> BandClassifier:
    """Classifier used only to label each message's band in the overhead CSV.

    Band labelling is independent of the scheduler in use — it characterises the
    message, so it is recorded uniformly across fifo/strict/triage4 runs.
    """
    return BandClassifier(
        high_zone_max=config.high_zone_max,
        standard_zone_max=config.standard_zone_max,
    )

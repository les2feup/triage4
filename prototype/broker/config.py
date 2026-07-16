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
    DrrEgressDispatcher,
    FifoEgressDispatcher,
    StrictEgressDispatcher,
    TbpEgressDispatcher,
    WfqEgressDispatcher,
)

# Full parity with the simulation's baseline set (Table 1): no scheduler is
# measured in simulation but withheld from hardware. ``t4-nosourcelimit`` is the
# T4-NoSourceLimit ablation — TRIAGE/4 with AAP still on but reduced to the
# band-global backstop alone. It is the reference arm for the R1/R2 containment
# claim: "no legitimate alarms shed" means little without a comparable
# configuration that does shed them.
SCHEDULERS = ("fifo", "strict", "wfq", "drr", "tbp", "triage4", "t4-nosourcelimit")

# Arms that run TRIAGE/4 with Adaptive Alarm Protection enabled. Every other
# scheduler has no AAP to speak of and is launched with --no-aap.
AAP_SCHEDULERS = ("triage4", "t4-nosourcelimit")

Dispatcher = Union[
    DrrEgressDispatcher,
    FifoEgressDispatcher,
    StrictEgressDispatcher,
    TbpEgressDispatcher,
    WfqEgressDispatcher,
    Triage4EgressDispatcher,
]


def build_config(scheduler: str, enable_alarm_protection: bool = True) -> TRIAGE4Config:
    """TRIAGE/4 configuration for the prototype runs.

    Part E pins the exact configuration used by the simulation; this default
    mirrors the package defaults with AAP toggled per the run (AAP is enabled
    for TRIAGE/4 on hardware, required for the R3 no-shed confirmation).

    The ``t4-nosourcelimit`` arm keeps AAP enabled and strips only the
    per-source layer, leaving the band-global backstop to decide alone. Both
    arms therefore differ in exactly one mechanism, which is what lets the
    R1/R2 comparison attribute the difference to that mechanism.
    """
    return TRIAGE4Config(
        enable_alarm_protection=enable_alarm_protection,
        disable_source_rate_limit=(scheduler == "t4-nosourcelimit"),
    )


def build_dispatcher(name: str, config: TRIAGE4Config) -> Dispatcher:
    """Construct the online dispatcher selected by ``name``."""
    if name == "fifo":
        return FifoEgressDispatcher()
    if name == "strict":
        return StrictEgressDispatcher()
    if name == "wfq":
        return WfqEgressDispatcher()
    if name == "drr":
        return DrrEgressDispatcher()
    if name == "tbp":
        return TbpEgressDispatcher(config)
    if name in ("triage4", "t4-nosourcelimit"):
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

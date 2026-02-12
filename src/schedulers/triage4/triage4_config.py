"""Compatibility wrapper for legacy TRIAGE/4 config imports."""

from triage4.triage4_config import (
    TRIAGE4Config,
    create_triage4_custom,
    create_triage4_default,
)

__all__ = ["TRIAGE4Config", "create_triage4_default", "create_triage4_custom"]

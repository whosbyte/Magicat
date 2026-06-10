# magicat/config.py
"""Environment-driven settings. Read at call time (never import time) so
tests can monkeypatch the environment.

Contract: this module is shared infrastructure - it must NEVER import from
magicat.modules.* (modules import config, not the other way around).
"""
from __future__ import annotations

import os

ACQUISITION_POLICIES = ("always", "licensed_only", "link_only")


def acquisition_policy() -> str:
    policy = os.environ.get("MAGICAT_ACQUISITION_POLICY", "always")
    if policy not in ACQUISITION_POLICIES:
        raise ValueError(
            f"MAGICAT_ACQUISITION_POLICY must be one of "
            f"{ACQUISITION_POLICIES}, got {policy!r}")
    return policy

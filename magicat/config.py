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


def music_timeout_s() -> float:
    """Wall-clock budget for music identification (CEO decision 2026-06-11:
    if no match within this budget, skip the layer and move on)."""
    raw = os.environ.get("MAGICAT_MUSIC_TIMEOUT_S", "20")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"MAGICAT_MUSIC_TIMEOUT_S must be a number, got {raw!r}")
    if value <= 0:
        raise ValueError(f"MAGICAT_MUSIC_TIMEOUT_S must be > 0, got {raw!r}")
    return value


def ingest_timeout_s() -> float:
    """Wall-clock budget for the source video download (yt-dlp). socket_timeout
    cannot bound a throttled-but-dribbling YouTube DASH stream; this watchdog
    is the only reliable total abort. Ingest is fatal-on-failure, so on expiry
    the job fails with an actionable error instead of hanging forever."""
    raw = os.environ.get("MAGICAT_INGEST_TIMEOUT_S", "120")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"MAGICAT_INGEST_TIMEOUT_S must be a number, got {raw!r}")
    if value <= 0:
        raise ValueError(f"MAGICAT_INGEST_TIMEOUT_S must be > 0, got {raw!r}")
    return value


def acquisition_timeout_s() -> float:
    """Wall-clock budget for a single music-acquisition download (yt-dlp).
    On expiry the candidate download aborts and the acquisition layer degrades
    to 'failed' (per-resolver, non-fatal) instead of hanging forever."""
    raw = os.environ.get("MAGICAT_ACQUISITION_TIMEOUT_S", "90")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"MAGICAT_ACQUISITION_TIMEOUT_S must be a number, got {raw!r}")
    if value <= 0:
        raise ValueError(
            f"MAGICAT_ACQUISITION_TIMEOUT_S must be > 0, got {raw!r}")
    return value

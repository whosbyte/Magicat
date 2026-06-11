# magicat/core/ffmpeg.py
"""Shared ffmpeg invocation helper - one place for the standard flags."""
from __future__ import annotations

import json
import subprocess


def run_ffmpeg(args: list[str], timeout_s: float = 300) -> None:
    """Run ffmpeg with the standard flags, bounded by timeout_s.

    A hung ffmpeg (corrupt input, stuck filter graph) would otherwise block
    forever; on expiry subprocess.TimeoutExpired propagates - analyzers and
    exporters already degrade per-layer, ingest normalize stays fatal.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True, capture_output=True, timeout=timeout_s,
    )


def run_ffprobe(path, entries: str, timeout_s: float = 300) -> dict:
    """ffprobe -show_entries wrapper returning parsed JSON.

    entries: ffprobe -show_entries value, e.g. "format=duration" or
    "stream=r_frame_rate,width,height" (multiple groups joined with ':').
    Bounded by timeout_s; a TimeoutExpired propagates on a hung probe.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entries,
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=timeout_s,
    ).stdout
    return json.loads(out)

# magicat/core/ffmpeg.py
"""Shared ffmpeg invocation helper - one place for the standard flags."""
from __future__ import annotations

import json
import subprocess


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True, capture_output=True,
    )


def run_ffprobe(path, entries: str) -> dict:
    """ffprobe -show_entries wrapper returning parsed JSON.

    entries: ffprobe -show_entries value, e.g. "format=duration" or
    "stream=r_frame_rate,width,height" (multiple groups joined with ':').
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entries,
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)

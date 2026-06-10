# magicat/core/ffmpeg.py
"""Shared ffmpeg invocation helper - one place for the standard flags."""
from __future__ import annotations

import subprocess


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True, capture_output=True,
    )

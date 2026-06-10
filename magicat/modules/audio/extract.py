# magicat/modules/audio/extract.py
"""Audio extraction and sliding-window cutting for music fingerprinting.

Windows are 12s every 10s (2s overlap): AudD analyzes <=~20s and ACRCloud
recommends <=15s, so 12s mono WAV clips sit comfortably inside both caps.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from magicat.core.ffmpeg import run_ffmpeg


def wav_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def extract_wav(video: Path, dest: Path, sample_rate: int = 44100) -> Path:
    """Full audio track as stereo WAV (44.1kHz - what demucs expects too)."""
    run_ffmpeg(["-i", str(video), "-vn", "-ac", "2", "-ar", str(sample_rate),
                str(dest)])
    return dest


@dataclass
class AudioWindow:
    t_start: float          # seconds into the video
    path: Path


def cut_windows(wav: Path, out_dir: Path, window_s: float = 12.0,
                stride_s: float = 10.0, max_windows: int = 5,
                min_window_s: float = 3.0) -> list[AudioWindow]:
    """Cut mono fingerprinting windows. Mono halves upload size; both
    providers fingerprint mono fine."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = wav_duration(wav)
    windows: list[AudioWindow] = []
    t = 0.0
    while t < duration and len(windows) < max_windows:
        remaining = duration - t
        if remaining < min_window_s and windows:
            break  # tail too short to fingerprint reliably
        clip = out_dir / f"win_{t:08.3f}.wav"
        run_ffmpeg(["-ss", f"{t:.3f}", "-t", f"{window_s:.3f}", "-i", str(wav),
                    "-ac", "1", str(clip)])
        windows.append(AudioWindow(t_start=t, path=clip))
        t += stride_s
    return windows

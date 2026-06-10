# magicat/modules/captions/sampling.py
"""Sample frames for OCR at a fixed rate (spec section 6.5 step 1)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from magicat.core.ffmpeg import run_ffmpeg


@dataclass
class FrameSample:
    t: float          # seconds into the video (frame n -> n / fps)
    path: Path


def sample_frames(video: Path, out_dir: Path,
                  fps: float = 5.0) -> list[FrameSample]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%05d.jpg"
    run_ffmpeg(["-i", str(video), "-vf", f"fps={fps}", "-q:v", "3",
                str(pattern)])
    frames = sorted(out_dir.glob("frame_*.jpg"))
    # ffmpeg numbers from 1; frame N samples the source around (N-1)/fps
    return [FrameSample(t=i / fps, path=p) for i, p in enumerate(frames)]

# tests/conftest.py
import json
import subprocess
from pathlib import Path

import pytest


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True, capture_output=True,
    )


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


@pytest.fixture(scope="session")
def fixture_video(tmp_path_factory) -> Path:
    """6s, 320x640@30fps: red 0-2s, green 2-4s, blue 4-6s. Cuts at 2.0, 4.0."""
    work = tmp_path_factory.mktemp("fixture")
    segments = []
    for i, color in enumerate(["red", "green", "blue"]):
        seg = work / f"seg{i}.mp4"
        run_ffmpeg([
            "-f", "lavfi", "-i", f"color=c={color}:s=320x640:r=30:d=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(seg),
        ])
        segments.append(seg)
    concat_list = work / "list.txt"
    concat_list.write_text(
        "".join(f"file '{s.as_posix()}'\n" for s in segments), encoding="ascii"
    )
    out = work / "fixture.mp4"
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-c", "copy", str(out)])
    return out


@pytest.fixture(autouse=True)
def _isolated_magicat_env(monkeypatch):
    """Tests never see ambient Magicat/provider configuration; tests that
    need a var set it explicitly via monkeypatch.setenv (composes fine)."""
    for var in ("AUDD_API_TOKEN", "ACR_HOST", "ACR_ACCESS_KEY",
                "ACR_ACCESS_SECRET", "MAGICAT_MUSIC_PROVIDER",
                "MAGICAT_ACQUISITION_POLICY", "MAGICAT_USE_SEPARATION"):
        monkeypatch.delenv(var, raising=False)

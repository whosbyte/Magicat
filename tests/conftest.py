# tests/conftest.py
import json
import subprocess
from pathlib import Path

import pytest

from magicat.core.ffmpeg import run_ffmpeg


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


@pytest.fixture(scope="session")
def long_wav(tmp_path_factory) -> Path:
    """25s mono sine WAV - long enough to produce 3 sliding windows."""
    out = tmp_path_factory.mktemp("audio") / "long.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=25",
                "-ac", "1", str(out)])
    return out


@pytest.fixture(autouse=True)
def _isolated_magicat_env(monkeypatch):
    """Tests never see ambient Magicat/provider configuration; tests that
    need a var set it explicitly via monkeypatch.setenv (composes fine)."""
    for var in ("AUDD_API_TOKEN", "ACR_HOST", "ACR_ACCESS_KEY",
                "ACR_ACCESS_SECRET", "MAGICAT_MUSIC_PROVIDER",
                "MAGICAT_ACQUISITION_POLICY", "MAGICAT_USE_SEPARATION",
                "MAGICAT_API_KEY", "SERPAPI_KEY", "MAGICAT_PUBLIC_BASE_URL",
                "GOOGLE_VISION_API_KEY", "MAGICAT_RIS_PROVIDER",
                "MAGICAT_CAPCUT_EXPORT"):
        monkeypatch.delenv(var, raising=False)


WINDOWS_FONT = Path("C:/Windows/Fonts/arial.ttf")


@pytest.fixture(scope="session")
def caption_video(tmp_path_factory) -> Path:
    """6s 480x854 dark clip with two burned captions at known times/positions:
    'HELLO WORLD' t=1.0-3.0 and 'SECOND LINE' t=3.5-5.2, both bottom-center.
    """
    if not WINDOWS_FONT.is_file():
        pytest.skip("test font not available")
    out = tmp_path_factory.mktemp("captions") / "captions.mp4"
    fontfile = "C\\:/Windows/Fonts/arial.ttf"
    draw1 = (f"drawtext=fontfile='{fontfile}':text='HELLO WORLD'"
             ":fontsize=42:fontcolor=white:x=(w-text_w)/2:y=h-150"
             ":enable='between(t,1,3)'")
    draw2 = (f"drawtext=fontfile='{fontfile}':text='SECOND LINE'"
             ":fontsize=42:fontcolor=white:x=(w-text_w)/2:y=h-150"
             ":enable='between(t,3.5,5.2)'")
    run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=0x202020:s=480x854:r=25:d=6",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-vf", f"{draw1},{draw2}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", str(out),
    ])
    return out

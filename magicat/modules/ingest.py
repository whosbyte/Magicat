# magicat/modules/ingest.py
"""Ingest: fetch (yt-dlp) or copy the input, normalize to H.264 MP4, probe."""
from __future__ import annotations

import subprocess
from pathlib import Path

from magicat import config
from magicat.core.download_guard import DownloadTimeout, timeout_hook
from magicat.core.ffmpeg import run_ffprobe
from magicat.core.registry import register_analyzer
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest

PLATFORMS = {
    "tiktok.com": "tiktok",
    "instagram.com": "instagram",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
}


def detect_platform(url: str) -> str | None:
    for domain, name in PLATFORMS.items():
        if domain in url:
            return name
    return None


def download(url: str, dest_dir: Path) -> Path:
    """Fetch a video with yt-dlp; returns the downloaded file path.

    Guarded against the freeze-forever case: socket_timeout bounds a single
    stalled read, but a throttled YouTube DASH stream keeps dribbling bytes so
    no read ever times out. The progress_hooks watchdog enforces a wall-clock
    budget - the only reliable total abort (verified empirically). The hook's
    DownloadTimeout propagates as-is from the file downloader; we also catch
    DownloadError defensively in case an extraction-phase failure wraps it.
    """
    import yt_dlp
    from yt_dlp.utils import DownloadError

    budget = config.ingest_timeout_s()
    template = str(dest_dir / "download.%(ext)s")
    ydl_opts = {
        "outtmpl": template,
        "format": "mp4/best",
        "quiet": True,
        "socket_timeout": 20,
        "retries": 3,
        "fragment_retries": 3,
        "progress_hooks": [timeout_hook(budget)],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info))
    except (DownloadTimeout, DownloadError) as exc:
        # DownloadTimeout: our watchdog fired (the common case). DownloadError
        # may wrap a DownloadTimeout raised on a non-file-downloader path.
        timed_out = isinstance(exc, DownloadTimeout) or isinstance(
            getattr(exc, "__cause__", None) or getattr(exc, "__context__", None),
            DownloadTimeout)
        if timed_out:
            raise RuntimeError(
                f"video download timed out after {budget:.0f}s - likely "
                "YouTube throttling; install a JavaScript runtime "
                "(winget install DenoLand.Deno) - see README") from exc
        raise


def normalize(src: Path, dest: Path) -> None:
    """Re-encode to H.264/AAC MP4 so every later module sees one format."""
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(dest)],
        check=True, capture_output=True,
    )


def probe(path: Path) -> dict:
    data = run_ffprobe(
        path, "stream=r_frame_rate,width,height:format=duration")
    streams = [s for s in data.get("streams", []) if "width" in s]
    if not streams:
        raise ValueError(f"no video stream in {path}")
    stream = streams[0]
    num, den = (float(x) for x in stream["r_frame_rate"].split("/"))
    return {
        "fps": num / den if den else 0.0,
        "resolution": f"{stream['width']}x{stream['height']}",
        "duration": float(data["format"]["duration"]),
    }


@register_analyzer
class IngestAnalyzer:
    name = "ingest"
    layer = "source"
    needs_gpu = False
    downloader = staticmethod(download)  # injectable for tests

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        src = manifest.source
        if src.url:
            raw = self.downloader(src.url, ws.media_dir)
            platform = detect_platform(src.url)
        elif src.file:
            raw = Path(src.file)
            if not raw.is_file():
                raise FileNotFoundError(f"input file not found: {raw}")
            platform = None
        else:
            raise ValueError("manifest.source needs url or file")

        normalized = ws.media_dir / "source.mp4"
        normalize(raw, normalized)
        meta = probe(normalized)
        return {
            "source": {
                "url": src.url,
                "platform": platform,
                "file": str(normalized),
                **meta,
            },
            "layers_status": {"source": "ok"},
        }

# magicat/modules/ingest.py
"""Ingest: fetch (yt-dlp) or copy the input, normalize to H.264 MP4, probe."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

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
    """Fetch a video with yt-dlp; returns the downloaded file path."""
    import yt_dlp

    template = str(dest_dir / "download.%(ext)s")
    with yt_dlp.YoutubeDL({"outtmpl": template, "format": "mp4/best",
                           "quiet": True}) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))


def normalize(src: Path, dest: Path) -> None:
    """Re-encode to H.264/AAC MP4 so every later module sees one format."""
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(dest)],
        check=True, capture_output=True,
    )


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,width,height",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "fps": float(num) / float(den),
        "resolution": f"{stream['width']}x{stream['height']}",
        "duration": float(data["format"]["duration"]),
    }


@register_analyzer
class IngestAnalyzer:
    name = "ingest"
    needs_gpu = False
    downloader = staticmethod(download)  # injectable for tests

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        src = manifest.source
        if src.url:
            raw = self.downloader(src.url, ws.media_dir)
            platform = detect_platform(src.url)
        elif src.file:
            raw = Path(src.file)
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

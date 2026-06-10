# magicat/modules/audio/acquire.py
"""Acquire the identified song per policy (spec section 6.4).

Resolver chain: scsearch1 (SoundCloud) -> ytsearch1 (YouTube), two-phase:
probe with extract_info(download=False) and validate (fuzzy title, duration
tolerance, no preview-only formats), then download per policy:
  always         download whatever validated (CEO launch policy - legal
                 review flagged; one env var flips this)
  licensed_only  download only Creative-Commons-licensed candidates
  link_only      never download; persist links only
"""
from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path

from pydantic import BaseModel

from magicat import config
from magicat.core.ffmpeg import run_ffmpeg
from magicat.core.registry import register_analyzer
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest

log = logging.getLogger(__name__)

TITLE_SIMILARITY_MIN = 0.6
DURATION_TOLERANCE = 0.2     # +/-20% when both durations are known
MAX_CANDIDATE_DURATION_S = 1800.0   # reject obvious loops/compilations


def sanitize_query(text: str) -> str:
    """Strip characters that confuse yt-dlp search-prefix parsing.

    Apostrophes stay - they are common in titles and harmless (no shell;
    only ':' is meaningful to the search prefix).
    """
    return re.sub(r"\s+", " ", re.sub(r'[:;|&"]', " ", text)).strip()


class Candidate(BaseModel):
    url: str                  # webpage_url
    title: str
    uploader: str | None = None
    duration: float | None = None
    license: str | None = None
    source: str               # "soundcloud" | "youtube"


def _ydl_opts(out_dir: Path) -> dict:
    return {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }


def probe_query(query: str) -> Candidate | None:
    """Phase 1: resolve a search query / URL without downloading."""
    import yt_dlp
    from yt_dlp.utils import DownloadError

    source = "soundcloud" if query.startswith("scsearch") else "youtube"
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                               "noplaylist": True}) as ydl:
            info = ydl.extract_info(query, download=False)
    except DownloadError as exc:
        log.warning("probe failed for %s: %s", query, exc)
        return None
    if "entries" in info:
        if not info["entries"]:
            return None
        info = info["entries"][0]
    formats = info.get("formats") or []
    if formats and all(
            str(f.get("format_id", "")).endswith("_preview")
            for f in formats):
        return None   # SoundCloud preview-only (not fully streamable)
    return Candidate(
        url=info["webpage_url"],
        title=info.get("title", ""),
        uploader=info.get("uploader") or info.get("channel"),
        duration=info.get("duration"),
        license=info.get("license"),
        source=source,
    )


def download_candidate(candidate: Candidate, out_dir: Path) -> Path:
    """Phase 2: download + extract MP3; returns the final audio path."""
    import yt_dlp

    with yt_dlp.YoutubeDL(_ydl_opts(out_dir)) as ydl:
        info = ydl.extract_info(candidate.url, download=True)
        if "entries" in info:
            info = info["entries"][0]
        return Path(info["requested_downloads"][0]["filepath"])


def validate_candidate(candidate: Candidate, match_info: dict) -> bool:
    """Fuzzy title check + duration tolerance against the identified song."""
    got = f"{candidate.uploader or ''} {candidate.title}".lower()
    ratio = difflib.SequenceMatcher(
        None, match_info["title"].lower(), candidate.title.lower()).ratio()
    contained = match_info["title"].lower() in got
    if ratio < TITLE_SIMILARITY_MIN and not contained:
        return False
    expected = match_info.get("duration_s")
    if expected and candidate.duration:
        if abs(candidate.duration - expected) > DURATION_TOLERANCE * expected:
            return False
    if candidate.duration and candidate.duration > MAX_CANDIDATE_DURATION_S:
        return False
    return True


def is_licensed_free(candidate: Candidate) -> bool:
    lic = (candidate.license or "").lower()
    return "creative commons" in lic or lic.startswith("cc-")


def trim_audio(audio: Path, start_s: float, duration_s: float,
               dest: Path) -> Path:
    run_ffmpeg(["-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}",
                "-i", str(audio), "-c:a", "libmp3lame", "-q:a", "2",
                str(dest)])
    return dest


@register_analyzer
class MusicAcquisition:
    name = "music_acquisition"
    layer = "music_acquisition"
    needs_gpu = False
    prober = staticmethod(probe_query)            # injectable for tests
    downloader = staticmethod(download_candidate)  # injectable for tests

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        music = manifest.audio.music
        if not music.detected:
            return {"layers_status": {"music_acquisition": "skipped"}}

        policy = config.acquisition_policy()
        match_info = {"title": music.title or "", "artist": music.artist or "",
                      "duration_s": music.duration_s}
        query_text = sanitize_query(f"{music.artist} {music.title}")
        links = dict(music.acquisition.links)

        chosen: Candidate | None = None
        for query in (f"scsearch1:{query_text}", f"ytsearch1:{query_text}"):
            try:
                candidate = self.prober(query)
            except Exception:
                log.exception("resolver probe failed for %s", query)
                continue
            if candidate is None:
                continue
            if not validate_candidate(candidate, match_info):
                continue
            links.setdefault(candidate.source, candidate.url)
            if chosen is None or (policy == "licensed_only"
                                  and not is_licensed_free(chosen)
                                  and is_licensed_free(candidate)):
                chosen = candidate  # prefer a CC candidate under licensed_only

        audio = manifest.audio.model_dump(mode="json")
        acq = audio["music"]["acquisition"]
        acq["links"] = links

        if chosen is None:
            acq["status"] = "failed"
            return {"audio": audio,
                    "layers_status": {"music_acquisition": "failed"}}

        acq["license"] = chosen.license
        download_allowed = policy == "always" or (
            policy == "licensed_only" and is_licensed_free(chosen))

        if not download_allowed:
            acq["status"] = "skipped"
            acq["skip_reason"] = (
                "policy:link_only" if policy == "link_only"
                else f"license:{chosen.license}")
            return {"audio": audio,
                    "layers_status": {"music_acquisition": "ok"}}

        try:
            full = self.downloader(chosen, ws.media_dir)
            seg = music.song_segment
            trimmed = trim_audio(Path(full), seg.start_in_song, seg.duration,
                                 ws.media_dir / "music.mp3")
            acq["status"] = "acquired"
            acq["file"] = str(trimmed)
            return {"audio": audio,
                    "layers_status": {"music_acquisition": "ok"}}
        except Exception:
            log.exception("download/trim failed for %s", chosen.url)
            acq["status"] = "failed"
            return {"audio": audio,
                    "layers_status": {"music_acquisition": "failed"}}

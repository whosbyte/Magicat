# magicat/modules/audio/identify.py
"""Turn per-window provider matches into one music description.

Alignment model: every matched window i gives an "anchor" a_i =
window_start_i - song_offset_i = the video time where the song's 0:00 would
land. A steadily-playing song gives identical anchors; we take the median
and drop windows deviating > ANCHOR_TOLERANCE_S (re-fingerprint noise,
DJ edits, lyric repeats). The matched span approximates where the music
plays in the video - M2 precision is window-level (~10s), refined later.
"""
from __future__ import annotations

import logging
import statistics
from typing import Any

from magicat.modules.audio.extract import AudioWindow
from magicat.modules.audio.providers import (
    MusicIdProvider,
    ProviderError,
    SongMatch,
)

log = logging.getLogger(__name__)

ANCHOR_TOLERANCE_S = 5.0


def recognize_windows(windows: list[AudioWindow],
                      provider: MusicIdProvider) -> list[SongMatch | None]:
    """Identify every window; provider errors degrade to no-match."""
    results: list[SongMatch | None] = []
    for window in windows:
        try:
            results.append(provider.identify(window.path))
        except (ProviderError, OSError) as exc:
            log.warning("window at %.1fs failed: %s", window.t_start, exc)
            results.append(None)
    return results


def align(windows: list[AudioWindow], matches: list[SongMatch | None],
          video_duration: float, window_s: float) -> dict[str, Any] | None:
    """Build the manifest's audio.music dict from window matches."""
    hits = [(w, m) for w, m in zip(windows, matches) if m is not None]
    if not hits:
        return None

    # majority vote on song identity
    def identity(match: SongMatch) -> tuple[str, str]:
        return (match.title.strip().lower(), match.artist.strip().lower())

    counts: dict[tuple[str, str], int] = {}
    for _, match in hits:
        counts[identity(match)] = counts.get(identity(match), 0) + 1
    winner = max(counts, key=lambda k: counts[k])
    hits = [(w, m) for w, m in hits if identity(m) == winner]

    # consensus anchor (video time of the song's 0:00), outliers dropped
    anchors = [w.t_start - m.song_offset_s for w, m in hits]
    consensus = statistics.median(anchors)
    inliers = [(w, m) for (w, m), a in zip(hits, anchors)
               if abs(a - consensus) <= ANCHOR_TOLERANCE_S]
    if not inliers:
        inliers = hits  # all "outliers": fall back rather than drop the song
    consensus = statistics.median(
        w.t_start - m.song_offset_s for w, m in inliers)

    first_w = inliers[0][0]
    last_w = inliers[-1][0]
    timeline_offset = first_w.t_start
    start_in_song = max(0.0, timeline_offset - consensus)
    span_end = min(video_duration, last_w.t_start + window_s)
    best = inliers[0][1]

    return {
        "detected": True,
        "title": best.title,
        "artist": best.artist,
        "duration_s": best.duration_s,
        "provider_ids": best.provider_ids,
        "song_segment": {
            "start_in_song": start_in_song,
            "duration": span_end - timeline_offset,
        },
        "timeline_offset": timeline_offset,
        "acquisition": {
            "status": "skipped",
            "links": best.links,
        },
    }

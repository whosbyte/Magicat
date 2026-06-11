# magicat/modules/export/capcut.py
"""CapCut draft exporter (spec section 7 CapCut row - feature-flagged).

Wraps pycapcut (PINNED ==0.0.3): generates a CapCut-International draft
folder (draft_content.json + draft_meta_info.json, microsecond times,
absolute media paths) and zips it. The format is reverse-engineered -
the test suite pins the load-bearing structure so a dependency bump that
silently changes it fails in CI, not in a user's CapCut.

MAGICAT_CAPCUT_EXPORT=0 disables via SkippedExport (layer 'skipped').
Media is referenced by absolute path into the JOB's media dir (CapCut
drafts always use absolute paths); the zip carries instructions.
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

from magicat.core.interfaces import SkippedExport
from magicat.core.registry import register_exporter
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest

MICROS = 1_000_000

# CapCut text alignment is an int enum (pycapcut TextStyle.align):
# 0=left, 1=center, 2=right (verified in
# .venv/Lib/site-packages/pycapcut/text_segment.py docstring).
_ALIGN = {"left": 0, "center": 1, "right": 2}


def _parse_fill(fill: str | None) -> tuple[float, float, float]:
    """Map a manifest '#RRGGBB' fill to an RGB float triple in [0, 1].

    Defaults to white on a missing or unparseable value (pycapcut's own
    TextStyle default), so a malformed caption colour never breaks export.
    """
    if not fill:
        return (1.0, 1.0, 1.0)
    s = fill.lstrip("#")
    if len(s) != 6:
        return (1.0, 1.0, 1.0)
    try:
        r, g, b = (int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (1.0, 1.0, 1.0)
    return (r, g, b)

INSTRUCTIONS = """Magicat CapCut draft
=====================

1. Extract this zip. Inside is one folder: the draft.
2. Move that folder into CapCut's local drafts directory, e.g.
   C:\\Users\\<you>\\AppData\\Local\\CapCut\\User Data\\Projects\\com.lveditor.draft\\
3. Restart CapCut - the project appears in your local drafts.
4. Media is referenced by ABSOLUTE path from the Magicat job folder -
   keep that folder, or relink the clips inside CapCut if you move it.

The draft format is reverse-engineered (pycapcut, CapCut International);
if a CapCut update rejects it, re-export with a newer Magicat.
"""


@register_exporter
class CapCutExporter:
    format = "capcut_zip"

    def export(self, manifest: Manifest, ws: Workspace) -> Path:
        if os.environ.get("MAGICAT_CAPCUT_EXPORT", "1") == "0":
            raise SkippedExport("MAGICAT_CAPCUT_EXPORT=0")
        if not manifest.shots:
            raise ValueError("no shots in manifest - nothing to export")

        import pycapcut as cc

        staging = ws.exports_dir / "capcut_staging"
        if staging.is_dir():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        width, height = 1080, 1920
        if manifest.source.resolution and "x" in manifest.source.resolution:
            width, height = (int(v)
                             for v in manifest.source.resolution.split("x"))

        draft_name = f"magicat_{manifest.job_id[:8]}"
        folder = cc.DraftFolder(str(staging))
        script = folder.create_draft(draft_name, width, height)

        script.add_track(cc.TrackType.video)
        source_material = cc.VideoMaterial(str(manifest.source.file))
        cursor = 0
        for shot in manifest.shots:
            start_us = round(shot.start * MICROS)
            duration_us = round((shot.end - shot.start) * MICROS)
            # pycapcut raises ValueError when source end exceeds the
            # MediaInfo-reported duration (which can run ~10ms short of
            # ffprobe's container duration) - clamp the tail shot
            duration_us = min(duration_us,
                              max(0, source_material.duration - start_us))
            if duration_us <= 0:
                continue   # shot starts past the probed media end: skip
            segment = cc.VideoSegment(
                source_material,
                target_timerange=cc.trange(cursor, duration_us),
                source_timerange=cc.trange(start_us, duration_us))
            script.add_segment(segment)
            cursor += duration_us

        music = manifest.audio.music
        if music.detected and music.acquisition.file \
                and Path(music.acquisition.file).is_file():
            script.add_track(cc.TrackType.audio)
            offset_us = round(music.timeline_offset * MICROS)
            duration_us = round(music.song_segment.duration * MICROS)
            audio_segment = cc.AudioSegment(
                cc.AudioMaterial(str(music.acquisition.file)),
                target_timerange=cc.trange(offset_us, duration_us),
                source_timerange=cc.trange(0, duration_us))
            script.add_segment(audio_segment)

        if manifest.captions.segments:
            script.add_track(cc.TrackType.text)
            for seg in manifest.captions.segments:
                start_us = round(seg.t_start * MICROS)
                duration_us = round((seg.t_end - seg.t_start) * MICROS)
                style = cc.TextStyle(
                    color=_parse_fill(seg.style.fill),
                    align=_ALIGN.get(seg.style.alignment or "", 0))
                text_segment = cc.TextSegment(
                    seg.text, cc.trange(start_us, duration_us),
                    style=style)
                script.add_segment(text_segment)

        script.save()

        out = ws.exports_dir / "capcut_draft.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("CAPCUT_INSTRUCTIONS.txt", INSTRUCTIONS)
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(staging).as_posix())
        shutil.rmtree(staging)
        return out

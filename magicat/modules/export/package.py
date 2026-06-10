# magicat/modules/export/package.py
"""The deliverable: one zip containing the xmeml project, SRT captions,
report, import instructions, and the media files the project references.

The xmeml inside the zip is REWRITTEN against a manifest whose media paths
point at the zip's own media/ folder (relative pathurls survive extraction
anywhere); the original manifest is not modified.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PureWindowsPath

from magicat.core.registry import register_exporter
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest
from magicat.modules.export.fcp7 import to_xmeml
from magicat.modules.export.srt import to_srt

INSTRUCTIONS = """Magicat project import
=======================

Adobe Premiere:
  1. Extract this zip somewhere permanent (the project references media/).
  2. File > Import... > project.xml
  3. If media appears offline, right-click > Link Media to the media/ folder.
  4. File > Import... > captions.srt for the caption track (style specs
     are in report.html - font, fill color, position).

DaVinci Resolve:
  1. Extract this zip somewhere permanent.
  2. File > Import > Timeline... > project.xml
  3. Captions are NOT in the xml: File > Import > Subtitle... >
     captions.srt (Resolve will not auto-load it).
  4. Relink media to the media/ folder if prompted.

report.html summarizes everything that was recovered (song, links, fonts).
"""


@register_exporter
class PremiereResolvePackage:
    format = "premiere_resolve_zip"

    def export(self, manifest: Manifest, ws: Workspace) -> Path:
        staging = ws.exports_dir / "package"
        if staging.is_dir():
            shutil.rmtree(staging)
        media_dir = staging / "media"
        media_dir.mkdir(parents=True)

        # bundle media + a manifest copy whose paths are RELATIVE to the
        # zip root ("media/<name>") - an absolute path into this staging
        # dir would dangle the moment we rmtree it below, leaving the
        # imported project permanently offline (panel-review finding)
        bundled = manifest.model_copy(deep=True)
        src = Path(manifest.source.file)
        shutil.copy2(src, media_dir / src.name)
        bundled.source.file = f"media/{src.name}"
        music = manifest.audio.music
        if music.detected and music.acquisition.file:
            mfile = Path(music.acquisition.file)
            if mfile.is_file():
                shutil.copy2(mfile, media_dir / mfile.name)
                bundled.audio.music.acquisition.file = f"media/{mfile.name}"

        (staging / "project.xml").write_text(to_xmeml(bundled),
                                             encoding="utf-8")
        srt = to_srt(manifest)
        if srt:
            (staging / "captions.srt").write_text(srt, encoding="utf-8")
        report = ws.exports_dir / "report.html"
        if report.is_file():
            shutil.copy2(report, staging / "report.html")
        (staging / "IMPORT_INSTRUCTIONS.txt").write_text(
            INSTRUCTIONS, encoding="utf-8")

        out = ws.exports_dir / "premiere_resolve.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(staging).as_posix())
        shutil.rmtree(staging)
        return out

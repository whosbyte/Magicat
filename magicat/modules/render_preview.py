# magicat/modules/render_preview.py
"""Preview exporter: cut shots from the source and concat into preview.mp4."""
from __future__ import annotations

import subprocess
from pathlib import Path

from magicat.core.registry import register_exporter
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True, capture_output=True,
    )


@register_exporter
class PreviewRenderer:
    format = "preview_mp4"

    def export(self, manifest: Manifest, ws: Workspace) -> Path:
        source = Path(manifest.source.file)
        if not manifest.shots:
            raise ValueError("no shots in manifest - nothing to render")
        seg_dir = ws.exports_dir / "preview_segments"
        seg_dir.mkdir(parents=True, exist_ok=True)

        segments = []
        for shot in manifest.shots:
            seg = seg_dir / f"{shot.id}.mp4"
            # re-encode for frame-exact cuts (copy snaps to keyframes)
            # NOTE: -ss/-to appear BEFORE -i (input seek): -to is then an
            # absolute end time, so the segment spans [start, end]. Moving
            # these after -i silently changes semantics - keep them input-side.
            _ffmpeg(["-ss", f"{shot.start:.3f}", "-to", f"{shot.end:.3f}",
                     "-i", str(source), "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-c:a", "aac", str(seg)])
            segments.append(seg)

        concat_list = seg_dir / "list.txt"
        concat_list.write_text(
            "".join(f"file '{s.resolve().as_posix()}'\n" for s in segments),
            encoding="ascii",
        )
        out = ws.exports_dir / "preview.mp4"
        _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_list),
                 "-c", "copy", str(out)])
        return out

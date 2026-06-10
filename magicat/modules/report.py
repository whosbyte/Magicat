# magicat/modules/report.py
"""User-facing summary (spec section 9): a dict in manifest.report and a
standalone report.html included in every export package."""
from __future__ import annotations

import html
from pathlib import Path
from string import Template
from typing import Any

from magicat.core.registry import register_exporter
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest


def build_report(manifest: Manifest) -> dict[str, Any]:
    music = manifest.audio.music
    fonts = sorted({seg.style.font_family
                    for seg in manifest.captions.segments
                    if seg.style.font_family})
    return {
        "job_id": manifest.job_id,
        "source": {
            "url": manifest.source.url,
            "platform": manifest.source.platform,
            "duration": manifest.source.duration,
            "resolution": manifest.source.resolution,
        },
        "shots": {
            "count": len(manifest.shots),
            "keyframes": [shot.keyframes[0] for shot in manifest.shots
                          if shot.keyframes],
        },
        "music": {
            "detected": music.detected,
            "title": music.title,
            "artist": music.artist,
            "identified_by": music.provider,
            "links": dict(music.acquisition.links),
            "used_segment": {
                "start_in_song": music.song_segment.start_in_song,
                "duration": music.song_segment.duration,
            } if music.detected else None,
            "acquisition_status": music.acquisition.status,
        },
        "captions": {
            "count": len(manifest.captions.segments),
            "fonts": fonts,
            "font_candidates": [
                [c.model_dump() for c in seg.style.font_candidates]
                for seg in manifest.captions.segments],
            "transcript": [seg.text for seg in manifest.captions.segments],
        },
        "layers": {k: v.value for k, v in manifest.layers_status.items()},
    }


_PAGE = Template("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Magicat report $job_id</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;
     padding:0 1rem;color:#222}
h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:1.5rem}
table{border-collapse:collapse;width:100%}
td,th{text-align:left;padding:.3rem .6rem;border-bottom:1px solid #eee}
.tag{display:inline-block;padding:.1rem .5rem;border-radius:.5rem;
     background:#eef;margin-right:.3rem;font-size:.85rem}
</style></head><body>
<h1>Magicat reconstruction report</h1>
<p>Job <code>$job_id</code> &middot; $platform &middot; $duration s
&middot; $resolution</p>
<h2>Scenes</h2><p>$shot_count shots detected.</p>
<h2>Music</h2>$music_html
<h2>Captions</h2>$captions_html
<h2>Layer status</h2><table>$layers_rows</table>
</body></html>
""")


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "-"))


def _render_html(report: dict[str, Any]) -> str:
    music = report["music"]
    if music["detected"]:
        links = " ".join(
            f'<a class="tag" href="{_esc(url)}">{_esc(name)}</a>'
            for name, url in music["links"].items())
        seg = music["used_segment"] or {}
        music_html = (
            f"<p><strong>{_esc(music['title'])}</strong> by "
            f"{_esc(music['artist'])} (identified by "
            f"{_esc(music['identified_by'])})<br>"
            f"Segment used: {_esc(seg.get('start_in_song'))}s for "
            f"{_esc(seg.get('duration'))}s &middot; acquisition: "
            f"{_esc(music['acquisition_status'])}<br>{links}</p>")
    else:
        music_html = "<p>No music detected.</p>"

    caps = report["captions"]
    if caps["count"]:
        fonts = ", ".join(_esc(f) for f in caps["fonts"]) or "uncertain"
        lines = "".join(f"<li>{_esc(t)}</li>" for t in caps["transcript"])
        captions_html = (f"<p>{caps['count']} caption(s); font: {fonts}</p>"
                         f"<ul>{lines}</ul>")
    else:
        captions_html = "<p>No captions detected.</p>"

    layers_rows = "".join(
        f"<tr><td>{_esc(layer)}</td><td>{_esc(state)}</td></tr>"
        for layer, state in report["layers"].items())

    return _PAGE.substitute(
        job_id=_esc(report["job_id"]),
        platform=_esc(report["source"]["platform"]),
        duration=_esc(report["source"]["duration"]),
        resolution=_esc(report["source"]["resolution"]),
        shot_count=report["shots"]["count"],
        music_html=music_html,
        captions_html=captions_html,
        layers_rows=layers_rows,
    )


@register_exporter
class ReportExporter:
    format = "report_html"

    def export(self, manifest: Manifest, ws: Workspace) -> Path:
        out = ws.exports_dir / "report.html"
        out.write_text(_render_html(build_report(manifest)),
                       encoding="utf-8")
        return out

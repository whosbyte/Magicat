# magicat/modules/render_preview.py
"""Preview exporter: single-pass filter_complex render (verified shape).

One ffmpeg invocation cuts every shot (trim/atrim + PTS reset), concats
them, and - when music was acquired - mixes it in at timeline_offset.
Single-pass eliminates the per-segment AAC priming drift of the old
two-pass approach (~80ms over 3 segments) that would desync music.

Load-bearing flags (empirically verified):
  amix duration=first  - default 'longest' overruns video length
  amix normalize=0     - default halves source dialog everywhere
  adelay ...:all=1     - one value for every channel, any channel count
"""
from __future__ import annotations

from pathlib import Path

from magicat.core.ffmpeg import run_ffmpeg
from magicat.core.registry import register_exporter
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest

MUSIC_VOLUME = 0.8


def build_filtergraph(segments: list[tuple[float, float]], with_music: bool,
                      music_offset_s: float = 0.0,
                      music_volume: float = MUSIC_VOLUME) -> str:
    parts: list[str] = []
    concat_inputs: list[str] = []
    for i, (start, end) in enumerate(segments):
        parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},"
                     f"setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
                     f"asetpts=PTS-STARTPTS[a{i}]")
        concat_inputs.append(f"[v{i}][a{i}]")
    n = len(segments)
    if with_music:
        parts.append("".join(concat_inputs)
                     + f"concat=n={n}:v=1:a=1[vout][aconcat]")
        delay_ms = int(round(music_offset_s * 1000))
        parts.append(f"[1:a]volume={music_volume},"
                     f"adelay={delay_ms}:all=1[music]")
        parts.append("[aconcat][music]"
                     "amix=inputs=2:duration=first:normalize=0[aout]")
    else:
        parts.append("".join(concat_inputs)
                     + f"concat=n={n}:v=1:a=1[vout][aout]")
    return ";".join(parts)


@register_exporter
class PreviewRenderer:
    format = "preview_mp4"

    def export(self, manifest: Manifest, ws: Workspace) -> Path:
        if not manifest.shots:
            raise ValueError("no shots in manifest - nothing to render")
        source = Path(manifest.source.file)

        music = manifest.audio.music
        music_file = (music.acquisition.file
                      if music.detected and music.acquisition.file else None)

        segments = [(shot.start, shot.end) for shot in manifest.shots]
        out = ws.exports_dir / "preview.mp4"
        args = ["-i", str(source)]
        if music_file:
            args += ["-i", music_file]
        args += [
            "-filter_complex",
            build_filtergraph(segments, with_music=music_file is not None,
                              music_offset_s=music.timeline_offset),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(out),
        ]
        run_ffmpeg(args)
        return out

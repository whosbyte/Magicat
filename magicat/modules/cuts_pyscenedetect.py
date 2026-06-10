# magicat/modules/cuts_pyscenedetect.py
"""Default cut detector: PySceneDetect ContentDetector (CPU, CI-friendly).

TransNetV2 (cuts_transnetv2.py) is the GPU upgrade behind the same
interface; the pipeline picks by analyzer name.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from scenedetect import ContentDetector, detect

from magicat.core.registry import register_analyzer
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest


def extract_keyframe(video: Path, t: float, dest: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(dest)],
        check=True, capture_output=True,
    )


@register_analyzer
class CutDetector:
    name = "cut_detection"
    layer = "shots"
    needs_gpu = False

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        video = Path(manifest.source.file)
        # start_in_scene=True: always returns >= 1 scene spanning the full video.
        scenes = detect(str(video), ContentDetector(min_scene_len=3), start_in_scene=True)
        # Encoder artefacts show up as micro-scenes (~2 frames). Merge anything
        # shorter than MIN_SHOT_DURATION into its neighbour instead of dropping
        # it, so the timeline always covers the full video — real flash cuts
        # (>= 0.2 s) survive as their own shots.
        MIN_SHOT_DURATION = 0.2
        spans: list[tuple[float, float]] = []
        for s, e in ((s.seconds, e.seconds) for s, e in scenes):
            if spans and (e - s) < MIN_SHOT_DURATION:
                spans[-1] = (spans[-1][0], e)      # absorb into previous shot
            else:
                spans.append((s, e))
        if len(spans) > 1 and (spans[0][1] - spans[0][0]) < MIN_SHOT_DURATION:
            spans[1] = (spans[0][0], spans[1][1])  # leading micro-scene: absorb forward
            spans.pop(0)
        if not spans:
            spans = [(0.0, manifest.source.duration or 0.0)]

        shots = []
        for i, (start, end) in enumerate(spans):
            shot_id = f"shot_{i:03d}"
            keyframes = []
            # start / middle / end-epsilon, clamped inside the shot
            for label, t in (("a", start), ("b", (start + end) / 2),
                             ("c", max(start, end - 0.05))):
                kf = ws.keyframes_dir / f"{shot_id}_{label}.jpg"
                extract_keyframe(video, t, kf)
                keyframes.append(str(kf))
            shots.append({
                "id": shot_id, "start": start, "end": end,
                "keyframes": keyframes, "confidence": 1.0,
            })
        return {"shots": shots, "layers_status": {"shots": "ok"}}

# magicat/modules/cuts_transnetv2.py
"""TransNetV2 cut detector (GPU). Optional: requires `pip install -e .[transnet]`.

Import of the heavy dependency happens inside run() so registering the
module never costs anything when the extra is absent.
"""
from __future__ import annotations

from pathlib import Path

from magicat.core.registry import register_analyzer
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest
from magicat.modules.cuts_pyscenedetect import extract_keyframe


@register_analyzer
class TransNetV2Detector:
    name = "cut_detection_transnetv2"
    needs_gpu = True

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        # API verified against transnetv2-pytorch README; re-check on first
        # real install (predict_video/predictions_to_scenes naming).
        from transnetv2_pytorch import TransNetV2

        video = Path(manifest.source.file)
        fps = manifest.source.fps or 30.0
        model = TransNetV2()
        model.eval()
        _, single_frame_pred, _ = model.predict_video(str(video))
        scenes = model.predictions_to_scenes(single_frame_pred)  # frame spans

        shots = []
        for i, (f_start, f_end) in enumerate(scenes):
            start, end = f_start / fps, (f_end + 1) / fps
            shot_id = f"shot_{i:03d}"
            keyframes = []
            for label, t in (("a", start), ("b", (start + end) / 2),
                             ("c", max(start, end - 0.05))):
                kf = ws.keyframes_dir / f"{shot_id}_{label}.jpg"
                extract_keyframe(video, t, kf)
                keyframes.append(str(kf))
            shots.append({
                "id": shot_id, "start": round(start, 3), "end": round(end, 3),
                "keyframes": keyframes, "confidence": 1.0,
            })
        return {"shots": shots, "layers_status": {"shots": "ok"}}

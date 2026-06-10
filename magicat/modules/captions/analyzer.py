# magicat/modules/captions/analyzer.py
"""Caption analysis: sample frames -> OCR -> cluster -> style (spec 6.5)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from magicat.core.registry import register_analyzer
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest
from magicat.modules.captions.clustering import cluster_detections, estimate_fill
from magicat.modules.captions.ocr import RapidOcrEngine
from magicat.modules.captions.sampling import sample_frames

SAMPLE_FPS = 5.0

CROP_MARGIN = 0.02   # normalized margin around the caption bbox


def save_crop(frame_path: Path, bbox, dest: Path) -> str:
    """Cut the caption region (plus margin) out of a frame; returns path."""
    with Image.open(frame_path) as img:
        width, height = img.size
        x, y, w, h = bbox
        box = (max(0, int((x - CROP_MARGIN) * width)),
               max(0, int((y - CROP_MARGIN) * height)),
               min(width, int((x + w + CROP_MARGIN) * width)),
               min(height, int((y + h + CROP_MARGIN) * height)))
        img.crop(box).save(dest)
    return str(dest)


@register_analyzer
class CaptionAnalyzer:
    name = "caption_analysis"
    layer = "captions"
    needs_gpu = False
    engine_factory = staticmethod(RapidOcrEngine)   # injectable for tests

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        engine = self.engine_factory()
        samples = sample_frames(Path(manifest.source.file),
                                ws.media_dir / "ocr_frames", fps=SAMPLE_FPS)
        detections = [(s.t, engine.read(s.path)) for s in samples]
        segments = cluster_detections(detections,
                                      frame_interval=1.0 / SAMPLE_FPS)

        # clustering extends t_end by one frame interval; never overshoot
        # the actual video duration
        if manifest.source.duration:
            for seg in segments:
                seg["t_end"] = min(seg["t_end"], manifest.source.duration)

        # style (spec 6.5 step 5, the cheaply-derivable parts): fill color
        # from the segment's middle frame, size from bbox height in pixels,
        # alignment from the bbox center. Stroke/shadow are M3.
        # NOTE: size is the glyph-bbox INK height in px (~80% of the
        # authoring font size), not the font's em size.
        frame_height = None
        if manifest.source.resolution:
            frame_height = int(manifest.source.resolution.split("x")[1])
        crops_dir = ws.media_dir / "caption_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for i, seg in enumerate(segments):
            mid_t = (seg["t_start"] + seg["t_end"]) / 2
            frame = min(samples, key=lambda s: abs(s.t - mid_t))
            crop_times = {seg["t_start"], mid_t,
                          max(seg["t_start"], seg["t_end"] - 1.0 / SAMPLE_FPS)}
            seg["crops"] = []
            for j, ct in enumerate(sorted(crop_times)):
                src_frame = min(samples, key=lambda s: abs(s.t - ct))
                seg["crops"].append(save_crop(
                    src_frame.path, seg["bbox"],
                    crops_dir / f"seg_{i:03d}_{j}.png"))
            x, _, w, h = seg["bbox"]
            center = x + w / 2
            if abs(center - 0.5) < 0.05:
                alignment = "center"
            else:
                alignment = "left" if center < 0.5 else "right"
            seg["style"] = {
                "fill": estimate_fill(frame.path, seg["bbox"]),
                "size": round(h * frame_height, 1) if frame_height else None,
                "alignment": alignment,
            }

        return {
            "captions": {"segments": segments},
            "layers_status": {"captions": "ok"},
        }

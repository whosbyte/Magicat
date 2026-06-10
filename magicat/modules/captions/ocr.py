# magicat/modules/captions/ocr.py
"""OCR engine protocol + RapidOCR implementation (PP-OCRv4 via ONNX).

RapidOCR specifics (verified against rapidocr 3.8.x): the RapidOCR()
constructor downloads ~15MB of models on first ever use (cached under
site-packages afterwards) - so construction is lazy and shared. Empty
frames return None (not []) in .txts/.boxes/.scores.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image
from pydantic import BaseModel


class OcrLine(BaseModel):
    text: str
    bbox: tuple[float, float, float, float]   # normalized x, y, w, h
    confidence: float


@runtime_checkable
class OcrEngine(Protocol):
    def read(self, image: Path) -> list[OcrLine]: ...


def quad_to_bbox(quad: np.ndarray, width: int,
                 height: int) -> tuple[float, float, float, float]:
    """RapidOCR box: (4,2) pixel corners TL,TR,BR,BL -> normalized xywh."""
    xs, ys = quad[:, 0], quad[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height)


class RapidOcrEngine:
    def __init__(self) -> None:
        self._engine = None   # constructed on first read (model download)

    def read(self, image: Path) -> list[OcrLine]:
        if self._engine is None:
            from rapidocr import RapidOCR
            self._engine = RapidOCR()
        result = self._engine(str(image))
        if not result.txts:          # None on empty frames
            return []
        width, height = Image.open(image).size
        lines = []
        for quad, text, score in zip(result.boxes, result.txts,
                                     result.scores):
            lines.append(OcrLine(
                text=text,
                bbox=quad_to_bbox(np.asarray(quad), width, height),
                confidence=float(score),
            ))
        return lines

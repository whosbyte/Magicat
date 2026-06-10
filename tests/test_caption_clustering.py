# tests/test_caption_clustering.py
from magicat.modules.captions.clustering import (
    bbox_iou,
    cluster_detections,
    estimate_fill,
    text_similar,
)
from magicat.modules.captions.ocr import OcrLine


BOTTOM = (0.25, 0.8, 0.5, 0.06)


def line(text: str, bbox=BOTTOM, conf: float = 0.95) -> OcrLine:
    return OcrLine(text=text, bbox=bbox, confidence=conf)


def test_bbox_iou():
    assert bbox_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert bbox_iou((0, 0, 0.5, 1), (0.5, 0, 0.5, 1)) == 0.0
    assert abs(bbox_iou((0, 0, 1, 1), (0, 0, 1, 0.5)) - 0.5) < 1e-9


def test_text_similar():
    assert text_similar("HELLO WORLD", "HELL0 WORLD") is True   # OCR noise
    assert text_similar("HELLO WORLD", "SECOND LINE") is False


def test_clusters_two_sequential_captions():
    detections = []
    for i in range(5, 15):                       # t=1.0..2.8: HELLO WORLD
        detections.append((i * 0.2, [line("HELLO WORLD")]))
    for i in range(15, 17):                      # gap (no text)
        detections.append((i * 0.2, []))
    for i in range(17, 26):                      # t=3.4..5.0: SECOND LINE
        detections.append((i * 0.2, [line("SECOND LINE")]))

    segments = cluster_detections(detections, frame_interval=0.2)
    assert len(segments) == 2
    first, second = segments
    assert first["text"] == "HELLO WORLD"
    assert abs(first["t_start"] - 1.0) < 0.01
    assert abs(first["t_end"] - 3.0) < 0.01      # last frame t + interval
    assert second["text"] == "SECOND LINE"
    assert abs(second["t_start"] - 3.4) < 0.01


def test_single_frame_noise_dropped():
    detections = [(0.0, []), (0.2, [line("GLITCH")]), (0.4, []),
                  (0.6, []), (0.8, [])]
    assert cluster_detections(detections, frame_interval=0.2) == []


def test_one_frame_ocr_miss_bridged():
    detections = []
    for i in range(10):
        if i == 5:                                # OCR missed one frame
            detections.append((i * 0.2, []))
        else:
            detections.append((i * 0.2, [line("STEADY CAPTION")]))
    segments = cluster_detections(detections, frame_interval=0.2)
    assert len(segments) == 1
    assert abs(segments[0]["t_end"] - 2.0) < 0.01


def test_ocr_text_variants_majority_vote():
    detections = [(i * 0.2, [line("HELLO WORLD")]) for i in range(4)]
    detections.append((0.8, [line("HELL0 WORLD")]))   # one noisy read
    detections.append((1.0, [line("HELLO WORLD")]))
    segments = cluster_detections(detections, frame_interval=0.2)
    assert segments[0]["text"] == "HELLO WORLD"


def test_low_confidence_lines_ignored():
    detections = [(i * 0.2, [line("???", conf=0.3)]) for i in range(6)]
    assert cluster_detections(detections, frame_interval=0.2) == []


def test_estimate_fill_white_text(tmp_path):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (480, 854), (32, 32, 32))
    draw = ImageDraw.Draw(img)
    draw.rectangle([140, 690, 340, 740], fill=(250, 250, 250))
    p = tmp_path / "frame.png"
    img.save(p)
    fill = estimate_fill(p, (140 / 480, 690 / 854, 200 / 480, 50 / 854))
    r, g, b = (int(fill[i:i + 2], 16) for i in (1, 3, 5))
    assert r > 200 and g > 200 and b > 200

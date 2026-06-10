# tests/test_caption_sampling.py
from magicat.modules.captions.sampling import sample_frames


def test_samples_at_five_fps(caption_video, tmp_path):
    samples = sample_frames(caption_video, tmp_path / "frames")
    # 6s at 5fps -> ~30 frames (container rounding may add/drop one)
    assert 28 <= len(samples) <= 31
    assert samples[0].t == 0.0
    assert abs(samples[1].t - 0.2) < 1e-9
    for s in samples[:3]:
        assert s.path.is_file() and s.path.stat().st_size > 0


def test_sample_timestamps_monotonic(caption_video, tmp_path):
    samples = sample_frames(caption_video, tmp_path / "frames")
    ts = [s.t for s in samples]
    assert ts == sorted(ts)

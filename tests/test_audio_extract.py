# tests/test_audio_extract.py
from magicat.modules.audio.extract import cut_windows, extract_wav, wav_duration
from tests.conftest import probe_duration


def test_extract_wav_from_video(fixture_video, tmp_path):
    wav = extract_wav(fixture_video, tmp_path / "audio.wav")
    assert wav.is_file()
    assert abs(probe_duration(wav) - 6.0) < 0.3


def test_wav_duration(long_wav):
    assert abs(wav_duration(long_wav) - 25.0) < 0.1


def test_cut_windows_short_audio_single_window(fixture_video, tmp_path):
    wav = extract_wav(fixture_video, tmp_path / "audio.wav")
    windows = cut_windows(wav, tmp_path / "win")
    assert len(windows) == 1
    assert windows[0].t_start == 0.0
    assert abs(probe_duration(windows[0].path) - 6.0) < 0.3  # min(12, remaining)


def test_cut_windows_long_audio(long_wav, tmp_path):
    windows = cut_windows(long_wav, tmp_path / "win")
    assert [w.t_start for w in windows] == [0.0, 10.0, 20.0]
    assert abs(probe_duration(windows[0].path) - 12.0) < 0.2
    assert abs(probe_duration(windows[2].path) - 5.0) < 0.3  # 25-20 remaining


def test_cut_windows_respects_max(long_wav, tmp_path):
    windows = cut_windows(long_wav, tmp_path / "win", max_windows=2)
    assert len(windows) == 2

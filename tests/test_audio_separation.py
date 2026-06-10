# tests/test_audio_separation.py
import pytest

from magicat.modules.audio import separation


def test_available_reflects_install_state():
    import importlib.util
    expected = importlib.util.find_spec("demucs_infer") is not None
    assert separation.available() is expected


def test_separation_disabled_by_env(monkeypatch):
    monkeypatch.setenv("MAGICAT_USE_SEPARATION", "never")
    assert separation.enabled() is False


def test_split_music_bed(long_wav, tmp_path):
    pytest.importorskip(
        "demucs_infer", reason="optional separation extra not installed")
    music_bed, vocals = separation.split_music_bed(long_wav, tmp_path)
    assert music_bed.is_file()
    assert vocals.is_file()

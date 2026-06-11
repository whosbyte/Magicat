# tests/test_download_guard.py
"""Tests for the download watchdog, config budgets, ingest error translation,
and ffmpeg subprocess timeouts - all offline (no real network downloads)."""
import subprocess

import pytest

from magicat import config
from magicat.core import download_guard
from magicat.core.download_guard import DownloadTimeout, timeout_hook
from magicat.core.ffmpeg import run_ffmpeg
from magicat.modules import ingest


# --- timeout_hook ---------------------------------------------------------

def test_timeout_hook_raises_once_deadline_passed(monkeypatch):
    """Deadline in the past -> first hook call raises DownloadTimeout."""
    times = iter([100.0, 100.0 + 999.0])  # build time, then check time
    monkeypatch.setattr(download_guard.time, "monotonic",
                        lambda: next(times))
    hook = timeout_hook(budget_s=5.0)
    with pytest.raises(DownloadTimeout, match="exceeded 5s budget"):
        hook({"status": "downloading"})


def test_timeout_hook_no_raise_before_deadline(monkeypatch):
    """Future deadline -> hook does not raise."""
    times = iter([100.0, 100.5])  # build time, then a check 0.5s later
    monkeypatch.setattr(download_guard.time, "monotonic",
                        lambda: next(times))
    hook = timeout_hook(budget_s=60.0)
    assert hook({"status": "downloading"}) is None


# --- config budgets -------------------------------------------------------

def test_config_ingest_timeout_s(monkeypatch):
    assert config.ingest_timeout_s() == 120.0
    monkeypatch.setenv("MAGICAT_INGEST_TIMEOUT_S", "45")
    assert config.ingest_timeout_s() == 45.0
    monkeypatch.setenv("MAGICAT_INGEST_TIMEOUT_S", "bogus")
    with pytest.raises(ValueError):
        config.ingest_timeout_s()
    monkeypatch.setenv("MAGICAT_INGEST_TIMEOUT_S", "-5")
    with pytest.raises(ValueError):
        config.ingest_timeout_s()


def test_config_acquisition_timeout_s(monkeypatch):
    assert config.acquisition_timeout_s() == 90.0
    monkeypatch.setenv("MAGICAT_ACQUISITION_TIMEOUT_S", "30")
    assert config.acquisition_timeout_s() == 30.0
    monkeypatch.setenv("MAGICAT_ACQUISITION_TIMEOUT_S", "bogus")
    with pytest.raises(ValueError):
        config.acquisition_timeout_s()
    monkeypatch.setenv("MAGICAT_ACQUISITION_TIMEOUT_S", "0")
    with pytest.raises(ValueError):
        config.acquisition_timeout_s()


# --- ingest error translation (offline) -----------------------------------

class _StubYDL:
    """Stub yt_dlp.YoutubeDL whose extract_info raises a chosen exception."""

    def __init__(self, exc):
        self._exc = exc

    def __call__(self, opts):  # yt_dlp.YoutubeDL(opts)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=True):
        raise self._exc


def _patch_yt_dlp(monkeypatch, exc):
    import yt_dlp
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _StubYDL(exc))


def test_ingest_download_translates_watchdog_timeout(monkeypatch, tmp_path):
    """A raw DownloadTimeout (verified yt-dlp behavior: hook exception
    propagates as-is) becomes an actionable RuntimeError."""
    _patch_yt_dlp(monkeypatch, DownloadTimeout("download exceeded 120s budget"))
    with pytest.raises(RuntimeError) as exc_info:
        ingest.download("https://youtu.be/abc", tmp_path)
    msg = str(exc_info.value)
    assert "timed out" in msg
    assert "Deno" in msg


def test_ingest_download_translates_wrapped_timeout(monkeypatch, tmp_path):
    """If a DownloadTimeout surfaces wrapped in DownloadError, it is still
    translated to the actionable RuntimeError."""
    from yt_dlp.utils import DownloadError
    wrapped = DownloadError("ERROR: download failed")
    wrapped.__cause__ = DownloadTimeout("download exceeded 120s budget")
    _patch_yt_dlp(monkeypatch, wrapped)
    with pytest.raises(RuntimeError) as exc_info:
        ingest.download("https://youtu.be/abc", tmp_path)
    assert "timed out" in str(exc_info.value)
    assert "Deno" in str(exc_info.value)


def test_ingest_download_reraises_unrelated_download_error(monkeypatch,
                                                           tmp_path):
    """A plain DownloadError (not a timeout) is re-raised, not masked."""
    from yt_dlp.utils import DownloadError
    _patch_yt_dlp(monkeypatch, DownloadError("ERROR: video unavailable"))
    with pytest.raises(DownloadError, match="video unavailable"):
        ingest.download("https://youtu.be/abc", tmp_path)


# --- ffmpeg subprocess timeout --------------------------------------------

def test_run_ffmpeg_times_out():
    """A realtime-paced 10s ffmpeg job exceeds a 0.5s budget -> TimeoutExpired.
    -re forces realtime so the process is still running when the timeout fires;
    output discarded to the null muxer so nothing is written."""
    with pytest.raises(subprocess.TimeoutExpired):
        run_ffmpeg(
            ["-re", "-f", "lavfi", "-i", "anullsrc", "-t", "10",
             "-f", "null", "-"],
            timeout_s=0.5,
        )

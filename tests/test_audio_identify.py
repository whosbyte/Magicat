# tests/test_audio_identify.py
import time
from pathlib import Path

from magicat.modules.audio import identify
from magicat.modules.audio.extract import AudioWindow
from magicat.modules.audio.identify import align, recognize_windows
from magicat.modules.audio.providers import ProviderError, SongMatch


def w(t: float) -> AudioWindow:
    return AudioWindow(t_start=t, path=Path(f"win_{int(t)}.wav"))


def m(offset: float, title: str = "Song", artist: str = "Artist",
      **kw) -> SongMatch:
    return SongMatch(title=title, artist=artist, song_offset_s=offset,
                     provider="fake", **kw)


class FakeProvider:
    name = "fake"

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def identify(self, clip):
        result = self.results[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


def test_recognize_windows_maps_provider_errors_to_none():
    provider = FakeProvider([m(30.0), ProviderError("quota"), None])
    windows = [w(0.0), w(10.0), w(20.0)]
    results, errors, timed_out = recognize_windows(windows, provider)
    assert results[0].song_offset_s == 30.0
    assert results[1] is None
    assert results[2] is None
    assert errors == 1
    assert timed_out is False


def test_recognize_windows_already_expired_deadline_makes_no_calls():
    provider = FakeProvider([m(30.0), m(40.0), m(50.0)])
    windows = [w(0.0), w(10.0), w(20.0)]
    results, errors, timed_out = recognize_windows(
        windows, provider, deadline=time.monotonic() - 1)
    assert results == []
    assert errors == 0
    assert provider.calls == 0
    assert timed_out is True


def test_recognize_windows_no_deadline_attempts_all():
    provider = FakeProvider([m(30.0), m(40.0), m(50.0)])
    windows = [w(0.0), w(10.0), w(20.0)]
    results, errors, timed_out = recognize_windows(windows, provider)
    assert len(results) == 3
    assert provider.calls == 3
    assert timed_out is False


def test_recognize_windows_deadline_expires_mid_list(monkeypatch):
    # A fake clock the provider advances past the deadline after the first
    # call - deterministic, no real sleeps.
    clock = {"now": 1000.0}
    monkeypatch.setattr(identify.time, "monotonic", lambda: clock["now"])
    deadline = clock["now"] + 5.0   # budget exhausted after the first call

    class AdvancingProvider:
        name = "fake"

        def __init__(self):
            self.calls = 0

        def identify(self, clip):
            self.calls += 1
            clock["now"] = deadline + 1.0   # push past the deadline
            return m(30.0)

    provider = AdvancingProvider()
    windows = [w(0.0), w(10.0), w(20.0)]
    results, errors, timed_out = recognize_windows(
        windows, provider, deadline=deadline)
    assert provider.calls == 1            # only the attempted prefix
    assert len(results) == 1
    assert results[0].song_offset_s == 30.0
    assert timed_out is True
    # align zips, pairing only the attempted prefix - the rest is ignored
    music = align(windows, results, video_duration=25.0, window_s=12.0)
    assert music["detected"] is True


def test_align_consistent_windows():
    # song plays from its 30s mark starting at video t=0
    windows = [w(0.0), w(10.0), w(20.0)]
    matches = [m(30.0), m(40.0), m(50.0)]
    music = align(windows, matches, video_duration=25.0, window_s=12.0)
    assert music["detected"] is True
    assert music["title"] == "Song"
    assert music["timeline_offset"] == 0.0
    assert music["song_segment"]["start_in_song"] == 30.0
    # matched span: first window start -> min(last start + 12, video end)
    assert music["song_segment"]["duration"] == 25.0


def test_align_rejects_outlier_window():
    windows = [w(0.0), w(10.0), w(20.0)]
    matches = [m(30.0), m(95.0), m(50.0)]   # middle anchor disagrees by 55s
    music = align(windows, matches, video_duration=25.0, window_s=12.0)
    assert music["detected"] is True
    assert music["song_segment"]["start_in_song"] == 30.0


def test_align_majority_vote_on_identity():
    windows = [w(0.0), w(10.0), w(20.0)]
    matches = [m(30.0), m(40.0), m(10.0, title="Other Song")]
    music = align(windows, matches, video_duration=25.0, window_s=12.0)
    assert music["title"] == "Song"


def test_align_music_starts_mid_video():
    # nothing matches at t=0; song matched from t=10 at its very start
    windows = [w(0.0), w(10.0), w(20.0)]
    matches = [None, m(0.0), m(10.0)]
    music = align(windows, matches, video_duration=25.0, window_s=12.0)
    assert music["timeline_offset"] == 10.0
    assert music["song_segment"]["start_in_song"] == 0.0
    assert music["song_segment"]["duration"] == 15.0


def test_align_no_matches_returns_none():
    assert align([w(0.0)], [None], video_duration=6.0, window_s=12.0) is None


def test_align_carries_metadata_from_best_match():
    windows = [w(0.0), w(10.0)]
    matches = [m(30.0, provider_ids={"spotify": "x"},
                 links={"spotify": "url"}, duration_s=200.0),
               m(40.0)]
    music = align(windows, matches, video_duration=22.0, window_s=12.0)
    assert music["provider_ids"] == {"spotify": "x"}
    assert music["acquisition"]["links"] == {"spotify": "url"}
    assert music["duration_s"] == 200.0   # consumed by acquisition validation
    assert music["provider"] == "fake"

# tests/test_audio_acquire.py
from pathlib import Path

import pytest

from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest, Source
from magicat.modules.audio.acquire import (
    Candidate,
    MusicAcquisition,
    is_licensed_free,
    validate_candidate,
)
from tests.conftest import probe_duration, run_ffmpeg


def music_manifest(**acq_kwargs) -> Manifest:
    return Manifest(job_id="j", source=Source(file="x.mp4", duration=25.0),
                    audio={"music": {
                        "detected": True, "title": "Around the World",
                        "artist": "Daft Punk",
                        "duration_s": 428.0,
                        "song_segment": {"start_in_song": 30.0,
                                         "duration": 20.0},
                        "timeline_offset": 0.0,
                        "acquisition": {"status": "skipped",
                                        "links": {"spotify": "sp"}},
                    }})


def cand(**kw) -> Candidate:
    base = dict(url="https://soundcloud.com/x/y", title="Around the World",
                uploader="daftpunk", duration=428.0,
                license="all-rights-reserved", source="soundcloud")
    base.update(kw)
    return Candidate(**base)


def test_validate_candidate_title_and_duration():
    match_info = {"title": "Around the World", "artist": "Daft Punk",
                  "duration_s": 428.0}
    assert validate_candidate(cand(), match_info) is True
    assert validate_candidate(cand(title="totally different song xyz"),
                              match_info) is False
    assert validate_candidate(cand(duration=120.0), match_info) is False
    # unknown durations are not held against the candidate
    assert validate_candidate(cand(duration=None),
                              {"title": "Around the World",
                               "artist": "Daft Punk",
                               "duration_s": None}) is True


def test_is_licensed_free():
    assert is_licensed_free(cand(license="cc-by-sa")) is True
    assert is_licensed_free(cand(
        license="Creative Commons Attribution license (reuse allowed)")) is True
    assert is_licensed_free(cand(license="all-rights-reserved")) is False
    assert is_licensed_free(cand(license=None)) is False


@pytest.fixture()
def song_mp3(tmp_path) -> Path:
    """60s sine 'song' to trim from."""
    p = tmp_path / "song.mp3"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=330:duration=60",
                "-c:a", "libmp3lame", str(p)])
    return p


def make_analyzer(monkeypatch, candidate, download_path, policy="always"):
    analyzer = MusicAcquisition()
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", policy)
    monkeypatch.setattr(analyzer, "prober",
                        lambda query: candidate)
    monkeypatch.setattr(analyzer, "downloader",
                        lambda c, out_dir: download_path)
    return analyzer


def test_policy_always_downloads_and_trims(tmp_path, monkeypatch, song_mp3):
    ws = Workspace(tmp_path / "job")
    analyzer = make_analyzer(monkeypatch, cand(), song_mp3)
    patch = analyzer.run(music_manifest(), ws)
    acq = patch["audio"]["music"]["acquisition"]
    assert acq["status"] == "acquired"
    trimmed = Path(acq["file"])
    assert trimmed.is_file()
    assert abs(probe_duration(trimmed) - 20.0) < 0.5   # song_segment.duration
    assert acq["license"] == "all-rights-reserved"
    assert acq["links"]["soundcloud"] == "https://soundcloud.com/x/y"
    assert acq["links"]["spotify"] == "sp"              # pre-existing kept
    assert patch["layers_status"] == {"music_acquisition": "ok"}


def test_policy_link_only_skips_download(tmp_path, monkeypatch, song_mp3):
    ws = Workspace(tmp_path / "job")
    analyzer = make_analyzer(monkeypatch, cand(), song_mp3,
                             policy="link_only")
    called = []
    monkeypatch.setattr(analyzer, "downloader",
                        lambda c, out_dir: called.append(1))
    patch = analyzer.run(music_manifest(), ws)
    acq = patch["audio"]["music"]["acquisition"]
    assert acq["status"] == "skipped"
    assert acq["file"] is None
    assert acq["links"]["soundcloud"] == "https://soundcloud.com/x/y"
    assert not called


def test_policy_licensed_only_blocks_reserved(tmp_path, monkeypatch,
                                              song_mp3):
    ws = Workspace(tmp_path / "job")
    analyzer = make_analyzer(monkeypatch,
                             cand(license="all-rights-reserved"),
                             song_mp3, policy="licensed_only")
    patch = analyzer.run(music_manifest(), ws)
    assert patch["audio"]["music"]["acquisition"]["status"] == "skipped"


def test_policy_licensed_only_allows_cc(tmp_path, monkeypatch, song_mp3):
    ws = Workspace(tmp_path / "job")
    analyzer = make_analyzer(monkeypatch, cand(license="cc-by"),
                             song_mp3, policy="licensed_only")
    patch = analyzer.run(music_manifest(), ws)
    assert patch["audio"]["music"]["acquisition"]["status"] == "acquired"


def test_no_candidates_marks_failed(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "job")
    analyzer = MusicAcquisition()
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "always")
    monkeypatch.setattr(analyzer, "prober", lambda query: None)
    patch = analyzer.run(music_manifest(), ws)
    assert patch["audio"]["music"]["acquisition"]["status"] == "failed"
    assert patch["layers_status"] == {"music_acquisition": "failed"}


def test_no_music_detected_skips(tmp_path):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file="x.mp4"))
    patch = MusicAcquisition().run(m, ws)
    assert patch == {"layers_status": {"music_acquisition": "skipped"}}


def test_wrong_duration_candidate_rejected_end_to_end(tmp_path, monkeypatch,
                                                      song_mp3):
    # manifest says the song is 428s; a 120s candidate (cover/preview) must
    # be rejected by the duration gate, leaving no candidate -> failed
    ws = Workspace(tmp_path / "job")
    analyzer = make_analyzer(monkeypatch, cand(duration=120.0), song_mp3)
    patch = analyzer.run(music_manifest(), ws)
    assert patch["audio"]["music"]["acquisition"]["status"] == "failed"


def test_one_failing_resolver_does_not_poison_chain(tmp_path, monkeypatch,
                                                    song_mp3):
    ws = Workspace(tmp_path / "job")
    analyzer = MusicAcquisition()
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "always")

    def prober(query):
        if query.startswith("scsearch"):
            raise RuntimeError("extractor exploded")
        return cand(url="https://youtube.com/watch?v=1", source="youtube")

    monkeypatch.setattr(analyzer, "prober", prober)
    monkeypatch.setattr(analyzer, "downloader", lambda c, out: song_mp3)
    patch = analyzer.run(music_manifest(), ws)
    assert patch["audio"]["music"]["acquisition"]["status"] == "acquired"


def test_licensed_only_prefers_cc_candidate_later_in_chain(tmp_path,
                                                           monkeypatch,
                                                           song_mp3):
    # SoundCloud hit is all-rights-reserved but the YouTube hit is CC:
    # licensed_only must pick the CC one and download it
    ws = Workspace(tmp_path / "job")
    analyzer = MusicAcquisition()
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "licensed_only")
    by_query = {
        "scsearch1:Daft Punk Around the World":
            cand(license="all-rights-reserved", source="soundcloud"),
        "ytsearch1:Daft Punk Around the World":
            cand(url="https://youtube.com/watch?v=1",
                 license="Creative Commons Attribution license (reuse allowed)",
                 source="youtube"),
    }
    monkeypatch.setattr(analyzer, "prober", lambda q: by_query.get(q))
    monkeypatch.setattr(analyzer, "downloader", lambda c, out: song_mp3)
    patch = analyzer.run(music_manifest(), ws)
    acq = patch["audio"]["music"]["acquisition"]
    assert acq["status"] == "acquired"
    assert "Creative Commons" in acq["license"]

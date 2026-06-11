# magicat/modules/audio/analyzer.py
"""Audio analysis: extract -> (optional separation, Task 7) -> windows ->
provider chain recognition -> offset alignment -> audio.music patch."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from magicat import config
from magicat.core.registry import register_analyzer
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest
from magicat.modules.audio.extract import cut_windows, extract_wav, wav_duration
from magicat.modules.audio.identify import align, recognize_windows
from magicat.modules.audio.providers import providers_from_env
from magicat.modules.audio import separation

log = logging.getLogger(__name__)

WINDOW_S = 12.0


@register_analyzer
class AudioAnalyzer:
    name = "audio_analysis"
    layer = "music"
    needs_gpu = False
    provider_factory = staticmethod(providers_from_env)  # injectable

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        """Run the audio layer.

        The wall-clock budget (config.music_timeout_s, CEO decision
        2026-06-11) covers the RECOGNITION phase only: extraction and the
        optional separation are local, bounded work, so the clock starts
        after the windows are cut. On expiry with no match the layer is
        skipped and the job moves on.
        """
        providers = self.provider_factory()
        if not providers:
            return {"layers_status": {"music": "skipped"}}

        wav = extract_wav(Path(manifest.source.file),
                          ws.media_dir / "audio.wav")
        fingerprint_input = wav
        speech_stem: str | None = None
        if separation.enabled():
            music_bed, vocals = separation.split_music_bed(
                wav, ws.media_dir / "stems")
            fingerprint_input = music_bed
            speech_stem = str(vocals)
        video_duration = manifest.source.duration or wav_duration(wav)

        windows = cut_windows(fingerprint_input,
                              ws.media_dir / "audio_windows",
                              window_s=WINDOW_S)
        budget = config.music_timeout_s()
        deadline = time.monotonic() + budget
        music = None
        all_providers_dead = True
        attempted_calls = 0
        timed_out = False
        for provider in providers:   # primary first; fallback on no result
            if time.monotonic() >= deadline:
                timed_out = True
                break
            matches, errors, provider_timed_out = recognize_windows(
                windows, provider, deadline=deadline)
            if provider_timed_out:
                timed_out = True
            attempted_calls += len(matches)
            if errors < len(matches):   # at least one non-error result
                all_providers_dead = False
            music = align(windows, matches,
                          video_duration=video_duration,
                          window_s=WINDOW_S)
            if music is not None:
                break

        if music is None:
            if attempted_calls and all_providers_dead:
                # bad keys / quota exhausted on every attempted window of
                # every provider: report a failure, not a confident
                # "no music detected"
                return {"layers_status": {"music": "failed"}}
            if timed_out:
                log.warning("music identification timed out after %.0fs - "
                            "skipping layer", budget)
                return {"layers_status": {"music": "skipped"}}

        audio = manifest.audio.model_dump(mode="json")
        if music is None:
            audio["music"]["detected"] = False
        else:
            audio["music"] = music
        audio["speech_stem"] = speech_stem
        return {"audio": audio, "layers_status": {"music": "ok"}}

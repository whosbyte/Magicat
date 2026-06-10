# magicat/modules/audio/analyzer.py
"""Audio analysis: extract -> (optional separation, Task 7) -> windows ->
provider chain recognition -> offset alignment -> audio.music patch."""
from __future__ import annotations

from pathlib import Path

from magicat.core.registry import register_analyzer
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest
from magicat.modules.audio.extract import cut_windows, extract_wav, wav_duration
from magicat.modules.audio.identify import align, recognize_windows
from magicat.modules.audio.providers import providers_from_env

WINDOW_S = 12.0


@register_analyzer
class AudioAnalyzer:
    name = "audio_analysis"
    layer = "music"
    needs_gpu = False
    provider_factory = staticmethod(providers_from_env)  # injectable

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        providers = self.provider_factory()
        if not providers:
            return {"layers_status": {"music": "skipped"}}

        wav = extract_wav(Path(manifest.source.file),
                          ws.media_dir / "audio.wav")
        fingerprint_input = wav  # Task 7 swaps in the separated music bed
        video_duration = manifest.source.duration or wav_duration(wav)

        windows = cut_windows(fingerprint_input,
                              ws.media_dir / "audio_windows",
                              window_s=WINDOW_S)
        music = None
        for provider in providers:   # primary first; fallback on no result
            matches = recognize_windows(windows, provider)
            music = align(windows, matches,
                          video_duration=video_duration,
                          window_s=WINDOW_S)
            if music is not None:
                break

        audio = manifest.audio.model_dump(mode="json")
        if music is None:
            audio["music"]["detected"] = False
        else:
            audio["music"] = music
        return {"audio": audio, "layers_status": {"music": "ok"}}

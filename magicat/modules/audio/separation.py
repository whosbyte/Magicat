# magicat/modules/audio/separation.py
"""Optional music/speech separation via demucs-infer (htdemucs).

Install: pip install -e .[separation]. All heavy imports are inside
functions; importing this module costs nothing. Music bed = drums+bass+other
(vocals stem carries both sung vocals and voiceover speech).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def available() -> bool:
    return importlib.util.find_spec("demucs_infer") is not None


def enabled() -> bool:
    mode = os.environ.get("MAGICAT_USE_SEPARATION", "auto")
    if mode == "never":
        return False
    return available()


def split_music_bed(wav: Path, out_dir: Path) -> tuple[Path, Path]:
    """Returns (music_bed_path, vocals_path). Caller guards with enabled()."""
    import soundfile as sf
    import torch
    from demucs_infer.apply import apply_model
    from demucs_infer.pretrained import get_model

    model = get_model("htdemucs")   # ~80 MB one-time weight download
    model.eval()

    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T)            # (channels, samples)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)              # mono -> stereo
    if sr != model.samplerate:
        import torchaudio
        audio = torchaudio.functional.resample(audio, sr, model.samplerate)

    ref = audio.mean(0)
    normalized = (audio - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        # NOTE: do not pass `segment` - htdemucs has a trained-segment cap
        # and apply_model falls back to the model's own value when omitted
        sources = apply_model(model, normalized[None], device="cpu",
                              shifts=0, split=True, overlap=0.25,
                              progress=False)[0]
    sources = sources * ref.std() + ref.mean()  # (4, 2, samples)

    stems = dict(zip(model.sources, sources))   # drums, bass, other, vocals
    music_bed = stems["drums"] + stems["bass"] + stems["other"]

    out_dir.mkdir(parents=True, exist_ok=True)
    music_path = out_dir / "music_bed.wav"
    vocals_path = out_dir / "vocals.wav"
    sf.write(str(music_path), music_bed.T.numpy(), model.samplerate)
    sf.write(str(vocals_path), stems["vocals"].T.numpy(), model.samplerate)
    return music_path, vocals_path

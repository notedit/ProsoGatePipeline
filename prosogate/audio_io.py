"""Audio I/O helpers — mono float32 in [-1, 1], soundfile-backed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def read_wav(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Return (mono_float32, sr). Resamples if target_sr is given and differs."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio.astype(np.float32, copy=False), sr


def write_wav(path: str | Path, audio: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype=subtype)


def slice_audio(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    s = max(0, int(start_sec * sr))
    e = min(len(audio), int(end_sec * sr))
    return audio[s:e]

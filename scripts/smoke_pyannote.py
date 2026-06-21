"""Smoke test: load community-1 diarization + standalone embedding from local dir.

Run after the four pyannote model dirs are present under `models/pyannote/`.
Uses in-memory waveform dicts to bypass torchcodec (which is broken in this env).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "models" / "pyannote"

DIAR_DIR = MODEL_ROOT / "pyannote--speaker-diarization-community-1"
EMBED_DIR = MODEL_ROOT / "pyannote--embedding"
TEST_WAV = ROOT / "examples" / "audio" / "spk001" / "session_001.wav"


def load_wav_for_pyannote(path: Path) -> dict:
    """Return {'waveform': (1, T) float32 tensor, 'sample_rate': int}. 16kHz mono."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        # crude resample via torch.nn.functional.interpolate to avoid librosa import here
        import numpy as np
        ratio = 16000 / sr
        n_target = int(len(audio) * ratio)
        idx = (np.arange(n_target) / ratio).astype(np.int64)
        idx = np.clip(idx, 0, len(audio) - 1)
        audio = audio[idx]
        sr = 16000
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    return {"waveform": waveform, "sample_rate": sr}


def _safe_cuda() -> bool:
    """torch.cuda.is_available() can lie when driver/cuda mismatch — actually try."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


def test_diarization() -> None:
    print("=== diarization (community-1) ===")
    from pyannote.audio import Pipeline

    cfg = DIAR_DIR / "config.yaml"
    print(f"  loading from: {cfg}")
    pipeline = Pipeline.from_pretrained(str(cfg))

    if _safe_cuda():
        pipeline.to(torch.device("cuda"))
        print("  device: cuda")
    else:
        print("  device: cpu")

    payload = load_wav_for_pyannote(TEST_WAV)
    print(f"  audio: {payload['waveform'].shape} @ {payload['sample_rate']}Hz")
    diar = pipeline(payload)
    annotation = getattr(diar, "speaker_diarization", diar)
    n_turns = 0
    speakers = set()
    for turn, _, label in annotation.itertracks(yield_label=True):
        n_turns += 1
        speakers.add(label)
        if n_turns <= 6:
            print(f"  turn: {turn.start:.2f}-{turn.end:.2f}s -> {label}")
    print(f"  total turns: {n_turns}, speakers: {sorted(speakers)}")
    if hasattr(diar, "speaker_embeddings") and diar.speaker_embeddings is not None:
        print(f"  speaker_embeddings shape: {diar.speaker_embeddings.shape}")
    assert n_turns > 0, "diarization produced 0 turns"
    print("  OK")


def test_embedding() -> None:
    print("=== standalone embedding (pyannote/embedding) ===")
    from pyannote.audio import Inference, Model

    ckpt = EMBED_DIR / "pytorch_model.bin"
    print(f"  loading: {ckpt}")
    model = Model.from_pretrained(str(ckpt))
    if _safe_cuda():
        model.to(torch.device("cuda"))
    inference = Inference(model, window="whole")

    payload = load_wav_for_pyannote(TEST_WAV)
    emb = inference(payload)
    shape = emb.data.shape if hasattr(emb, "data") else getattr(emb, "shape", "?")
    print(f"  embedding shape: {shape}")
    print("  OK")


def main() -> int:
    if not TEST_WAV.exists():
        print(f"FAIL: {TEST_WAV} not found. Run scripts/gen_smoke_data.py first.")
        return 1
    for d in [DIAR_DIR, EMBED_DIR]:
        if not d.exists():
            print(f"FAIL: missing {d}")
            return 1

    # torchcodec has a noisy import-time warning about libnvrtc; silence it
    warnings.filterwarnings("ignore", message=".*torchcodec.*")

    try:
        test_diarization()
        test_embedding()
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate synthetic smoke-test data: 2 short Chinese clips with ground-truth text.

Produces:
  examples/audio/spk001/session_001.wav  (~40s, 24kHz mono)
  examples/audio/spk002/session_001.wav  (~35s, 24kHz mono)
  examples/text/spk001/session_001.txt
  examples/text/spk002/session_001.txt
  examples/metadata.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR = 24000


def synth_speech(duration_sec: float, base_f0: float = 200.0, seed: int = 0) -> np.ndarray:
    """Crude voiced/unvoiced signal — not real speech, but gives F0/VAD something to bite."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * SR)
    t = np.arange(n) / SR
    # Pseudo prosody: F0 modulated by slow sine + noise
    f0 = base_f0 + 30 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 5, n)
    phase = np.cumsum(2 * np.pi * f0 / SR)
    voiced = 0.3 * np.sin(phase)
    voiced += 0.1 * np.sin(2 * phase) + 0.05 * np.sin(3 * phase)
    # Insert short silences every ~3 seconds
    audio = voiced
    for chunk_start in range(0, n - SR, 3 * SR):
        sil = chunk_start + int(2.5 * SR)
        audio[sil : sil + int(0.4 * SR)] = 0
    audio += rng.normal(0, 0.005, n)  # mild noise
    return audio.astype(np.float32)


def main() -> None:
    audio_dir = ROOT / "examples" / "audio"
    text_dir = ROOT / "examples" / "text"
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "speaker_id": "spk001",
            "session": "session_001",
            "duration": 40.0,
            "base_f0": 220.0,
            "seed": 1,
            "text": (
                "今天的天气非常适合出门散步。"
                "我们刚刚下了课，正好可以去公园走一走。"
                "你想一起去吗？我已经准备好了。"
                "公园里的樱花开得正盛，错过就要再等一年。"
                "记得带上相机，这种景色值得拍下来。"
            ),
            "language": "zh",
            "domain": "casual",
            "recording_type": "studio",
        },
        {
            "speaker_id": "spk002",
            "session": "session_001",
            "duration": 35.0,
            "base_f0": 160.0,
            "seed": 2,
            "text": (
                "人工智能正在改变我们的生活方式。"
                "从语音助手到自动驾驶，技术在快速进步。"
                "我们既要拥抱变化，也要保持思考。"
                "未来属于那些愿意学习和适应的人。"
            ),
            "language": "zh",
            "domain": "tech",
            "recording_type": "studio",
        },
    ]

    rows = [["audio_path", "speaker_id", "language", "domain", "recording_type", "transcript_path"]]
    for s in samples:
        spk_audio_dir = audio_dir / s["speaker_id"]
        spk_text_dir = text_dir / s["speaker_id"]
        spk_audio_dir.mkdir(parents=True, exist_ok=True)
        spk_text_dir.mkdir(parents=True, exist_ok=True)

        wav_path = spk_audio_dir / f"{s['session']}.wav"
        txt_path = spk_text_dir / f"{s['session']}.txt"

        audio = synth_speech(s["duration"], s["base_f0"], s["seed"])
        sf.write(wav_path, audio, SR, subtype="PCM_16")
        txt_path.write_text(s["text"], encoding="utf-8")

        rows.append([
            f"examples/audio/{s['speaker_id']}/{s['session']}.wav",
            s["speaker_id"],
            s["language"],
            s["domain"],
            s["recording_type"],
            f"examples/text/{s['speaker_id']}/{s['session']}.txt",
        ])
        print(f"wrote {wav_path} ({s['duration']:.1f}s)")

    meta_path = ROOT / "examples" / "metadata.csv"
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()

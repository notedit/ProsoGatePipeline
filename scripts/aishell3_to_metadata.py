"""Adapt AISHELL-3 (already-segmented short utterances) into long audios for
the full ProsoGate pipeline.

AISHELL-3 layout (after download from HuggingFace AISHELL/AISHELL-3):
  <root>/test/content.txt              one line per wav: "<file>\t<char pinyin char pinyin ...>"
  <root>/test/wav/<spkid>/<file>.wav   44.1 kHz mono studio recordings, ~1-10s each

Strategy: For each speaker, concatenate N short utts (with inter-utt silence)
into a synthetic long audio (~30-90s, single-speaker, all-studio quality).
This makes the input look like the "long audio" that ProsoGate expects, while
preserving the underlying studio quality that distinguishes AISHELL-3 from
RAMC phone-quality data.

Output:
  <out-root>/audio/<spkid>_part<idx>.wav    concatenated long audio
  <out-root>/text/<spkid>_part<idx>.txt     concatenated text (filler-free char stream)
  <out-root>/metadata.csv                    pipeline ingest manifest
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_content_txt(path: Path) -> dict[str, str]:
    """Return {wav_filename: char_only_text} extracted from AISHELL-3 content.txt."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            fn, annot = parts
            # annot interleaves "char pinyin char pinyin ..." separated by spaces.
            # Keep only CJK characters.
            chars = re.findall(r"[一-鿿]", annot)
            out[fn] = "".join(chars)
    return out


def concat_speaker(
    wavs: list[Path],
    texts: list[str],
    target_min_sec: float,
    target_max_sec: float,
    silence_min_ms: int,
    silence_max_ms: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, str, int]]:
    """Walk wavs in order, group into parts of target duration."""
    parts: list[tuple[np.ndarray, str, int]] = []
    cur_audio: list[np.ndarray] = []
    cur_text: list[str] = []
    cur_dur = 0.0
    sr_ref: int | None = None

    def _flush():
        if cur_audio:
            audio = np.concatenate(cur_audio)
            text = "".join(cur_text)
            parts.append((audio, text, sr_ref))

    for w, t in zip(wavs, texts):
        try:
            audio, sr = sf.read(w, dtype="float32", always_2d=False)
        except Exception:
            continue
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            continue  # skip mixed-SR
        dur = len(audio) / sr

        # If adding this utt exceeds max, flush.
        if cur_dur + dur > target_max_sec and cur_dur >= target_min_sec:
            _flush()
            cur_audio = []
            cur_text = []
            cur_dur = 0.0

        # Append inter-utt silence (skip before first utt of a part).
        if cur_audio:
            sil_ms = int(rng.integers(silence_min_ms, silence_max_ms + 1))
            sil_n = int(sr * sil_ms / 1000.0)
            cur_audio.append(np.zeros(sil_n, dtype=np.float32))
            cur_dur += sil_n / sr

        cur_audio.append(audio)
        cur_text.append(t)
        cur_dur += dur

    # Final flush
    if cur_audio and cur_dur >= target_min_sec:
        _flush()
    return parts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", default="/workspace/data/aishell3_sample",
                   help="AISHELL-3 download root (containing test/content.txt + test/wav/)")
    p.add_argument("--out-root", default="data_test/aishell3",
                   help="Where to write metadata.csv + audio/ + text/")
    p.add_argument("--speakers", nargs="*", default=None,
                   help="Specific speaker IDs to include (default: all under test/wav/)")
    p.add_argument("--target-min-sec", type=float, default=45.0)
    p.add_argument("--target-max-sec", type=float, default=75.0)
    p.add_argument("--silence-min-ms", type=int, default=400)
    p.add_argument("--silence-max-ms", type=int, default=900)
    p.add_argument("--max-parts-per-speaker", type=int, default=2,
                   help="Cap concatenated parts per speaker (keeps the dataset bounded)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.source_root)
    out = Path(args.out_root)
    audio_dir = out / "audio"
    text_dir = out / "text"
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    content_path = src / "test" / "content.txt"
    if not content_path.exists():
        # Try root-level content.txt
        content_path = src / "content.txt"
    text_map = parse_content_txt(content_path)
    print(f"[content] {len(text_map)} utterances annotated in {content_path}")

    wav_root = src / "test" / "wav"
    if args.speakers:
        speakers = [s for s in args.speakers if (wav_root / s).exists()]
    else:
        speakers = sorted(d.name for d in wav_root.iterdir() if d.is_dir())
    print(f"[speakers] {len(speakers)} found: {speakers}")

    rng = np.random.default_rng(args.seed)
    csv_rows = [["audio_path", "speaker_id", "language", "domain", "recording_type", "transcript_path"]]
    total_parts = 0

    for spk in speakers:
        spk_dir = wav_root / spk
        utts = sorted(spk_dir.glob("*.wav"))
        if not utts:
            print(f"  [skip] {spk}: no wavs")
            continue
        # Pair wav files with their text annotation, only keep those with text
        pairs = [(u, text_map.get(u.name, "")) for u in utts]
        pairs = [(u, t) for u, t in pairs if t]
        wavs = [u for u, _ in pairs]
        texts = [t for _, t in pairs]

        parts = concat_speaker(
            wavs, texts,
            args.target_min_sec, args.target_max_sec,
            args.silence_min_ms, args.silence_max_ms,
            rng,
        )
        if not parts:
            print(f"  [skip] {spk}: no parts produced")
            continue
        parts = parts[: args.max_parts_per_speaker]

        for i, (audio, text, sr) in enumerate(parts):
            base = f"{spk}_part{i:02d}"
            out_wav = audio_dir / f"{base}.wav"
            out_txt = text_dir / f"{base}.txt"
            sf.write(out_wav, audio, sr, subtype="PCM_16")
            out_txt.write_text(text, encoding="utf-8")
            csv_rows.append([
                str(out_wav.relative_to(Path.cwd())) if out_wav.is_relative_to(Path.cwd()) else str(out_wav),
                spk,
                "zh",
                "tts_corpus",
                "studio",
                str(out_txt.relative_to(Path.cwd())) if out_txt.is_relative_to(Path.cwd()) else str(out_txt),
            ])
            total_parts += 1
            print(f"  [{spk}] part{i:02d} dur={len(audio)/sr:.1f}s chars={len(text)} -> {out_wav.name}")

    meta = out / "metadata.csv"
    with open(meta, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"\nWrote metadata.csv with {total_parts} entries -> {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

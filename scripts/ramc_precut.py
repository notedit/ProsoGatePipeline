"""Pre-cut RAMC long dialogues into ≤target_sec chunks using silero-VAD silence boundaries.

RAMC test set has 43 dialogues, each 15-30 min. pyannote diarization on a single
30-min clip on CPU takes ~70 minutes. We pre-cut to ≤10 min at long silences so
diarization runs in ~20 min per chunk.

Outputs:
  data_test/ramc_cut/audio/{base}_p{N}.wav    — cut clips
  data_test/ramc_cut/metadata.csv             — ingest manifest input
  data_test/ramc_cut/cut_offsets.jsonl        — per-clip offset back to original

Strategy:
  1. Load silero-vad, get speech timestamps.
  2. Walk forward, accumulate speech until clip duration ≥ target_min.
  3. Cut at the next silence longer than min_silence_ms.
  4. If accumulated duration ≥ target_max without a silence, force-cut at the
     largest silence within the window.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prosogate.logging_utils import get_logger

log = get_logger("ramc_precut")


def load_silero_vad():
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps

        model = load_silero_vad()
        return model, get_speech_timestamps
    except Exception as e:
        log.warning(f"silero-vad unavailable ({e}); falling back to energy VAD")
        return None, None


def speech_timestamps_energy(audio: np.ndarray, sr: int) -> list[dict]:
    """Crude RMS-based VAD fallback. Returns [{start_sample, end_sample}, ...]."""
    win = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_frames = max(1, 1 + (len(audio) - win) // hop)
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        seg = audio[s : s + win]
        rms[i] = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
    thr = max(1e-4, np.percentile(rms, 30) * 1.2)
    voiced = rms > thr
    # Group contiguous voiced frames
    out = []
    in_run = False
    run_start = 0
    for i, v in enumerate(voiced):
        if v and not in_run:
            run_start = i * hop
            in_run = True
        elif not v and in_run:
            out.append({"start": run_start, "end": i * hop})
            in_run = False
    if in_run:
        out.append({"start": run_start, "end": len(audio)})
    # Filter tiny segments < 200ms
    out = [s for s in out if (s["end"] - s["start"]) >= 0.2 * sr]
    return out


def cut_long_audio(
    audio: np.ndarray,
    sr: int,
    speech_ts: list[dict],
    target_min_sec: float,
    target_max_sec: float,
    min_silence_sec: float,
) -> list[tuple[float, float]]:
    """Return [(start_sec, end_sec), ...] for cut clips."""
    if not speech_ts:
        return [(0.0, len(audio) / sr)]

    # Build silence list between speech segments
    silences = []
    if speech_ts[0]["start"] > 0:
        silences.append((0, speech_ts[0]["start"]))
    for i in range(1, len(speech_ts)):
        s = speech_ts[i - 1]["end"]
        e = speech_ts[i]["start"]
        if e > s:
            silences.append((s, e))
    if speech_ts[-1]["end"] < len(audio):
        silences.append((speech_ts[-1]["end"], len(audio)))

    cuts = [0.0]
    cur_pos = 0.0
    total_dur = len(audio) / sr

    while cur_pos < total_dur:
        target_end = cur_pos + target_min_sec
        force_end = cur_pos + target_max_sec
        # Find the earliest silence ≥ min_silence_sec AFTER target_end
        chosen_cut = None
        for s_start_samp, s_end_samp in silences:
            s_start = s_start_samp / sr
            s_end = s_end_samp / sr
            if s_start < target_end:
                continue
            if s_start >= force_end:
                # No good silence within range; use largest silence before force_end
                break
            sil_dur = s_end - s_start
            if sil_dur >= min_silence_sec:
                chosen_cut = (s_start + s_end) / 2
                break
        if chosen_cut is None:
            # No qualifying silence — find largest silence in [target_end, force_end]
            candidates = [
                (s_start_samp / sr, s_end_samp / sr)
                for s_start_samp, s_end_samp in silences
                if target_end <= s_start_samp / sr < force_end
            ]
            if candidates:
                candidates.sort(key=lambda x: -(x[1] - x[0]))
                s_start, s_end = candidates[0]
                chosen_cut = (s_start + s_end) / 2
            else:
                chosen_cut = min(force_end, total_dur)

        if chosen_cut >= total_dur - 1.0:
            chosen_cut = total_dur
            cuts.append(chosen_cut)
            break
        cuts.append(chosen_cut)
        cur_pos = chosen_cut

    # Convert cuts -> [(start, end)]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        default="work_test/ramc_ingest.jsonl",
        help="Pre-built RAMC ingest jsonl",
    )
    parser.add_argument("--n-source", type=int, default=5, help="How many source dialogues to take")
    parser.add_argument("--out-root", default="data_test/ramc_cut")
    parser.add_argument("--target-min-sec", type=float, default=480, help="Aim for ≥ this duration")
    parser.add_argument("--target-max-sec", type=float, default=600, help="Hard cap (10 min)")
    parser.add_argument("--min-silence-sec", type=float, default=0.6)
    args = parser.parse_args()

    out_root = ROOT / args.out_root
    audio_dir = out_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    vad_model, ts_fn = load_silero_vad()

    cut_records = []
    csv_rows = [["audio_path", "speaker_id", "language", "domain", "recording_type", "transcript_path"]]

    with open(ROOT / args.input_manifest) as f:
        sources = [json.loads(l) for l in f][: args.n_source]

    for src in sources:
        wav_path = src["audio_path"]
        audio_id = src["audio_id"]
        log.info(f"loading {wav_path}")
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        # silero needs 16kHz
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Run VAD
        if vad_model is not None and ts_fn is not None:
            import torch
            wave_t = torch.from_numpy(audio)
            ts = ts_fn(wave_t, vad_model, sampling_rate=sr)
        else:
            ts = speech_timestamps_energy(audio, sr)
        log.info(f"  {len(ts)} speech segments detected")

        # Cut
        cuts = cut_long_audio(
            audio, sr, ts,
            args.target_min_sec, args.target_max_sec, args.min_silence_sec,
        )
        log.info(f"  cut into {len(cuts)} chunks")

        for i, (cs, ce) in enumerate(cuts):
            base = f"{audio_id}_p{i:02d}"
            out_wav = audio_dir / f"{base}.wav"
            clip = audio[int(cs * sr) : int(ce * sr)]
            sf.write(out_wav, clip, sr, subtype="PCM_16")
            dur = ce - cs
            log.info(f"    -> {base}.wav  [{cs:.1f}-{ce:.1f}] ({dur:.1f}s)")
            cut_records.append({
                "cut_id": base,
                "source_audio_id": audio_id,
                "start": cs,
                "end": ce,
                "duration": dur,
                "out_path": str(out_wav.relative_to(ROOT)),
            })
            csv_rows.append([
                str(out_wav.relative_to(ROOT)),
                "multi", "zh", "conversation", "interview", "",
            ])

    # Write metadata
    meta_csv = out_root / "metadata.csv"
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)
    log.info(f"wrote {meta_csv} ({len(csv_rows)-1} rows)")

    # Write cut offsets (for joining results back to original time)
    offsets_path = out_root / "cut_offsets.jsonl"
    with open(offsets_path, "w", encoding="utf-8") as f:
        for r in cut_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"wrote {offsets_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

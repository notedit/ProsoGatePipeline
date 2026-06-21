"""Convert WenetSpeech4TTS Premium subset to ProsoGate stage-08 manifest.

WenetSpeech4TTS already provides:
  - 16 kHz mono PCM_16 wavs
  - Character-level timestamps in the .txt sidecar

So we can skip stages 1-7 entirely and feed the result directly into stage 09
(spk_consistency) onwards. This validates §9-§14 on a real Chinese audiobook /
podcast distribution.

Output:
  - work/08_fine_segment.jsonl    (utt-level manifest)
  - tts_dataset/wavs/{utt_id}.wav (re-encoded to 24 kHz mono, copy from source)
  - tts_dataset/alignments/{utt_id}.json (char-level alignment, utt-relative s)

Usage:
  python scripts/wenetspeech_to_manifest.py \
      --shard-root /workspace/data/wenetspeech4tts/extracted/WenetSpeech4TTS_Premium_0 \
      --n-utts 2000 \
      --target-sr 24000

The utt_id is derived from the source filename ({source}_{episode}_{seg_range}),
and speaker_id is set to the {source}_{episode} pair (each YouTube/podcast episode
is treated as one speaker, which is approximately right for audiobook/podcast).
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prosogate.audio_io import read_wav, write_wav
from prosogate.logging_utils import get_logger
from prosogate.manifest import write_jsonl

log = get_logger("wenet_to_manifest")


def parse_txt(txt_path: Path) -> tuple[str, list[tuple[int, int]]] | None:
    """Parse the WenetSpeech4TTS .txt format.

    Format:
        <utt_id>\t<text>
        \t[[s,e], ...]
    """
    raw = txt_path.read_text(encoding="utf-8").strip()
    lines = raw.split("\n")
    if len(lines) < 2:
        return None
    try:
        first = lines[0].split("\t", 1)
        if len(first) != 2:
            return None
        text = first[1]
        ts_line = lines[1].lstrip("\t").strip()
        ts = ast.literal_eval(ts_line)
        if not isinstance(ts, list):
            return None
        return text, [tuple(x) for x in ts]
    except Exception as e:
        log.debug(f"parse fail {txt_path}: {e}")
        return None


def chars_with_ts(text: str, ts: list[tuple[int, int]]) -> list[dict]:
    """Pair non-whitespace chars with timestamps. Pad missing trailing chars."""
    char_list = [c for c in text if c.strip()]
    n = min(len(char_list), len(ts))
    out = []
    for ch, (s_ms, e_ms) in zip(char_list[:n], ts[:n]):
        out.append({
            "char": ch,
            "start": s_ms / 1000.0,
            "end": e_ms / 1000.0,
            "confidence": 0.95,  # WenetSpeech4TTS char timestamps are derived from forced alignment, assume high
        })
    # Trailing punctuation: attach to last char's end
    if len(char_list) > n and out:
        last_end = out[-1]["end"]
        for ch in char_list[n:]:
            out.append({
                "char": ch,
                "start": last_end,
                "end": last_end,
                "confidence": 0.95,
            })
    return out


def derive_speaker_id(stem: str) -> tuple[str, str]:
    """utt_id stem -> (speaker_id, source_audio_id).

    Stem looks like: X0000000021_240514196_S00041 or Y0000018236_kvsv5nDmOZ8_S00435-S00438
    speaker_id = {source}_{episode}  -> X0000000021_240514196
    source_audio_id = same           -> one episode == one source
    """
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].startswith("S"):
        episode = parts[0]
    else:
        episode = stem
    return episode, episode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-root",
        required=True,
        help="Path to extracted WenetSpeech4TTS_Premium_N/ directory",
    )
    parser.add_argument(
        "--out-manifest",
        default="work/08_fine_segment.jsonl",
        help="Output stage-08 manifest path (relative to project root)",
    )
    parser.add_argument(
        "--out-wavs-root",
        default="tts_dataset/wavs",
        help="Where to write re-sampled wavs",
    )
    parser.add_argument(
        "--out-align-root",
        default="tts_dataset/alignments",
        help="Where to write per-utt alignment json",
    )
    parser.add_argument("--n-utts", type=int, default=2000)
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=0,
        help="If >0, cap the number of distinct speaker (episode) IDs sampled",
    )
    parser.add_argument(
        "--min-utts-per-speaker",
        type=int,
        default=0,
        help="If >0, only sample from speakers with at least this many utts in the source",
    )
    parser.add_argument(
        "--episode-filter-file",
        default="",
        help="Optional text file with one episode id per line; restrict sampling to these",
    )
    args = parser.parse_args()

    shard_root = Path(args.shard_root)
    wavs_dir = shard_root / "wavs"
    txts_dir = shard_root / "txts"
    assert wavs_dir.exists(), f"missing {wavs_dir}"
    assert txts_dir.exists(), f"missing {txts_dir}"

    # 1. enumerate utts (deterministic, sorted)
    log.info("scanning wavs ...")
    all_wavs = sorted(wavs_dir.glob("*.wav"))
    log.info(f"found {len(all_wavs)} wavs")

    # Group by speaker first
    spk_to_wavs: dict[str, list[Path]] = {}
    for w in all_wavs:
        spk, _ = derive_speaker_id(w.stem)
        spk_to_wavs.setdefault(spk, []).append(w)
    log.info(f"distinct episodes in source: {len(spk_to_wavs)}")

    # Optional filters
    if args.min_utts_per_speaker > 0:
        before = len(spk_to_wavs)
        spk_to_wavs = {
            s: ws for s, ws in spk_to_wavs.items() if len(ws) >= args.min_utts_per_speaker
        }
        log.info(
            f"min_utts_per_speaker={args.min_utts_per_speaker}: kept {len(spk_to_wavs)}/{before} episodes"
        )

    if args.episode_filter_file:
        wanted = set(Path(args.episode_filter_file).read_text().split())
        spk_to_wavs = {s: ws for s, ws in spk_to_wavs.items() if s in wanted}
        log.info(f"episode_filter_file: kept {len(spk_to_wavs)} episodes")

    # Pick subset of speakers if capped
    rng = random.Random(args.seed)
    spk_pool = sorted(spk_to_wavs.keys())
    rng.shuffle(spk_pool)
    if args.max_speakers > 0 and args.max_speakers < len(spk_pool):
        spk_pool = spk_pool[: args.max_speakers]
        log.info(f"capped to {len(spk_pool)} speakers via max_speakers")

    # Build candidate list: utts per speaker shuffled in
    all_wavs = []
    for spk in spk_pool:
        wavs_for_spk = spk_to_wavs[spk][:]
        rng.shuffle(wavs_for_spk)
        all_wavs.extend(wavs_for_spk)
    log.info(f"candidate wavs: {len(all_wavs)} from {len(spk_pool)} speakers")

    out_manifest_path = ROOT / args.out_manifest
    out_wavs_root = ROOT / args.out_wavs_root
    out_align_root = ROOT / args.out_align_root
    out_wavs_root.mkdir(parents=True, exist_ok=True)
    out_align_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    speakers_seen: set[str] = set()
    n_skipped = {"no_txt": 0, "parse": 0, "dur": 0, "spk_cap": 0}

    for wav_path in all_wavs:
        if len(records) >= args.n_utts:
            break
        stem = wav_path.stem
        txt_path = txts_dir / f"{stem}.txt"
        if not txt_path.exists():
            n_skipped["no_txt"] += 1
            continue
        parsed = parse_txt(txt_path)
        if parsed is None:
            n_skipped["parse"] += 1
            continue
        text, ts = parsed

        speaker_id, source_audio_id = derive_speaker_id(stem)
        speakers_seen.add(speaker_id)

        # Load audio, resample, write
        try:
            audio, sr = read_wav(wav_path, target_sr=args.target_sr)
        except Exception as e:
            log.warning(f"read fail {wav_path}: {e}")
            continue
        duration = len(audio) / sr
        if duration < args.min_duration or duration > args.max_duration:
            n_skipped["dur"] += 1
            continue

        utt_id = stem
        out_wav = out_wavs_root / f"{utt_id}.wav"
        write_wav(out_wav, audio, sr, subtype="PCM_16")

        # Char alignment (utt-relative, in seconds)
        chars = chars_with_ts(text, ts)
        if not chars:
            n_skipped["parse"] += 1
            continue
        confs = [c["confidence"] for c in chars]
        align_conf_mean = float(np.mean(confs))
        align_conf_p10 = float(np.percentile(confs, 10))

        align_obj = {
            "seg_id": utt_id,
            "chars": chars,
            "align_conf_mean": align_conf_mean,
            "align_conf_p10": align_conf_p10,
        }
        align_path = out_align_root / f"{utt_id}.json"
        align_path.write_text(json.dumps(align_obj, ensure_ascii=False))

        rec = {
            "utt_id": utt_id,
            "speaker_id": speaker_id,
            "speaker_label": speaker_id,
            "source_audio_id": source_audio_id,
            "wav": str(out_wav.relative_to(ROOT)),
            "text": text,
            "text_normalized": text,
            "prev_text": "",
            "next_text": "",
            "duration": duration,
            "sample_rate": sr,
            "start": 0.0,
            "end": duration,
            "alignment_json_path": str(align_path.relative_to(ROOT)),
            "align_conf_mean": align_conf_mean,
            "align_conf_p10": align_conf_p10,
            "high_conf_char_ratio": float(np.mean([c["confidence"] >= 0.6 for c in chars])),
            "n_chars": len(chars),
            "source": "wenetspeech4tts_premium",
        }
        records.append(rec)

    write_jsonl(out_manifest_path, records)
    log.info(f"wrote {len(records)} utts to {out_manifest_path}")
    log.info(f"distinct speakers: {len(speakers_seen)}")
    log.info(f"skipped: {n_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

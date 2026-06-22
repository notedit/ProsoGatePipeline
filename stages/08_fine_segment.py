"""Stage 08 — Fine-grained segmentation based on character alignment.

Reads seg-level manifest from stage 07 plus per-seg alignment JSON. Slices each
seg into 3-20s utterances using punctuation + silence + alignment confidence
heuristics, then writes physical 24kHz wavs into tts_dataset/wavs/ and copies
the corresponding char-level alignment slices into tts_dataset/alignments/.

Cut-point priority (design.md §8):
  1. Sentence-ending punctuation (。！？) + post silence >= 200ms + char conf >= 0.9
  2. Long silence >= 300ms not inside a word
  3. Mid-sentence punctuation (，；：) + char conf >= 0.85
  4. Fallback: forced cut at max_duration, snapped to nearest silence

Utterances under min_duration are dropped with reject reason "duration_too_short".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from prosogate.audio_io import read_wav, slice_audio, write_wav
from prosogate.logging_utils import get_logger
from prosogate.manifest import add_reject, read_jsonl, write_jsonl

log = get_logger(__name__)

SENT_END = set("。！？!?.")
SENT_MID = set("，；：,;:")

LONG_SILENCE_SEC = 0.300
PUNCT_SILENCE_SEC = 0.200
HIGH_CONF = 0.90
MID_CONF = 0.85


def _silence_after(chars: list[dict[str, Any]], i: int) -> float:
    if i + 1 >= len(chars):
        return 1e9  # treat end as "infinite" silence
    return float(chars[i + 1]["start"]) - float(chars[i]["end"])


def _find_cuts(
    chars: list[dict[str, Any]], min_dur: float, max_dur: float, preferred_max: float
) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx_exclusive) char slices forming each utt.

    Greedy left-to-right walk: tries to commit at each char index using priority
    rules; honors min/max duration constraints.
    """
    n = len(chars)
    if n == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        anchor_t = float(chars[start]["start"])
        # Default end is the last char of seg.
        commit = -1
        for i in range(start, n):
            ch = chars[i]
            elapsed = float(ch["end"]) - anchor_t
            if elapsed < min_dur:
                continue
            sil = _silence_after(chars, i)
            conf = float(ch.get("confidence", 0.9))
            ch_text = ch.get("char", "")

            # Rule 1: sentence-ending punct + post-silence + high conf
            if (
                ch_text in SENT_END
                and sil >= PUNCT_SILENCE_SEC
                and conf >= HIGH_CONF
                and elapsed >= min_dur
            ):
                commit = i
                # Sentence-end is the strongest signal; if we are within preferred range, take it.
                if elapsed >= 5.0 or i == n - 1:
                    break
            # Rule 2: long silence and not in-word (we treat any inter-char gap as not-in-word
            # since these are characters, not phonemes)
            elif sil >= LONG_SILENCE_SEC and elapsed >= min_dur:
                commit = i
                if elapsed >= preferred_max - 1.0:
                    break
            # Rule 3: mid-sentence punct + mid conf
            elif ch_text in SENT_MID and conf >= MID_CONF and elapsed >= min_dur:
                if commit < 0:
                    commit = i
                if elapsed >= preferred_max:
                    break
            # Rule 4: max-duration fallback — pick the latest silence so far
            if elapsed >= max_dur:
                if commit < 0:
                    # Snap to nearest preceding silence (max gap) within window
                    best = i
                    best_sil = _silence_after(chars, i)
                    for j in range(i, start - 1, -1):
                        s = _silence_after(chars, j)
                        if s > best_sil:
                            best_sil = s
                            best = j
                    commit = best
                break

        if commit < 0:
            # Couldn't find any cut — emit remainder as a single span if it satisfies min_dur.
            commit = n - 1
        spans.append((start, commit + 1))
        start = commit + 1
    return spans


def run(cfg: Any) -> int:
    in_path = cfg.paths.manifests.align
    out_path = cfg.paths.manifests.fine_segment

    fs_cfg = cfg.fine_segment
    min_dur = float(fs_cfg.get("min_duration", 3.0))
    max_dur = float(fs_cfg.get("max_duration", 20.0))
    preferred_max = float(fs_cfg.get("preferred_max", 15.0))

    output_root = Path(cfg.paths.get("output_root", "tts_dataset"))
    wav_dir = output_root / "wavs"
    align_out_dir = output_root / "alignments"
    wav_dir.mkdir(parents=True, exist_ok=True)
    align_out_dir.mkdir(parents=True, exist_ok=True)

    cache_audio_train: dict[str, tuple[np.ndarray, int]] = {}

    # First pass: count emitted utt index per (speaker_label, source_audio_id)
    utt_counters: dict[tuple[str, str], int] = {}

    out_records: list[dict[str, Any]] = []
    n_kept = 0
    n_dropped = 0

    segs = list(read_jsonl(in_path))
    # Sort segs by (source_audio_id, start) so utt indices are stable across runs.
    segs.sort(
        key=lambda r: (
            r.get("source_audio_id") or "",
            float(r.get("start") or 0.0),
            r.get("seg_id") or "",
        )
    )

    for seg in segs:
        if seg.get("status") == "rejected":
            continue
        align_json_path = seg.get("alignment_json_path")
        if not align_json_path or not Path(align_json_path).exists():
            log.warning(f"seg {seg.get('seg_id')} missing alignment json; skipping")
            continue

        with open(align_json_path, "r", encoding="utf-8") as f:
            align_obj = json.load(f)
        chars = align_obj.get("chars", [])
        if not chars:
            continue

        spans = _find_cuts(chars, min_dur, max_dur, preferred_max)

        train_path = seg.get("audio_train_path")
        if not train_path or not Path(train_path).exists():
            log.warning(f"seg {seg.get('seg_id')} missing audio_train_path; skipping")
            continue

        if train_path not in cache_audio_train:
            cache_audio_train[train_path] = read_wav(train_path)
        train_audio, train_sr = cache_audio_train[train_path]

        seg_chars_text = [c.get("char", "") for c in chars]

        for span_idx, (s_idx, e_idx) in enumerate(spans):
            sub_chars = chars[s_idx:e_idx]
            if not sub_chars:
                continue
            u_start = float(sub_chars[0]["start"])
            u_end = float(sub_chars[-1]["end"])
            duration = u_end - u_start

            speaker_label = seg.get("speaker_label") or seg.get("speaker_id") or "spk"
            speaker_id = seg.get("speaker_id") or speaker_label
            source_audio_id = seg.get("source_audio_id") or "src"
            key = (speaker_label, source_audio_id)
            idx = utt_counters.get(key, 0) + 1
            utt_counters[key] = idx
            utt_id = f"{speaker_label}_{source_audio_id}_{idx:06d}"

            text = "".join(c.get("char", "") for c in sub_chars)
            prev_text = "".join(seg_chars_text[max(0, s_idx - 20):s_idx])
            next_text = "".join(seg_chars_text[e_idx:e_idx + 20])

            rec: dict[str, Any] = {
                "utt_id": utt_id,
                "speaker_id": speaker_id,
                "speaker_label": speaker_label,
                "source_audio_id": source_audio_id,
                "seg_id": seg.get("seg_id"),
                "audio_align_path": seg.get("audio_align_path"),
                "audio_train_path": seg.get("audio_train_path"),
                "text": text,
                "prev_text": prev_text,
                "next_text": next_text,
                "start": u_start,
                "end": u_end,
                "duration": float(duration),
                "align_conf_mean": float(
                    np.mean([c.get("confidence", 0.9) for c in sub_chars])
                ),
            }
            # Per-utt timing-derived align signals (server confidence is hardcoded,
            # so we compute these from the chars actually inside this utt slice).
            char_durs = np.array(
                [max(0.0, float(c["end"]) - float(c["start"])) for c in sub_chars],
                dtype=np.float64,
            )
            est_speech = float(char_durs.sum())
            rec["align_coverage"] = float(est_speech / duration) if duration > 0 else 0.0
            rec["align_degenerate_char_ratio"] = (
                float(np.mean((char_durs < 0.02) | (char_durs > 0.5))) if len(char_durs) else 1.0
            )
            n_text_chars = sum(1 for ch in text if ch.strip())
            rec["align_char_match_ratio"] = (
                float(len(sub_chars) / n_text_chars) if n_text_chars else 0.0
            )
            # Carry source-level QC metrics through so stage 12 scoring isn't constant.
            for qc_k in ("snr_db", "lufs", "peak_db", "clipping_ratio", "effective_bw_hz"):
                if seg.get(qc_k) is not None:
                    rec[qc_k] = seg[qc_k]

            if duration < min_dur:
                add_reject(rec, "duration_too_short")
                out_records.append(rec)
                n_dropped += 1
                continue

            # Slice 24kHz training audio and write physical wav.
            clip = slice_audio(train_audio, train_sr, u_start, u_end)
            wav_path = wav_dir / f"{utt_id}.wav"
            write_wav(wav_path, clip, train_sr)
            rec["wav"] = str(wav_path)
            rec["sample_rate"] = train_sr

            # Char-level alignment slice for the utt.
            align_out_path = align_out_dir / f"{utt_id}.json"
            with open(align_out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "utt_id": utt_id,
                        "chars": sub_chars,
                        "align_conf_mean": rec["align_conf_mean"],
                    },
                    f,
                    ensure_ascii=False,
                )
            rec["alignment_json_path"] = str(align_out_path)

            out_records.append(rec)
            n_kept += 1

    n = write_jsonl(out_path, out_records)
    log.info(f"wrote {n} utt records to {out_path} (kept={n_kept}, short_dropped={n_dropped})")
    return 0

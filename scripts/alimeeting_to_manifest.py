"""Parse AliMeeting Eval set TextGrid GT and emit ProsoGate ingest-style manifests.

AliMeeting layout:
  Eval_Ali_far/audio_dir/R{room}_M{meeting}_MS{mic}.wav     (16kHz 8ch, ~25min)
  Eval_Ali_far/textgrid_dir/R{room}_M{meeting}.TextGrid     (4-tier diarization GT)

Each TextGrid has one tier per speaker (e.g. N_SPK8013, N_SPK8014, ...) with
non-empty intervals = speech turns.

Output:
  1. work_test/alimeeting_ingest.jsonl  — audio-level ingest manifest (one row
     per meeting; speaker_id = "multi" since multi-speaker source)
  2. work_test/alimeeting_diarization_gt.jsonl  — GT diarization: per meeting,
     all (start, end, speaker, text) tuples. Use this to evaluate stage 4 DER.

For multi-channel far-field wav we keep channel 0 only (drop the rest); the
8-mic array is overkill for diarization and pyannote expects mono.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prosogate.audio_io import read_wav, write_wav
from prosogate.hash_utils import file_hash
from prosogate.logging_utils import get_logger
from prosogate.manifest import write_jsonl

log = get_logger("alimeeting_to_manifest")


# Praat short-format TextGrid parser. The Eval files use the verbose ("ooTextFile") form.
INTERVAL_RE = re.compile(
    r'intervals \[\d+\]:\s*\n\s*xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*)"',
    re.MULTILINE,
)
TIER_NAME_RE = re.compile(r'name = "([^"]+)"')


def parse_textgrid(path: Path) -> list[dict]:
    """Return list of {tier, start, end, text}."""
    text = path.read_text(encoding="utf-8")
    out = []
    # Split per item to get tier name
    item_blocks = re.split(r"item \[\d+\]:", text)
    for blk in item_blocks[1:]:
        m_name = TIER_NAME_RE.search(blk)
        if not m_name:
            continue
        tier = m_name.group(1)
        for m in INTERVAL_RE.finditer(blk):
            xmin = float(m.group(1))
            xmax = float(m.group(2))
            txt = m.group(3).strip()
            if not txt:
                continue
            out.append({"tier": tier, "start": xmin, "end": xmax, "text": txt})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root",
        default="/workspace/data/alimeeting/Eval/Eval_Ali",
    )
    parser.add_argument(
        "--out-audio-root",
        default="data_test/alimeeting/audio",
        help="Where to write mono wavs (relative to project root)",
    )
    parser.add_argument(
        "--out-manifest",
        default="work_test/alimeeting_ingest.jsonl",
    )
    parser.add_argument(
        "--out-gt",
        default="work_test/alimeeting_diarization_gt.jsonl",
    )
    args = parser.parse_args()

    far_audio = Path(args.eval_root) / "Eval_Ali_far" / "audio_dir"
    far_tg = Path(args.eval_root) / "Eval_Ali_far" / "textgrid_dir"
    assert far_audio.exists(), far_audio
    assert far_tg.exists(), far_tg

    out_audio_root = ROOT / args.out_audio_root
    out_audio_root.mkdir(parents=True, exist_ok=True)

    ingest_records = []
    gt_records = []

    for wav_path in sorted(far_audio.glob("*.wav")):
        # Filename: R8001_M8004_MS801.wav -> meeting id = R8001_M8004
        stem = wav_path.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        meeting_id = f"{parts[0]}_{parts[1]}"
        tg_path = far_tg / f"{meeting_id}.TextGrid"
        if not tg_path.exists():
            log.warning(f"no TextGrid for {stem}")
            continue

        # Convert 8ch -> mono ch0 only (avoid pyannote 8ch confusion)
        log.info(f"converting {stem} ...")
        audio, sr = read_wav(wav_path, target_sr=None)  # keep 16kHz
        # read_wav already collapses to mono via mean; but for AliMeeting we want
        # channel 0 specifically (microphone 1, more consistent). Re-read raw.
        import soundfile as sf
        raw, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        ch0 = raw[:, 0]
        out_wav = out_audio_root / f"{meeting_id}.wav"
        write_wav(out_wav, ch0, sr, subtype="PCM_16")
        duration = len(ch0) / sr

        ingest_records.append({
            "audio_id": meeting_id,
            "audio_path": str(out_wav.relative_to(ROOT)),
            "speaker_id": "multi",  # multi-speaker source
            "language": "zh",
            "domain": "meeting",
            "recording_type": "interview",
            "transcript_path": "",
            "duration_sec": duration,
            "audio_hash": file_hash(out_wav),
        })

        # Parse GT
        intervals = parse_textgrid(tg_path)
        log.info(f"  {len(intervals)} turns from {len(set(i['tier'] for i in intervals))} speakers")
        for it in intervals:
            gt_records.append({
                "meeting_id": meeting_id,
                "speaker": it["tier"],  # e.g. N_SPK8013
                "start": it["start"],
                "end": it["end"],
                "text": it["text"],
            })

    write_jsonl(ROOT / args.out_manifest, ingest_records)
    write_jsonl(ROOT / args.out_gt, gt_records)
    log.info(
        f"wrote {len(ingest_records)} meetings to {args.out_manifest}, "
        f"{len(gt_records)} GT turns to {args.out_gt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

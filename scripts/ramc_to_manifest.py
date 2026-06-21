"""Parse MagicData-RAMC test set GT and emit ProsoGate ingest manifest.

RAMC layout:
  test/wav/CTS-CN-F2F-*.wav     (16kHz mono, ~30min, 2-speaker conversation)
  test/txt/CTS-CN-F2F-*.txt     each line: [start,end]<tab>speaker<tab>gender,dialect<tab>text

Output:
  1. work_test/ramc_ingest.jsonl                — ingest manifest (one row per dialogue)
  2. work_test/ramc_diarization_gt.jsonl        — GT diarization turns

Noise/silence rows (speaker == G00000000) are kept in GT (marked is_noise=true)
but excluded when computing speaker counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prosogate.hash_utils import file_hash
from prosogate.logging_utils import get_logger
from prosogate.manifest import write_jsonl

log = get_logger("ramc_to_manifest")


def parse_ramc_txt(path: Path) -> list[dict]:
    """Parse RAMC .txt; one line per turn: [s,e]<TAB>spk<TAB>gender,dialect<TAB>text."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        time_field = parts[0].strip("[]")
        try:
            start, end = time_field.split(",")
            start = float(start)
            end = float(end)
        except Exception:
            continue
        speaker = parts[1]
        gender_dialect = parts[2]
        text = parts[3]
        out.append({
            "start": start,
            "end": end,
            "speaker": speaker,
            "gender_dialect": gender_dialect,
            "text": text,
            "is_noise": speaker == "G00000000",
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-root",
        default="/workspace/data/magicdata_ramc/extracted/test",
    )
    parser.add_argument(
        "--out-manifest",
        default="work_test/ramc_ingest.jsonl",
    )
    parser.add_argument(
        "--out-gt",
        default="work_test/ramc_diarization_gt.jsonl",
    )
    args = parser.parse_args()

    wav_dir = Path(args.test_root) / "wav"
    txt_dir = Path(args.test_root) / "txt"
    assert wav_dir.exists(), wav_dir
    assert txt_dir.exists(), txt_dir

    import soundfile as sf

    ingest = []
    gt = []
    for wav in sorted(wav_dir.glob("*.wav")):
        stem = wav.stem
        txt = txt_dir / f"{stem}.txt"
        if not txt.exists():
            log.warning(f"no txt for {stem}")
            continue
        info = sf.info(str(wav))
        ingest.append({
            "audio_id": stem,
            "audio_path": str(wav),
            "speaker_id": "multi",  # 2-speaker conversation source
            "language": "zh",
            "domain": "conversation",
            "recording_type": "interview",
            "transcript_path": "",
            "duration_sec": info.duration,
            "audio_hash": file_hash(wav),
        })

        turns = parse_ramc_txt(txt)
        speakers = sorted({t["speaker"] for t in turns if not t["is_noise"]})
        log.info(f"{stem}: {info.duration:.1f}s, {len(turns)} turns, speakers={speakers}")
        for t in turns:
            gt.append({"meeting_id": stem, **t})

    write_jsonl(ROOT / args.out_manifest, ingest)
    write_jsonl(ROOT / args.out_gt, gt)
    log.info(
        f"wrote {len(ingest)} dialogues to {args.out_manifest}, {len(gt)} GT turns to {args.out_gt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

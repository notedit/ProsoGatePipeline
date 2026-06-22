"""Export a single grade (default: A) from train/valid/test manifests into a
self-contained subdirectory with hard-linked wavs / alignments / f0 features.

Usage:
  python scripts/export_grade.py \
      --dataset-root tts_dataset_ramc10 \
      --grade A \
      --out-dir tts_dataset_ramc10/grade_A

Output layout (mirrors the parent dataset):
  <out-dir>/
    manifests/{train,valid,test}.jsonl    # filtered records, paths rewritten to point inside out-dir
    wavs/<utt_id>.wav                     # hardlinked (no extra disk usage on the same FS)
    alignments/<utt_id>.json
    features/<utt_id>_f0.npy
    grade_summary.json                    # count per split, total duration, speakers

The exported manifests have `wav`/`alignment_json_path`/`f0_npy_path` rewritten
relative to <out-dir> so the bundle is portable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _link_or_copy(src: Path, dst: Path) -> str:
    """Try hardlink first (no disk), fall back to copy on cross-device errors."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True, help="Parent dataset dir (contains manifests/, wavs/, ...)")
    p.add_argument("--grade", default="A", help="Grade letter to export (A/B/C)")
    p.add_argument("--out-dir", required=True, help="Where to write the exported subset")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = p.parse_args()

    root = Path(args.dataset_root).resolve()
    out = Path(args.out_dir).resolve()
    grade = args.grade

    in_manifests = root / "manifests"
    out_manifests = out / "manifests"
    out_wavs = out / "wavs"
    out_aligns = out / "alignments"
    out_feats = out / "features"
    for d in (out_manifests, out_wavs, out_aligns, out_feats):
        d.mkdir(parents=True, exist_ok=True)

    summary: dict = {"grade": grade, "splits": {}, "modes": {"link": 0, "copy": 0, "missing": 0}}

    for split in args.splits:
        src_manifest = in_manifests / f"{split}.jsonl"
        if not src_manifest.exists():
            print(f"[skip] {src_manifest} not found")
            continue

        kept = []
        with open(src_manifest, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("grade") != grade:
                    continue

                utt_id = rec["utt_id"]
                new_rec = dict(rec)

                # Hardlink wav
                src_wav = Path(rec["wav"])
                if not src_wav.is_absolute():
                    src_wav = root.parent / src_wav  # tts_dataset_ramc10/wavs/...
                if src_wav.exists():
                    dst_wav = out_wavs / f"{utt_id}.wav"
                    mode = _link_or_copy(src_wav, dst_wav)
                    summary["modes"][mode] += 1
                    new_rec["wav"] = str(dst_wav.relative_to(out.parent)) if dst_wav.is_relative_to(out.parent) else str(dst_wav)
                else:
                    summary["modes"]["missing"] += 1

                # Hardlink alignment JSON
                if rec.get("alignment_json_path"):
                    src_a = Path(rec["alignment_json_path"])
                    if not src_a.is_absolute():
                        src_a = root.parent / src_a
                    if src_a.exists():
                        dst_a = out_aligns / f"{utt_id}.json"
                        _link_or_copy(src_a, dst_a)
                        new_rec["alignment_json_path"] = str(dst_a.relative_to(out.parent)) if dst_a.is_relative_to(out.parent) else str(dst_a)

                # Hardlink f0 npy
                if rec.get("f0_npy_path"):
                    src_f = Path(rec["f0_npy_path"])
                    if not src_f.is_absolute():
                        src_f = root.parent / src_f
                    if src_f.exists():
                        dst_f = out_feats / f"{utt_id}_f0.npy"
                        _link_or_copy(src_f, dst_f)
                        new_rec["f0_npy_path"] = str(dst_f.relative_to(out.parent)) if dst_f.is_relative_to(out.parent) else str(dst_f)

                kept.append(new_rec)

        out_path = out_manifests / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        total_dur = sum(r.get("duration", 0.0) for r in kept)
        spk_counts: dict[str, int] = {}
        for r in kept:
            sl = r.get("speaker_label", "?")
            spk_counts[sl] = spk_counts.get(sl, 0) + 1
        summary["splits"][split] = {
            "n": len(kept),
            "duration_sec": round(total_dur, 1),
            "speaker_labels": spk_counts,
        }
        print(f"[{split}] {len(kept)} records, {total_dur:.1f}s ({total_dur/60:.1f} min)")

    with open(out / "grade_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nsummary: {summary['modes']}")
    print(f"wrote {out}/grade_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

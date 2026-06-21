"""Evaluate stage-04 diarization output against GT using DER + JER.

Reads:
  - Pipeline output: work/04_vad_coarse.jsonl (segment-level, each row has
    {seg_id, source_audio_id, speaker_label, start, end, duration, ...})
  - Ground truth: work_test/{alimeeting,ramc}_diarization_gt.jsonl
    (per-turn {meeting_id, speaker, start, end, text, is_noise?})

Computes per-meeting and overall:
  - DER = (false alarm + missed + speaker confusion) / total speech
  - JER = Jaccard Error Rate (speaker-level)
  - Speaker count error (predicted N speakers vs GT N speakers)
  - DER components breakdown

Usage:
  python scripts/eval_diarization.py \
      --predictions work/04_vad_coarse.jsonl \
      --gt work_test/alimeeting_diarization_gt.jsonl \
      [--collar 0.25] [--skip-overlap]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prosogate.logging_utils import get_logger

log = get_logger("eval_diar")


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Group segments by source_audio_id (== meeting_id)."""
    by_meeting: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Skip rejected short segments
            if rec.get("status") == "rejected":
                continue
            mid = rec.get("source_audio_id") or rec.get("audio_id") or "?"
            by_meeting[mid].append(rec)
    return by_meeting


def load_gt(path: Path, skip_noise: bool = True) -> dict[str, list[dict[str, Any]]]:
    by_meeting: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            if skip_noise and t.get("is_noise"):
                continue
            mid = t["meeting_id"]
            by_meeting[mid].append(t)
    return by_meeting


def to_annotation(turns: list[dict[str, Any]], spk_key: str):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for i, t in enumerate(turns):
        s, e = float(t["start"]), float(t["end"])
        if e <= s:
            continue
        speaker = t.get(spk_key) or t.get("speaker") or t.get("speaker_label") or "UNK"
        ann[Segment(s, e), i] = speaker
    return ann


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="work/04_vad_coarse.jsonl",
        help="Stage 04 output JSONL (relative to project root)",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="GT diarization JSONL (e.g. work_test/alimeeting_diarization_gt.jsonl)",
    )
    parser.add_argument(
        "--collar",
        type=float,
        default=0.25,
        help="Collar in seconds around each turn boundary to ignore (DIHARD uses 0.25)",
    )
    parser.add_argument(
        "--skip-overlap",
        action="store_true",
        help="Exclude overlapping speech from scoring (lenient)",
    )
    parser.add_argument(
        "--out-report",
        default="work_test/diarization_eval_report.json",
    )
    args = parser.parse_args()

    pred_path = ROOT / args.predictions
    gt_path = ROOT / args.gt
    assert pred_path.exists(), f"missing {pred_path}"
    assert gt_path.exists(), f"missing {gt_path}"

    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    der_metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    jer_metric = JaccardErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)

    preds = load_predictions(pred_path)
    gts = load_gt(gt_path)

    common_ids = sorted(set(preds.keys()) & set(gts.keys()))
    log.info(
        f"pred meetings: {len(preds)}, gt meetings: {len(gts)}, common: {len(common_ids)}"
    )
    if not common_ids:
        # Try suffix matching: pipeline often prefixes audio_id with speaker_id like "multi_"
        id_map: dict[str, str] = {}
        for p in preds:
            for g in gts:
                if p == g or p.endswith("_" + g) or p.endswith(g):
                    id_map[p] = g
                    break
        if id_map:
            log.info(f"applying suffix match: {len(id_map)} mappings")
            for p, g in id_map.items():
                preds[g] = preds.pop(p)
            common_ids = sorted(set(preds.keys()) & set(gts.keys()))

    if not common_ids:
        log.error(
            "no overlapping meeting ids — check that pipeline ran on the SAME audio files"
        )
        log.info(f"  pred sample: {list(preds.keys())[:3]}")
        log.info(f"  gt sample: {list(gts.keys())[:3]}")
        return 1

    per_meeting = []
    for mid in common_ids:
        ref = to_annotation(gts[mid], spk_key="speaker")
        hyp = to_annotation(preds[mid], spk_key="speaker_label")
        if len(ref) == 0 or len(hyp) == 0:
            log.warning(f"{mid}: empty ref/hyp, skipping")
            continue
        # Get component breakdown via detailed=True
        components = der_metric(ref, hyp, detailed=True)
        jer_value = jer_metric(ref, hyp)
        # Speaker count
        ref_spks = set(ref.labels())
        hyp_spks = set(hyp.labels())
        per_meeting.append({
            "meeting_id": mid,
            "duration_total": components.get("total", 0.0),
            "der": components.get("diarization error rate", 0.0),
            "false_alarm": components.get("false alarm", 0.0),
            "missed": components.get("missed detection", 0.0),
            "confusion": components.get("confusion", 0.0),
            "jer": jer_value,
            "n_speakers_ref": len(ref_spks),
            "n_speakers_hyp": len(hyp_spks),
            "speaker_count_err": len(hyp_spks) - len(ref_spks),
        })
        log.info(
            f"{mid}: DER={components.get('diarization error rate',0):.3f} "
            f"JER={jer_value:.3f} "
            f"ref_spk={len(ref_spks)} hyp_spk={len(hyp_spks)}"
        )

    if not per_meeting:
        log.error("no scorable meetings")
        return 1

    # Overall metrics by accumulating
    overall_der = abs(der_metric)
    overall_jer = abs(jer_metric)
    overall = {
        "n_meetings": len(per_meeting),
        "der": overall_der,
        "jer": overall_jer,
        "der_components_aggregate": {
            "false_alarm": sum(m["false_alarm"] for m in per_meeting),
            "missed": sum(m["missed"] for m in per_meeting),
            "confusion": sum(m["confusion"] for m in per_meeting),
            "total": sum(m["duration_total"] for m in per_meeting),
        },
        "mean_speaker_count_err": sum(abs(m["speaker_count_err"]) for m in per_meeting) / len(per_meeting),
    }

    print("\n=== overall ===")
    print(f"  meetings:               {overall['n_meetings']}")
    print(f"  DER:                    {overall['der']*100:.2f}%")
    print(f"  JER:                    {overall['jer']*100:.2f}%")
    agg = overall["der_components_aggregate"]
    if agg["total"] > 0:
        print(f"  false_alarm / total:    {agg['false_alarm']/agg['total']*100:.2f}%")
        print(f"  missed / total:         {agg['missed']/agg['total']*100:.2f}%")
        print(f"  confusion / total:      {agg['confusion']/agg['total']*100:.2f}%")
    print(f"  mean |#spk_hyp - #spk_ref|: {overall['mean_speaker_count_err']:.2f}")
    print(f"\n  collar:    {args.collar}s")
    print(f"  skip_overlap: {args.skip_overlap}")

    out_path = ROOT / args.out_report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "overall": overall,
        "per_meeting": per_meeting,
        "config": {"collar": args.collar, "skip_overlap": args.skip_overlap},
    }, indent=2, ensure_ascii=False))
    print(f"\nreport saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

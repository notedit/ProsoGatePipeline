"""Pipeline orchestrator — runs all 14 stages in order.

Usage:
  python scripts/run_pipeline.py --config configs/pipeline.yaml [--from N] [--to N]
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prosogate.config import load_config
from prosogate.logging_utils import get_logger

STAGES = [
    ("01_ingest", "Ingest"),
    ("02_audio_qc", "Audio QC"),
    ("03_resample", "Resample"),
    ("04_vad_coarse", "VAD + Diarization"),
    ("05_asr_qwen3", "Qwen3-ASR"),
    ("06_text_normalize", "Text Normalize"),
    ("07_align_qwen3", "Qwen3-ForcedAligner"),
    ("08_fine_segment", "Fine Segment"),
    ("09_spk_consistency", "Speaker Consistency"),
    ("10_extract_f0", "F0 Extract"),
    ("11_rate_metrics", "Rate Metrics"),
    ("12_filter_score", "Filter + Score"),
    ("13_split_dataset", "Split Dataset"),
    ("14_report", "Report"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--from", dest="from_stage", type=int, default=1)
    parser.add_argument("--to", dest="to_stage", type=int, default=len(STAGES))
    args = parser.parse_args()

    log = get_logger("pipeline")
    cfg = load_config(args.config)

    for i, (mod_name, title) in enumerate(STAGES, start=1):
        if i < args.from_stage or i > args.to_stage:
            continue
        log.info(f"[{i}/14] {title} -> {mod_name}.py")
        t0 = time.time()
        try:
            mod = importlib.import_module(f"stages.{mod_name}")
            rc = mod.run(cfg)
        except Exception as e:
            log.exception(f"stage {mod_name} failed: {e}")
            return 1
        log.info(f"[{i}/14] {title} done in {time.time() - t0:.1f}s (rc={rc})")
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())

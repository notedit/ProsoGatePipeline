"""Download a 50-hour subset of WenetSpeech4TTS Premium for ProsoGate validation.

WenetSpeech4TTS is gated on HuggingFace. Before running this script:
  1. Visit https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS
  2. Accept the license (Terms of Access form)
  3. Confirm HUGGINGFACE_TOKEN env var is set with read scope

The Premium subset is split into ~14 tar.gz files (~50-100h each).
We download Premium_0.tar.gz only (smallest hop to validate the pipeline), then
sample utterances to total ~50 hours of long audio.

NOTE: WenetSpeech4TTS stores **already-segmented utterances** (1-30s each), not
raw long-form audio. For ProsoGate §1-§8 validation we **concatenate them per
podcast/audiobook source** into synthetic long audios in a separate step (see
`scripts/wenetspeech_build_long_audios.py`).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/workspace/data/wenetspeech4tts",
        help="Where to put the downloaded tar.gz and filelist",
    )
    parser.add_argument(
        "--subset",
        choices=["Premium", "Standard", "Basic", "Rest"],
        default="Premium",
    )
    parser.add_argument(
        "--shards",
        type=int,
        nargs="+",
        default=[0],
        help="Which shard indices to download (e.g. 0 1 2)",
    )
    parser.add_argument(
        "--filelist-only",
        action="store_true",
        help="Only download the .lst filelist, no tar.gz (used to plan sampling)",
    )
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("FAIL: HUGGINGFACE_TOKEN / HF_TOKEN env var not set")
        return 1

    from huggingface_hub import hf_hub_download

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. filelist always
    print(f"[1/2] downloading filelist for {args.subset}")
    lst = hf_hub_download(
        repo_id="Wenetspeech4TTS/WenetSpeech4TTS",
        filename=f"filelists/{args.subset}_filelist.lst",
        repo_type="dataset",
        local_dir=str(out_root),
        token=token,
    )
    print(f"  -> {lst}")

    if args.filelist_only:
        return 0

    # 2. selected shards
    print(f"[2/2] downloading shards {args.shards}")
    for s in args.shards:
        name = f"{args.subset}/WenetSpeech4TTS_{args.subset}_{s}.tar.gz"
        print(f"  -> {name}")
        p = hf_hub_download(
            repo_id="Wenetspeech4TTS/WenetSpeech4TTS",
            filename=name,
            repo_type="dataset",
            local_dir=str(out_root),
            token=token,
        )
        print(f"     {p} ({os.path.getsize(p) / 1e9:.2f} GB)")

    md5 = hf_hub_download(
        repo_id="Wenetspeech4TTS/WenetSpeech4TTS",
        filename=f"{args.subset}/{args.subset}_md5check.txt",
        repo_type="dataset",
        local_dir=str(out_root),
        token=token,
    )
    print(f"md5 list: {md5}")

    print("\nDone. Next steps:")
    print(f"  cd {out_root}")
    print(f"  for f in {args.subset}/*.tar.gz; do echo $f; tar -tzf $f | head; done   # peek")
    print(f"  for f in {args.subset}/*.tar.gz; do tar -xzf $f -C extracted/; done    # extract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

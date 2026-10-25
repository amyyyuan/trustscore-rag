#!/usr/bin/env python3
"""Step 0: prepare both retrieval settings under data/.

Needs the upstream repositories under third_party/ (setup_mac.sh clones them) and, for nq_beir,
downloads the BEIR NQ corpus once (about a few hundred MB zipped; the zip is deleted afterwards and
only the needed passages are kept).

Usage: python scripts/00_prepare_data.py --config configs/default.yaml [--keep-zip]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trustscore.config import load_config, resolve  # noqa: E402
from trustscore.data import prepare_nq_beir, prepare_nq_google  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides a.b=c")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    third_party, data_dir = resolve(cfg, "third_party"), resolve(cfg, "data")
    if cfg["datasets"]["nq_google"]["enabled"]:
        prepare_nq_google(cfg, third_party, data_dir)
    if cfg["datasets"]["nq_beir"]["enabled"]:
        prepare_nq_beir(cfg, third_party, data_dir)
        zip_path = data_dir / "downloads" / "nq.zip"
        if zip_path.exists() and not args.keep_zip:
            zip_path.unlink()
            print("deleted the corpus zip (use --keep-zip to keep it)")
    print("done")


if __name__ == "__main__":
    main()

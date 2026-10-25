#!/usr/bin/env python3
"""Step 5: figures for the report from the tables written by steps 3 and 4.

Usage: python scripts/05_figures.py --config configs/default.yaml
Output: results/<setting>/figures/*.pdf (+ .png previews)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trustscore.config import load_config, resolve  # noqa: E402
from trustscore.plots import make_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--settings", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    results = resolve(cfg, "results")
    names = args.settings or [n for n, d in cfg["datasets"].items() if d["enabled"]]
    for name in names:
        if not (results / name / "tables").exists():
            print(f"[{name}] no tables yet, skipping")
            continue
        made = make_all(results / name, results / name / "figures")
        print(f"[{name}] figures: {made}")


if __name__ == "__main__":
    main()

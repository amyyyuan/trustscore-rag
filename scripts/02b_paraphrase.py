#!/usr/bin/env python3
"""Optional step (needs an LLM backend): write LLM paraphrases of the attack passages so that
01_build_sets.py can add the adaptive conditions fact_para_m* and inj_para_m*.

Usage: python scripts/02b_paraphrase.py --config configs/default.yaml
Output: data/<setting>/paraphrases.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trustscore.config import load_config, resolve  # noqa: E402
from trustscore.data import Setting  # noqa: E402
from trustscore.llm import LLMClient  # noqa: E402
from trustscore.paraphrase import generate_paraphrases  # noqa: E402
from trustscore.utils import write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--settings", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    data_dir, results = resolve(cfg, "data"), resolve(cfg, "results")
    model = cfg["attacks"]["paraphrase"].get("model") or cfg["llm"]["model"]
    llm = LLMClient(cfg["llm"], results / "llm_cache", model=model)
    names = args.settings or cfg["llm"].get("settings") or [n for n, d in cfg["datasets"].items() if d["enabled"]]
    for name in names:
        setting = Setting(name, data_dir)
        stats = generate_paraphrases(setting, cfg, llm, m_max=max(cfg["attacks"]["m_values"]))
        stats["model"] = model
        write_json(stats, results / name / "paraphrase_stats.json")
        print(f"[{name}] paraphrase stats: {stats} (LLM calls: {llm.n_calls}, cached: {llm.n_cached})")


if __name__ == "__main__":
    main()

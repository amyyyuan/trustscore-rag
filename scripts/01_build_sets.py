#!/usr/bin/env python3
"""Step 1: build retrieved sets for every condition (clean, fact poisoning, instruction injection,
random passages, and the paraphrased variants when data/<setting>/paraphrases.json exists).

Usage: python scripts/01_build_sets.py --config configs/default.yaml
Output: results/<setting>/sets.jsonl and results/<setting>/sets_summary.json
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trustscore.attacks import Passage  # noqa: E402
from trustscore.config import load_config, resolve  # noqa: E402
from trustscore.data import Setting  # noqa: E402
from trustscore.retrieval import Embedder, score  # noqa: E402
from trustscore.sets import build_sets  # noqa: E402
from trustscore.utils import read_jsonl, set_seed, write_json, write_jsonl  # noqa: E402


def add_random_sets(setting: Setting, cfg: dict, embedder: Embedder, sets_path: Path, n_sets: int = 100) -> int:
    """Cuconasu-style irrelevant-but-benign sets: k random corpus passages for never-attacked queries."""
    if not setting.random_passages:
        return 0
    k = cfg["retrieval"]["k"]
    rng = random.Random(cfg["seed"])
    rows = read_jsonl(sets_path)
    extra = setting.extra_ids[:n_sets]
    q_vecs = embedder.embed([setting.queries[q]["question"] for q in extra])
    pool = setting.random_passages
    new_rows = []
    for qi, qid in enumerate(extra):
        picks = rng.sample(pool, k)
        pairs = [setting.random_texts(d) for d in picks]
        texts = [t for t, _ in pairs]
        vecs = embedder.embed([r for _, r in pairs])
        passages = []
        for d, t, v in zip(picks, texts, vecs):
            passages.append(Passage(doc_id=f"rand|{d['_id']}", text=t,
                                    score=float(score(q_vecs[qi], v[None, :], cfg["retrieval"]["score"])[0])))
        passages.sort(key=lambda p: -p.score)
        for r, p in enumerate(passages, start=1):
            p.rank = r
        q = setting.queries[qid]
        new_rows.append({
            "set_id": f"{setting.name}|random|{qid}", "setting": setting.name, "condition": "random",
            "qid": qid, "question": q["question"], "is_target": False, "family": None, "m": 0, "k": k,
            "correct_answer": None, "incorrect_answer": None, "n_poison_in_topk": 0, "n_poison_total": 0,
            "passages": [p.to_dict() for p in passages],
        })
    write_jsonl(rows + new_rows, sets_path)
    return len(new_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--settings", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    data_dir, results = resolve(cfg, "data"), resolve(cfg, "results")
    names = args.settings or [n for n, d in cfg["datasets"].items() if d["enabled"]]
    embedder = Embedder(cfg["retrieval"]["model"], cfg["signals"].get("device", "auto"),
                        cfg["retrieval"]["max_length"], cfg["signals"].get("batch_size", 16),
                        results / "cache" / "emb_retrieval.pkl")
    for name in names:
        setting = Setting(name, data_dir, cfg["datasets"][name].get("include_title"))
        out = results / name / "sets.jsonl"
        summary = build_sets(setting, cfg, embedder, out)
        summary["n_random_sets"] = add_random_sets(setting, cfg, embedder, out)
        write_json(summary, results / name / "sets_summary.json")
        print(f"[{name}] wrote {summary['n_sets'] + summary['n_random_sets']} sets to {out}")


if __name__ == "__main__":
    main()

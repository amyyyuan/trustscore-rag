#!/usr/bin/env python3
"""Step 4 (needs an LLM backend): end-task evaluation of the defences.

For each selected condition and defence, the generator answers every target question and we record
attack success, accuracy and abstention. The known-answer detector's per-passage flags are also
saved and scored as a detector.

Usage: python scripts/04_llm_endtask.py --config configs/default.yaml [--max-questions N]
Output: results/<setting>/endtask/<tag>/responses.jsonl, results/<setting>/tables/endtask_<tag>.*,
        results/<setting>/tables/baseline_known_answer_<tag>.*
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trustscore.baselines import KeywordExtractor, known_answer_flags  # noqa: E402
from trustscore.config import load_config, resolve  # noqa: E402
from trustscore.endtask import judge, run_defence, summarise_endtask  # noqa: E402
from trustscore.llm import LLMClient  # noqa: E402
from trustscore.utils import read_jsonl, set_seed, write_json, write_jsonl  # noqa: E402


def load_flags(scores_csv: Path, fpr: float) -> dict:
    df = pd.read_csv(scores_csv)
    out: dict[str, dict] = {}
    for set_id, g in df.groupby("set_id"):
        out[set_id] = {"flag": {d: bool(s >= t) for d, s, t in zip(g["doc_id"], g["score"], g[f"thr_{fpr}"])}}
    return out


def best_single_signal(results: Path, setting: str, fpr: float, cfg: dict) -> tuple[str, str]:
    """The single signal with the highest mean TPR at `fpr` across conditions. To keep the choice
    out of the evaluation data it is made on the other retrieval setting when that has been run
    (its questions and corpus are disjoint); otherwise on the same setting, which is reported."""
    others = [n for n, d in cfg["datasets"].items() if d["enabled"] and n != setting
              and (results / n / "tables" / "single_signals.csv").exists()]
    chosen_on = others[0] if others else setting
    single = pd.read_csv(results / chosen_on / "tables" / "single_signals.csv")
    single = single[single["signal"] != cfg["evaluation"].get("length_baseline", "n_tokens")]
    means = single.groupby("signal")[f"tpr@{fpr}"].mean().sort_values(ascending=False)
    return str(means.index[0]), chosen_on


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--settings", nargs="*", default=None)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--filter-fpr", type=float, default=0.05)
    ap.add_argument("--defenses", nargs="*", default=None, help="override llm.defenses")
    ap.add_argument("--conditions", nargs="*", default=None, help="override llm.endtask_conditions")
    ap.add_argument("--tag", default=None, help="output tag (default: the model name)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    results = resolve(cfg, "results")
    llm_cfg = cfg["llm"]
    llm = LLMClient(llm_cfg, results / "llm_cache")
    fake = llm_cfg["backend"] == "fake"
    extractor = KeywordExtractor()
    defenses = args.defenses or llm_cfg["defenses"]
    tag = args.tag or re.sub(r"[^A-Za-z0-9_.-]", "_", llm.model)
    names = args.settings or llm_cfg.get("settings") or [n for n, d in cfg["datasets"].items() if d["enabled"]]
    for name in names:
        out_dir = results / name
        sets = read_jsonl(out_dir / "sets.jsonl")
        wanted = list(args.conditions or llm_cfg["endtask_conditions"])
        present = {s["condition"] for s in sets}
        missing = [c for c in wanted if c not in present]
        if missing:
            print(f"[{name}] conditions not built (skipped): {missing}")
        wanted = [c for c in wanted if c in present]
        target_sets = [s for s in sets if s["is_target"] and s["condition"] in wanted]
        if args.max_questions:
            keep_q = sorted({s["qid"] for s in target_sets})[: args.max_questions]
            target_sets = [s for s in target_sets if s["qid"] in keep_q]
        scores = {}
        best, chosen_on = None, None
        if any(d.startswith("filter_") for d in defenses):
            scores["filter_fused"] = load_flags(out_dir / "scores_fused_all.csv", args.filter_fpr)
            scores["filter_fused_no_overlap"] = load_flags(out_dir / "scores_fused_no_overlap.csv", args.filter_fpr)
            best, chosen_on = best_single_signal(results, name, args.filter_fpr, cfg)
            scores["filter_best_single"] = load_flags(out_dir / f"scores_single_{best}.csv", args.filter_fpr)
            print(f"[{name}] best single signal at FPR {args.filter_fpr}: {best} (chosen on {chosen_on})")
        rows, ka_rows = [], []
        for i, s in enumerate(target_sets):
            for defence in defenses:
                res = run_defence(llm, defence, s, cfg, scores.get(defence), extractor, fake=fake)
                verdict = judge(res["response"], s["correct_answer"], s["incorrect_answer"] if s["condition"] != "clean" else None)
                rows.append({"setting": name, "condition": s["condition"], "qid": s["qid"], "defence": defence,
                             "n_poison_in_topk": s["n_poison_in_topk"], **res, **verdict})
            if (i + 1) % 20 == 0:
                print(f"[{name}] {i + 1}/{len(target_sets)} sets done (LLM calls {llm.n_calls}, cached {llm.n_cached})", flush=True)
                write_jsonl(rows, out_dir / "endtask" / tag / "responses.jsonl")
        write_jsonl(rows, out_dir / "endtask" / tag / "responses.jsonl")
        summary = pd.DataFrame(summarise_endtask(rows, cfg["evaluation"]["n_bootstrap"], cfg["seed"]))
        (out_dir / "tables").mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "tables" / f"endtask_{tag}.csv", index=False, float_format="%.3f")
        with open(out_dir / "tables" / f"endtask_{tag}.md", "w") as f:
            f.write(summary.to_markdown(index=False, floatfmt=".3f") + "\n")
        write_json({"best_single_signal": best, "best_single_chosen_on": chosen_on if best else None,
                    "filter_fpr": args.filter_fpr, "model": llm.model, "tag": tag,
                    "defenses": defenses, "conditions": wanted, "llm_calls": llm.n_calls, "cached": llm.n_cached},
                   out_dir / "endtask" / tag / "meta.json")

        # known-answer detection scored as a detector (subset of conditions, to bound the number of calls)
        if "known_answer" in defenses:
            ka_conds = set(llm_cfg["known_answer"].get("conditions", wanted))
            for s in [s for s in target_sets if s["condition"] in ka_conds]:
                texts = [p["text"] for p in s["passages"]]
                flags = known_answer_flags(llm, texts, llm_cfg["known_answer"]["key_length"],
                                           fake_hint_fn=(lambda i, s=s: {"hijack": s["passages"][i]["is_poison"]}) if fake else None)
                for p, f in zip(s["passages"], flags):
                    ka_rows.append({"condition": s["condition"], "qid": s["qid"], "is_poison": p["is_poison"], "flag": f})
            ka = pd.DataFrame(ka_rows)
            tab = []
            for cond, g in ka.groupby("condition"):
                pos, neg = g[g["is_poison"]], g[~g["is_poison"]]
                tab.append({"condition": cond, "known_answer_tpr": float(pos["flag"].mean()) if len(pos) else float("nan"),
                            "known_answer_fpr": float(neg["flag"].mean()) if len(neg) else float("nan"),
                            "n_pos": len(pos), "n_neg": len(neg)})
            pd.DataFrame(tab).to_csv(out_dir / "tables" / f"baseline_known_answer_{tag}.csv", index=False, float_format="%.3f")
        print(summary.to_string())


if __name__ == "__main__":
    main()

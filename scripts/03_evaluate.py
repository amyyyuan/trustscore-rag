#!/usr/bin/env python3
"""Step 3: single-signal audit, fused scorer, ablations, cross-family transfer, leave-one-condition-
out generalisation, baselines, and false-positive checks on other clean populations.

Usage: python scripts/03_evaluate.py --config configs/default.yaml
Output: results/<setting>/tables/*.csv and *.md, results/<setting>/scores_<experiment>.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trustscore.config import load_config, resolve  # noqa: E402
from trustscore.fusion import cross_validated_scores, raw_auroc_table, summarise  # noqa: E402
from trustscore.signals import ALL_SIGNALS, PER_SET, PER_TEXT  # noqa: E402
from trustscore.utils import read_jsonl, set_seed, write_json  # noqa: E402


def recipe(condition: str) -> str:
    """Attack recipe of a condition: the name without the m count and the repeat count, so that
    fact_prefix_m1 and fact_prefix_m5 share a recipe, as do inj_standalone_m1 and inj_standalone_rep10_m1."""
    return re.sub(r"_rep\d+", "", re.sub(r"_m\d+$", "", condition))


def save(df: pd.DataFrame, path: Path, floatfmt: str = "%.3f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False, float_format=floatfmt)
    with open(path.with_suffix(".md"), "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=".3f") + "\n")


def run_experiment(df, features, train_conditions, cfg, name, out_dir, exclude=None, eval_conditions=None):
    ev = cfg["evaluation"]
    scored, coefs = cross_validated_scores(df, features, train_conditions, ev["fpr_targets"], ev["n_folds"],
                                           ev["fusion_C"], cfg["seed"], exclude_from_training=exclude,
                                           balance_conditions=ev.get("balance_conditions", True))
    table = summarise(scored, ev["fpr_targets"], ev["n_bootstrap"], cfg["seed"], conditions=eval_conditions)
    table.insert(0, "experiment", name)
    save(table, out_dir / "tables" / f"{name}")
    scored[["set_id", "doc_id", "qid", "condition", "is_poison", "fold", "score"]
           + [f"thr_{f}" for f in ev["fpr_targets"]]].to_csv(out_dir / f"scores_{name}.csv", index=False)
    return scored, table, coefs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--settings", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    results = resolve(cfg, "results")
    ev = cfg["evaluation"]
    names = args.settings or [n for n, d in cfg["datasets"].items() if d["enabled"]]
    all_signals = [f for f in ev["features_all"] if f in ALL_SIGNALS]

    for name in names:
        out_dir = results / name
        df = pd.read_csv(out_dir / "signals.csv")
        df["is_poison"] = df["is_poison"].astype(bool)
        df["is_target"] = df["is_target"].astype(bool)
        main_df = df[df["condition"] != "random"].copy()
        random_df = df[df["condition"] == "random"].copy()
        attacked = sorted(c for c in main_df["condition"].unique() if c != "clean")
        # Conditions matching the test-only pattern (the paraphrased adaptive variants by default) are
        # scored out of fold like every other condition but never used to fit a scorer, so that they
        # test attacks the scorer has not seen. Every experiment below trains on `train_pool` only.
        test_only_pat = ev.get("test_only_pattern", "_para_")
        test_only = [c for c in attacked if test_only_pat and test_only_pat in c]
        train_pool = [c for c in attacked if c not in test_only]
        fact_conds = [c for c in train_pool if c.startswith("fact")]
        inj_conds = [c for c in train_pool if c.startswith("inj")]
        print(f"[{name}] {len(main_df)} passages, conditions: {attacked}; test-only (never trained on): {test_only}")

        # 0. raw AUROC of each signal in its a-priori direction
        save(raw_auroc_table(main_df, all_signals), out_dir / "tables" / "raw_auroc")

        # 1. single-signal audit (one-feature logistic regression, grouped CV); the length-only
        #    reference (n_tokens) is audited the same way but never fused
        single_tables = []
        length_baseline = ev.get("length_baseline")
        audited = all_signals + ([length_baseline] if length_baseline and length_baseline in main_df.columns else [])
        for f in audited:
            _, t, _ = run_experiment(main_df, [f], train_pool, cfg, f"single_{f}", out_dir)
            t.insert(1, "signal", f)
            single_tables.append(t)
        single = pd.concat(single_tables, ignore_index=True)
        save(single, out_dir / "tables" / "single_signals")
        pivot_cols = {"auroc": "auroc"}
        pivot_cols.update({f"tpr@{fpr}": f"tpr_at_{int(round(fpr * 100))}pct" for fpr in ev["fpr_targets"]})
        for col, tag in pivot_cols.items():
            save(single.pivot(index="condition", columns="signal", values=col).reset_index(),
                 out_dir / "tables" / f"single_pivot_{tag}")

        # 2. fused scorers and feature ablations (trained on all attacked conditions)
        fused_specs = {
            "fused_all": all_signals,
            "fused_no_overlap": [f for f in ev["features_no_overlap"] if f in ALL_SIGNALS],
            "fused_no_queryshape": [f for f in ev.get("features_no_queryshape", ev["features_no_overlap"]) if f in ALL_SIGNALS],
            "fused_per_text": [f for f in PER_TEXT if f in all_signals],
            "fused_per_set": [f for f in PER_SET if f in all_signals],
        }
        fused_tables, coef_tables = [], {}
        for exp, feats in fused_specs.items():
            _, t, coefs = run_experiment(main_df, feats, train_pool, cfg, exp, out_dir)
            fused_tables.append(t)
            coef_tables[exp] = coefs
        save(pd.concat(fused_tables, ignore_index=True), out_dir / "tables" / "fused")
        coef_summary = coef_tables["fused_all"].drop(columns=["fold"]).agg(["mean", "std"]).T.reset_index()
        coef_summary.columns = ["signal", "coef_mean", "coef_std"]
        save(coef_summary.sort_values("coef_mean", key=np.abs, ascending=False), out_dir / "tables" / "coefficients")

        # 3. cross-family transfer: train on one family, test on everything
        transfer = []
        for train_name, train_conds in (("train_fact", fact_conds), ("train_inj", inj_conds), ("train_both", train_pool)):
            if not train_conds:
                continue
            _, t, _ = run_experiment(main_df, all_signals, train_conds, cfg, f"transfer_{train_name}", out_dir)
            t.insert(1, "trained_on", train_name)
            transfer.append(t)
        save(pd.concat(transfer, ignore_index=True), out_dir / "tables" / "transfer")

        # 4. leave-one-recipe-out: can the scorer catch an attack recipe it never saw? Every
        #    condition of the held-out recipe (all m and repeat counts) is removed from training,
        #    otherwise the same recipe would still be seen at another m.
        loro = []
        for rec in sorted({recipe(c) for c in train_pool}):
            held = [c for c in train_pool if recipe(c) == rec]
            _, t, _ = run_experiment(main_df, all_signals, train_pool, cfg, f"loro_{rec}", out_dir, exclude=held,
                                     eval_conditions=held)
            t.insert(1, "held_out_recipe", rec)
            loro.append(t)
        save(pd.concat(loro, ignore_index=True), out_dir / "tables" / "leave_one_recipe_out")

        # 5. TrustRAG stage-1 baseline (hard flags) and the cue-count heuristic
        rows = []
        neg = main_df[(main_df["condition"] == "clean") & (~main_df["is_poison"])]
        for cond in attacked:
            pos = main_df[(main_df["condition"] == cond) & (main_df["is_poison"])]
            rows.append({"condition": cond, "trustrag_tpr": float(pos["trustrag_flag"].mean()),
                         "trustrag_fpr_clean": float(neg["trustrag_flag"].mean()),
                         "trustrag_fpr_collateral": float(main_df[(main_df["condition"] == cond) & (~main_df["is_poison"])]["trustrag_flag"].mean())})
        save(pd.DataFrame(rows), out_dir / "tables" / "baseline_trustrag")

        # 6. false positives on other clean populations with the fused thresholds
        fused_scores = pd.read_csv(out_dir / "scores_fused_all.csv")
        thr = {fpr: float(np.nanmedian(fused_scores[f"thr_{fpr}"])) for fpr in ev["fpr_targets"]}
        checks = {"thresholds_median_over_folds": thr}
        if len(random_df):
            # score the random sets with the fold model of their question (they are never in training)
            scored_all, _ = cross_validated_scores(pd.concat([main_df, random_df]), all_signals, train_pool,
                                                   ev["fpr_targets"], ev["n_folds"], ev["fusion_C"], cfg["seed"])
            rnd = scored_all[scored_all["condition"] == "random"]
            checks["random_passages"] = {f"fpr@{fpr}": float(np.mean(rnd["score"] >= rnd[f"thr_{fpr}"]))
                                         for fpr in ev["fpr_targets"]}
            checks["random_passages"]["n"] = int(len(rnd))
        write_json(checks, out_dir / "tables" / "fpr_checks.json")

        # 7. retrieval success per condition (how many poison passages reached the top-k)
        ret = (main_df[main_df["condition"] != "clean"].groupby("condition")
               .agg(sets=("set_id", "nunique"), poison_in_topk=("is_poison", "sum"), m=("m", "first")).reset_index())
        ret["retrieved_frac"] = ret["poison_in_topk"] / (ret["sets"] * ret["m"])
        save(ret, out_dir / "tables" / "retrieval_success")

        # 8. instruction templates: retrieval rate of every planted passage (from the planted log,
        #    which includes passages that never reached the top-k) and detection rate of the retrieved
        #    ones under the fused scorer, per template and condition
        log_path = out_dir / "planted_log.jsonl"
        if log_path.exists():
            log = pd.DataFrame(read_jsonl(log_path))
            log = log[log["template_id"].notna()]
            if len(log):
                fused_scores = pd.read_csv(out_dir / "scores_fused_all.csv")
                det = main_df[["set_id", "doc_id", "template_id"]].merge(
                    fused_scores[["set_id", "doc_id", "is_poison", "score"] + [f"thr_{f}" for f in ev["fpr_targets"]]],
                    on=["set_id", "doc_id"])
                det = det[det["is_poison"].astype(bool) & det["template_id"].notna()].copy()
                for fpr in ev["fpr_targets"]:
                    det[f"hit_{fpr}"] = (det["score"] >= det[f"thr_{fpr}"]).astype(float)
                det["condition"] = det["set_id"].str.split("|").str[1]
                hits = det.groupby(["condition", "template_id"])[[f"hit_{f}" for f in ev["fpr_targets"]]].mean()
                hits.columns = [f"tpr@{f}" for f in ev["fpr_targets"]]
                retr = log.groupby(["condition", "template_id"]).agg(planted=("doc_id", "size"), retrieved=("retrieved", "mean"))
                save(retr.join(hits, how="left").reset_index(), out_dir / "tables" / "templates")

        # 9. realised false-positive rate of the fused scorer, once (it is the same for every condition)
        fused_tab = pd.read_csv(out_dir / "tables" / "fused.csv")
        fpr_rows = fused_tab[fused_tab["experiment"] == "fused_all"].iloc[:1][
            [f"fpr@{f}_realised" for f in ev["fpr_targets"]]]
        save(fpr_rows, out_dir / "tables" / "fused_realised_fpr")
        print(f"[{name}] tables written to {out_dir / 'tables'}")


if __name__ == "__main__":
    main()

"""Report figures (IEEE column width, vector PDF plus PNG preview).

Palette: a validated colour-blind-safe categorical set in fixed slot order; sequential heatmaps use
one hue; coefficient signs use a two-hue diverging pair around neutral grey.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COL_W = 3.5      # IEEE single column, inches
FULL_W = 7.16    # IEEE double column

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e4e0", "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

PRETTY = {
    "log_ppl": "perplexity", "log_ppl_win": "windowed ppl", "inj_prob": "injection clf", "cue_count": "cue words",
    "sim_z": "query-sim z", "q_overlap": "query overlap", "q_lcs": "query copy", "nn_dist": "nearest dist",
    "mean_dist": "mean dist", "clus_tight": "cluster tightness", "nli_contra_max": "NLI contra (max)",
    "nli_contra_mean": "NLI contra (mean)", "nli_entail_mean": "NLI entail (mean)", "n_tokens": "length only (ref.)",
    "fused_all": "fused (all)", "fused_no_overlap": "fused (no query copy)", "fused_no_queryshape": "fused (no query shape)", "fused_per_text": "fused (per-text)",
    "fused_per_set": "fused (per-set)", "none": "no defence", "filter_fused": "trust filter (fused)",
    "filter_fused_no_overlap": "trust filter (no query copy)",
    "filter_best_single": "trust filter (best single)", "robustrag_keyword": "RobustRAG keyword",
    "known_answer": "known-answer filter",
}

# Canonical order of attack conditions in figures: fact poisoning first, then instruction injection.
CONDITION_ORDER = ["clean",
                   "fact_prefix_m1", "fact_prefix_m5", "fact_noprefix_m1", "fact_noprefix_m5",
                   "fact_para_m1", "fact_para_m5",
                   "inj_carrier_m1", "inj_carrier_m5", "inj_standalone_m1", "inj_standalone_m5",
                   "inj_standalone_rep10_m1", "inj_para_m1", "inj_para_m5"]


def _order(conds) -> list[str]:
    key = {c: i for i, c in enumerate(CONDITION_ORDER)}
    return sorted(conds, key=lambda c: (key.get(c, 99), c))


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def _cond_label(c: str) -> str:
    return (c.replace("fact_prefix", "fact+prefix").replace("fact_noprefix", "fact, no prefix")
            .replace("inj_carrier", "inj in passage").replace("inj_standalone_rep", "inj alone x")
            .replace("inj_standalone", "inj alone")
            .replace("fact_para", "fact paraphrased").replace("inj_para", "inj paraphrased").replace("_m", ", m="))


def heatmap_single(tables: Path, out: Path, metric: str = "tpr_at_5pct") -> None:
    p = tables / f"single_pivot_{metric}.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).set_index("condition")
    cols = [c for c in PRETTY if c in df.columns]
    df = df.loc[_order(df.index), cols]
    fig, ax = plt.subplots(figsize=(FULL_W, 0.28 * len(df) + 1.0))
    im = ax.imshow(df.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [PRETTY[c] for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(len(df)), [_cond_label(c) for c in df.index])
    ax.grid(False)
    for i in range(len(df)):
        for j in range(len(cols)):
            v = df.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 else "#0b0b0b")
    label = {"tpr_at_5pct": "TPR at 5% FPR", "tpr_at_1pct": "TPR at 1% FPR", "auroc": "AUROC"}.get(metric, metric)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=label)
    ax.set_title(f"Single signals: {label} per attack condition")
    _save(fig, out)


def bars_with_ci(ax, df: pd.DataFrame, group_col: str, series_col: str, metric: str, order_series: list[str],
                 label_fn=lambda s: s) -> None:
    groups = _order(df[group_col].unique()) if group_col == "condition" else list(df[group_col].unique())
    n_s = len(order_series)
    width = 0.8 / n_s
    x = np.arange(len(groups))
    for si, s in enumerate(order_series):
        sub = df[df[series_col] == s].set_index(group_col).reindex(groups)
        vals = sub[metric].to_numpy(float)
        lo = sub.get(f"{metric}_lo", pd.Series(vals, index=sub.index)).to_numpy(float)
        hi = sub.get(f"{metric}_hi", pd.Series(vals, index=sub.index)).to_numpy(float)
        err = np.vstack([np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)])
        ax.bar(x + (si - (n_s - 1) / 2) * width, vals, width * 0.92, color=SERIES[si % len(SERIES)],
               label=label_fn(s), yerr=err, error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "#52514e"})
    ax.set_xticks(x, [_cond_label(g) for g in groups], rotation=30, ha="right")


def fused_vs_single(tables: Path, out: Path, fpr: float = 0.05) -> None:
    f, s = tables / "fused.csv", tables / "single_signals.csv"
    if not f.exists() or not s.exists():
        return
    fused = pd.read_csv(f)
    single = pd.read_csv(s)
    metric = f"tpr@{fpr}"
    single = single[single["signal"] != "n_tokens"]          # the length-only reference is not a candidate
    best = single.groupby("signal")[metric].mean().idxmax()
    best_rows = single[single["signal"] == best].copy()
    best_rows["experiment"] = f"best single ({PRETTY.get(best, best)})"
    df = pd.concat([fused[fused["experiment"].isin(["fused_all", "fused_no_overlap"])], best_rows])
    order = [f"best single ({PRETTY.get(best, best)})", "fused_all", "fused_no_overlap"]
    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    bars_with_ci(ax, df, "condition", "experiment", metric, order, lambda s: PRETTY.get(s, s))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"TPR at {int(fpr * 100)}% FPR")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    _save(fig, out)


def transfer(tables: Path, out: Path, fpr: float = 0.05) -> None:
    p = tables / "transfer.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    order = [t for t in ["train_fact", "train_inj", "train_both"] if t in df["trained_on"].unique()]
    names = {"train_fact": "trained on fact poisoning", "train_inj": "trained on injection", "train_both": "trained on both"}
    bars_with_ci(ax, df, "condition", "trained_on", f"tpr@{fpr}", order, lambda s: names[s])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"TPR at {int(fpr * 100)}% FPR")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    _save(fig, out)


def coefficients(tables: Path, out: Path) -> None:
    p = tables / "coefficients.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("coef_mean")
    fig, ax = plt.subplots(figsize=(COL_W, 0.22 * len(df) + 0.8))
    colors = ["#2a78d6" if v >= 0 else "#eb6834" for v in df["coef_mean"]]
    ax.barh([PRETTY.get(s, s) for s in df["signal"]], df["coef_mean"], xerr=df["coef_std"], color=colors,
            error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "#52514e"})
    ax.axvline(0, color="#52514e", linewidth=0.6)
    ax.set_xlabel("standardised logistic-regression coefficient (positive = more suspicious)")
    _save(fig, out)


def roc_curves(results_dir: Path, out: Path, conditions: list[str], experiments: dict[str, str]) -> None:
    """experiments: {label: scores csv stem}. One panel per condition; negatives = clean passages."""
    panels = []
    for cond in conditions:
        curves = []
        for label, stem in experiments.items():
            p = results_dir / f"scores_{stem}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            pos = df[(df["condition"] == cond) & df["is_poison"].astype(bool)]["score"].to_numpy(float)
            neg = df[(df["condition"] == "clean") & ~df["is_poison"].astype(bool)]["score"].to_numpy(float)
            if len(pos) == 0:
                continue
            ts = np.unique(np.concatenate([pos, neg]))
            fpr = [(neg >= t).mean() for t in ts]
            tpr = [(pos >= t).mean() for t in ts]
            curves.append((label, np.array(fpr), np.array(tpr)))
        if curves:
            panels.append((cond, curves))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(min(FULL_W, 2.3 * len(panels)), 2.3), squeeze=False)
    for ax, (cond, curves) in zip(axes[0], panels):
        for i, (label, fpr, tpr) in enumerate(curves):
            ax.plot(fpr, tpr, color=SERIES[i % len(SERIES)], linewidth=1.2, label=label)
        ax.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=0.6, linestyle="--")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("false-positive rate (log)")
        ax.set_title(_cond_label(cond))
    axes[0][0].set_ylabel("true-positive rate")
    axes[0][0].legend(loc="lower right")
    _save(fig, out)


def endtask(tables: Path, out: Path) -> None:
    files = [f for f in sorted(tables.glob("endtask_*.csv")) if "quick" not in f.name] or sorted(tables.glob("endtask_*.csv"))
    if not files:
        return
    p = files[0]
    df = pd.read_csv(p)
    defences = [d for d in ["none", "filter_fused", "filter_fused_no_overlap", "filter_best_single", "robustrag_keyword", "known_answer"]
                if d in df["defence"].unique()]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 4.2), sharex=True)
    bars_with_ci(axes[0], df, "condition", "defence", "asr", defences, lambda s: PRETTY.get(s, s))
    axes[0].set_ylabel("attack success rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(ncol=len(defences), loc="upper center", bbox_to_anchor=(0.5, 1.25))
    bars_with_ci(axes[1], df, "condition", "defence", "acc", defences, lambda s: PRETTY.get(s, s))
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1.05)
    _save(fig, out)


def make_all(results_dir: Path, fig_dir: Path) -> list[str]:
    tables = results_dir / "tables"
    made = []
    for metric in ("tpr_at_5pct", "tpr_at_1pct", "auroc"):
        heatmap_single(tables, fig_dir / f"single_{metric}", metric)
        made.append(f"single_{metric}")
    fused_vs_single(tables, fig_dir / "fused_vs_single")
    transfer(tables, fig_dir / "transfer")
    coefficients(tables, fig_dir / "coefficients")
    endtask(tables, fig_dir / "endtask")
    scores = results_dir
    conds = [c for c in ["fact_prefix_m5", "fact_noprefix_m5", "inj_carrier_m1"]
             if (scores / "scores_fused_all.csv").exists()]
    roc_curves(scores, fig_dir / "roc", conds, {"fused (all)": "fused_all", "fused (no query copy)": "fused_no_overlap",
                                                "perplexity": "single_log_ppl", "injection clf": "single_inj_prob"})
    made += ["fused_vs_single", "transfer", "coefficients", "endtask", "roc"]
    return made

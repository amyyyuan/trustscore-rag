"""Cross-validated single-signal and fused scoring, with thresholds set on held-out clean data.

Protocol (per retrieval setting)
    * Folds are grouped by question: every passage of a question is in exactly one test fold, so
      no score is ever produced by a model that saw that question.
    * The scorer is a logistic regression on standardised signals (missing values imputed with
      training medians). A single signal is scored by a one-feature logistic regression, which
      learns its direction and scale from the training folds instead of from the test data.
    * The scorer is fitted on the attacked questions' sets only (poisoned and clean passages).
      Thresholds for a target false-positive rate are set on the clean passages of the training
      fold's never-attacked questions, which the scorer was not fitted on, then applied unchanged
      to the test fold. The realised FPR on the test fold is reported next to the TPR so
      calibration can be checked.
    * Confidence intervals are bootstrapped by resampling questions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auroc, bootstrap_by_group, threshold_at_fpr, tpr_at_threshold

A_PRIORI_SIGN = {
    "log_ppl": +1, "log_ppl_win": +1, "inj_prob": +1, "cue_count": +1, "sim_z": +1, "q_overlap": +1,
    "q_lcs": +1, "mean_dist": +1, "clus_tight": +1, "nli_contra_max": +1, "nli_contra_mean": +1,
    "nn_dist": -1, "nli_entail_mean": -1,
}


def _make_model(C: float):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def condition_weights(train: pd.DataFrame) -> np.ndarray:
    """Sample weights that give the planted passages of every attack condition the same total
    weight (an m=5 condition has five times the planted passages of an m=1 condition, and
    retrieval success differs between conditions). Clean passages keep weight 1; the class
    balance between planted and clean is handled separately by the class weights."""
    w = np.ones(len(train))
    pos = train["is_poison"].to_numpy(bool)
    conds = train.loc[pos, "condition"]
    counts = conds.value_counts()
    target = pos.sum() / len(counts)
    w[pos] = conds.map(lambda c: target / counts[c]).to_numpy()
    return w


def cross_validated_scores(df: pd.DataFrame, features: list[str], train_conditions: list[str] | None,
                           fprs: list[float], n_folds: int = 5, C: float = 1.0, seed: int = 0,
                           exclude_from_training: list[str] | None = None,
                           balance_conditions: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df with out-of-fold `score`, `fold`, `thr_<fpr>` columns, coefficient table).

    train_conditions: attacked conditions whose passages (poison and clean) are used for training.
    None means all attacked conditions. The clean condition is always included.
    """
    df = df.copy()
    df["is_target"] = df["is_target"].astype(bool)
    qids = df["qid"].to_numpy()
    rng = np.random.default_rng(seed)
    fold_of_q = {}
    # assign folds separately for attacked and never-attacked questions so every fold has both
    for group in (True, False):
        uniq = np.unique(qids[df["is_target"].to_numpy() == group])
        for i, q in enumerate(rng.permutation(uniq)):
            fold_of_q[q] = i % n_folds
    df["fold"] = [fold_of_q[q] for q in qids]
    df["score"] = np.nan
    for fpr in fprs:
        df[f"thr_{fpr}"] = np.nan
    coef_rows = []
    attacked = sorted(c for c in df["condition"].unique() if c not in ("clean", "random"))
    train_conds = set(train_conditions) if train_conditions is not None else set(attacked)
    train_conds -= set(exclude_from_training or [])
    for fold in range(n_folds):
        # the scorer is fitted on attacked questions only; never-attacked questions are reserved for
        # calibration, so thresholds come from clean traffic the scorer has not been fitted on
        train = df[(df["fold"] != fold) & df["is_target"] & (df["condition"].isin(train_conds | {"clean"}))]
        test_mask = df["fold"] == fold
        X = train[features].to_numpy(float)
        y = train["is_poison"].to_numpy(bool)
        if y.sum() == 0 or (~y).sum() == 0:
            raise ValueError("training fold needs both classes")
        model = _make_model(C)
        if balance_conditions:
            model.fit(X, y, logisticregression__sample_weight=condition_weights(train))
        else:
            model.fit(X, y)
        df.loc[test_mask, "score"] = model.decision_function(df.loc[test_mask, features].to_numpy(float))
        # thresholds from never-attacked questions' clean passages in the training folds
        calib = df[(df["fold"] != fold) & (df["condition"] == "clean") & (~df["is_target"])]
        calib_scores = model.decision_function(calib[features].to_numpy(float))
        for fpr in fprs:
            df.loc[test_mask, f"thr_{fpr}"] = threshold_at_fpr(calib_scores, fpr)
        lr = model.named_steps["logisticregression"]
        coef_rows.append({"fold": fold, **{f: c for f, c in zip(features, lr.coef_[0])}})
    return df, pd.DataFrame(coef_rows)


def summarise(df: pd.DataFrame, fprs: list[float], n_boot: int = 1000, seed: int = 0,
              conditions: list[str] | None = None) -> pd.DataFrame:
    """Per-condition AUROC, TPR at each FPR, realised FPR, and bootstrap CIs."""
    neg = df[(df["condition"] == "clean") & (~df["is_poison"])]
    neg_scores, neg_groups = neg["score"].to_numpy(float), neg["qid"].to_numpy()
    rows = []
    conds = conditions or sorted(c for c in df["condition"].unique() if c not in ("clean", "random"))
    for cond in conds:
        sub = df[df["condition"] == cond]
        pos = sub[sub["is_poison"]]
        if len(pos) == 0:
            continue
        pos_scores, pos_groups = pos["score"].to_numpy(float), pos["qid"].to_numpy()
        row = {"condition": cond, "n_pos": len(pos), "n_neg": len(neg), "n_sets": sub["set_id"].nunique()}
        row["auroc"] = auroc(pos_scores, neg_scores)
        lo, hi = bootstrap_by_group(pos_scores, pos_groups, auroc, n_boot, seed, neg_scores, neg_groups)
        row["auroc_lo"], row["auroc_hi"] = lo, hi
        # scores are pooled across the fold models above; the mean of the per-fold AUROCs is the
        # version that never compares scores from two different models
        per_fold = [auroc(pos[pos["fold"] == f]["score"].to_numpy(float), neg[neg["fold"] == f]["score"].to_numpy(float))
                    for f in sorted(df["fold"].unique()) if (pos["fold"] == f).any() and (neg["fold"] == f).any()]
        row["auroc_fold_mean"] = float(np.nanmean(per_fold)) if per_fold else float("nan")
        collateral = sub[~sub["is_poison"]]
        for fpr in fprs:
            thr_pos = pos[f"thr_{fpr}"].to_numpy(float)
            hits = (pos_scores >= thr_pos).astype(float)
            row[f"tpr@{fpr}"] = float(hits.mean())
            lo, hi = bootstrap_by_group(hits, pos_groups, np.mean, n_boot, seed)
            row[f"tpr@{fpr}_lo"], row[f"tpr@{fpr}_hi"] = lo, hi
            row[f"fpr@{fpr}_realised"] = float(np.mean(neg_scores >= neg[f"thr_{fpr}"].to_numpy(float)))
            if len(collateral):
                row[f"fpr@{fpr}_collateral"] = float(np.mean(
                    collateral["score"].to_numpy(float) >= collateral[f"thr_{fpr}"].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def raw_auroc_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """AUROC of each raw signal in its a-priori direction (no learning, for reference)."""
    neg = df[(df["condition"] == "clean") & (~df["is_poison"])]
    rows = []
    for cond in sorted(c for c in df["condition"].unique() if c not in ("clean", "random")):
        pos = df[(df["condition"] == cond) & (df["is_poison"])]
        if len(pos) == 0:
            continue
        row = {"condition": cond}
        for f in features:
            sign = A_PRIORI_SIGN.get(f, +1)
            p, n = sign * pos[f].to_numpy(float), sign * neg[f].to_numpy(float)
            p, n = p[~np.isnan(p)], n[~np.isnan(n)]
            row[f] = auroc(p, n)
        rows.append(row)
    return pd.DataFrame(rows)


def fpr_on(df_scored_other: pd.DataFrame, thresholds: dict[float, float]) -> dict[str, float]:
    """FPR of fixed thresholds on another clean population (e.g. random passages)."""
    out = {}
    for fpr, t in thresholds.items():
        out[f"fpr@{fpr}"] = float(np.mean(df_scored_other["score"].to_numpy(float) >= t))
    return out

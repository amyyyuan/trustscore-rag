"""Detection metrics: AUROC, TPR at a fixed FPR with thresholds set on held-out clean data, and
bootstrap confidence intervals that resample questions rather than passages (passages from one
question are not independent)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC: P(score(pos) > score(neg)), ties count half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def threshold_at_fpr(clean_scores: np.ndarray, fpr: float) -> float:
    """Smallest threshold t such that at most `fpr` of clean scores are >= t (higher = more suspicious)."""
    clean = np.sort(np.asarray(clean_scores, float))
    if len(clean) == 0:
        return float("inf")
    n_allowed = int(np.floor(fpr * len(clean)))
    # the (n_allowed)-th largest value must be below threshold; t sits just above the (n_allowed+1)-th largest
    idx = len(clean) - n_allowed - 1
    idx = max(idx, 0)
    t = clean[idx]
    return float(np.nextafter(t, np.inf))


def tpr_at_threshold(scores: np.ndarray, t: float) -> float:
    scores = np.asarray(scores, float)
    return float(np.mean(scores >= t)) if len(scores) else float("nan")


def bootstrap_by_group(values: np.ndarray, groups: np.ndarray, stat, n_boot: int = 1000, seed: int = 0,
                       values2: np.ndarray | None = None, groups2: np.ndarray | None = None) -> tuple[float, float]:
    """95% percentile CI of stat(values[, values2]) resampling groups with replacement.

    When values2/groups2 are given (e.g. negatives for AUROC), both samples are resampled by group.
    """
    rng = np.random.default_rng(seed)
    values, groups = np.asarray(values), np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    if values2 is not None:
        values2, groups2 = np.asarray(values2), np.asarray(groups2)
        uniq2 = np.unique(groups2)
        idx_by_group2 = {g: np.where(groups2 == g)[0] for g in uniq2}
    stats = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in gs])
        if values2 is None:
            stats.append(stat(values[idx]))
        else:
            gs2 = rng.choice(uniq2, size=len(uniq2), replace=True)
            idx2 = np.concatenate([idx_by_group2[g] for g in gs2])
            stats.append(stat(values[idx], values2[idx2]))
    stats = np.array(stats, float)
    return float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


def rouge_l_f1(a_tokens: list[str], b_tokens: list[str]) -> float:
    """ROUGE-L F-measure on token lists (LCS based), used by the TrustRAG baseline."""
    if not a_tokens or not b_tokens:
        return 0.0
    m, n = len(a_tokens), len(b_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a_tokens[i - 1]
        for j in range(1, n + 1):
            if ai == b_tokens[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    p, r = lcs / n, lcs / m
    return 2 * p * r / (p + r)

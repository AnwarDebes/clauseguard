"""Task metrics + interpretability metrics + stats helpers.

Mirrors ``paper-c-tm-robustness/src/eval/stats.py`` for the stats
helpers (paired Wilcoxon, bootstrap CI) so cross-paper comparisons use
identical machinery.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


# --------------------------------------------------------------------------
# Task metrics
# --------------------------------------------------------------------------

def label_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if len(yt) == 0:
        return float("nan")
    return float((yt == yp).mean())


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = 3) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    f1s: list[float] = []
    for c in range(n_classes):
        tp = int(((yt == c) & (yp == c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def fever_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    ev_recall_correct: Sequence[bool],
) -> float:
    """FEVER score: label correct AND evidence recall correct.

    Following Thorne et al. (2018), a sample contributes to the FEVER
    score only if both the label is correct AND at least one of the
    predicted evidence sentences matches the gold evidence (the
    ``ev_recall_correct`` flag).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    ec = np.asarray(ev_recall_correct, dtype=bool)
    if len(yt) == 0:
        return float("nan")
    return float(((yt == yp) & ec).mean())


def auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Binary AUROC using the Mann-Whitney U formulation.

    For multi-class outputs, compute one-vs-rest AUROC per class and
    average outside this function.
    """
    yt = np.asarray(y_true).astype(int)
    ys = np.asarray(y_score, dtype=float)
    pos = ys[yt == 1]
    neg = ys[yt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Compute U via rank sum.
    combined = np.concatenate([pos, neg])
    ranks = combined.argsort().argsort() + 1
    rank_pos = ranks[: len(pos)].sum()
    u = rank_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


# --------------------------------------------------------------------------
# Stats helpers, mirrors paper-c
# --------------------------------------------------------------------------

def paired_wilcoxon(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alternative: str = "greater",
) -> dict:
    """Paired Wilcoxon signed-rank test via scipy.

    Returns ``{"statistic": float, "pvalue": float, "alternative": str}``.
    """
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"a and b must have same length, got {len(a)} vs {len(b)}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(alternative)
    res = wilcoxon(a, b, alternative=alternative)
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "alternative": alternative,
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap mean + (lo, hi) CI at ``confidence`` level."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.array(
        [arr[rng.integers(0, n, size=n)].mean() for _ in range(n_resamples)]
    )
    lo = float(np.percentile(boots, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(boots, (1 + confidence) / 2 * 100))
    return float(arr.mean()), lo, hi


# --------------------------------------------------------------------------
# Interpretability metrics
# --------------------------------------------------------------------------

def clause_set_size(verifier) -> int:
    """Total clause count across all classes."""
    cfg = getattr(verifier, "config", None)
    if cfg is None:
        return 0
    return int(getattr(cfg, "n_classes", 3) * getattr(cfg, "number_of_clauses", 0))


def median_clause_length(clause_specs: Sequence) -> float:
    """Median number of literals per clause across ``clause_specs``.

    ``clause_specs`` is the export from :class:`GraphTMVerifier`'s
    ``export_clause_summary`` or an equivalent walker. When empty we
    return ``nan`` so callers can detect the no-clauses path.
    """
    lengths = []
    for c in clause_specs:
        if hasattr(c, "positive_literal_ids") and hasattr(c, "negative_literal_ids"):
            lengths.append(len(c.positive_literal_ids) + len(c.negative_literal_ids))
        elif isinstance(c, dict) and "literals" in c:
            lengths.append(len(c["literals"]))
    if not lengths:
        return float("nan")
    return float(np.median(lengths))


def sat_verification_latency(latencies_ms: Sequence[float]) -> dict:
    arr = np.asarray(latencies_ms, dtype=float)
    if len(arr) == 0:
        return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan")}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
    }

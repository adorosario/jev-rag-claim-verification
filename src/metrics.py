"""Benchmark metrics, hand-implemented (spec §18).

This module is deliberately written without sklearn. `tests/test_metrics_crosscheck.py`
recomputes everything with sklearn/scipy on random fixtures and requires agreement
to 1e-12, so an error here has to be reproduced independently in two libraries to
survive. Nothing in this file may import from sklearn.

Convention throughout: gold labels are ints in {0, 1} where 1 == SUPPORTED, and
`p` is p_supported in [0, 1].
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Spec §18.4 requires negative log loss, but Jev returns probabilities quantized
# to 2dp that saturate at exactly 0.0 and 1.0 (see docs/reference/jev-api.md).
# log(0) is -inf, so a clipping epsilon is unavoidable. The VALUE is a protocol
# decision preregistered at G2A and applied identically to every model with a
# native probability; this is only the default used on dev.
DEFAULT_LOGLOSS_EPS = 1e-15


@dataclass(frozen=True)
class Confusion:
    tp: int  # gold 1, pred 1
    fp: int  # gold 0, pred 1
    tn: int  # gold 0, pred 0
    fn: int  # gold 1, pred 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn


def confusion(gold: Sequence[int], pred: Sequence[int]) -> Confusion:
    if len(gold) != len(pred):
        raise ValueError("gold and pred must be the same length")
    tp = fp = tn = fn = 0
    for g, p in zip(gold, pred, strict=True):
        if g not in (0, 1) or p not in (0, 1):
            raise ValueError(f"labels must be 0/1, got gold={g!r} pred={p!r}")
        if g == 1 and p == 1:
            tp += 1
        elif g == 0 and p == 1:
            fp += 1
        elif g == 0 and p == 0:
            tn += 1
        else:
            fn += 1
    return Confusion(tp, fp, tn, fn)


def _safe_div(num: float, den: float) -> float:
    return float("nan") if den == 0 else num / den


def balanced_accuracy(gold: Sequence[int], pred: Sequence[int]) -> float:
    """Spec §18.1: (TPR + TNR) / 2."""
    c = confusion(gold, pred)
    tpr = _safe_div(c.tp, c.tp + c.fn)
    tnr = _safe_div(c.tn, c.tn + c.fp)
    return (tpr + tnr) / 2


def mean_balanced_accuracy(per_dataset: dict[str, float]) -> float:
    """Spec §18.1 headline: the mean of per-dataset BAcc, NOT a pooled BAcc.

    Pooling would let RAGTruth, which is half of the dev split, dominate the
    headline number.
    """
    vals = list(per_dataset.values())
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def balanced_accuracy_by_dataset(
    gold: Sequence[int], pred: Sequence[int], dataset: Sequence[str]
) -> dict[str, float]:
    groups: dict[str, tuple[list[int], list[int]]] = {}
    for g, p, d in zip(gold, pred, dataset, strict=True):
        groups.setdefault(d, ([], []))
        groups[d][0].append(g)
        groups[d][1].append(p)
    return {d: balanced_accuracy(gs, ps) for d, (gs, ps) in sorted(groups.items())}


def accuracy(gold: Sequence[int], pred: Sequence[int]) -> float:
    c = confusion(gold, pred)
    return _safe_div(c.tp + c.tn, c.n)


def precision_recall_f1(gold: Sequence[int], pred: Sequence[int], positive: int
                        ) -> tuple[float, float, float]:
    c = confusion(gold, pred)
    if positive == 1:
        tp, fp, fn = c.tp, c.fp, c.fn
    else:
        tp, fp, fn = c.tn, c.fn, c.fp
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    if math.isnan(prec) or math.isnan(rec) or (prec + rec) == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def macro_f1(gold: Sequence[int], pred: Sequence[int]) -> float:
    _, _, f1_pos = precision_recall_f1(gold, pred, 1)
    _, _, f1_neg = precision_recall_f1(gold, pred, 0)
    return (f1_pos + f1_neg) / 2


def mcc(gold: Sequence[int], pred: Sequence[int]) -> float:
    c = confusion(gold, pred)
    num = c.tp * c.tn - c.fp * c.fn
    den = math.sqrt(float((c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn)))
    return 0.0 if den == 0 else num / den


def false_verification_rate(gold: Sequence[int], pred: Sequence[int]) -> float:
    """Spec §18.3, the CustomGPT-critical metric.

    Of everything genuinely NOT supported, what fraction did we verify anyway?
    """
    c = confusion(gold, pred)
    return _safe_div(c.fp, c.fp + c.tn)


def verified_precision(gold: Sequence[int], pred: Sequence[int]) -> float:
    """Spec §18.3: of everything we called SUPPORTED, how much really was."""
    c = confusion(gold, pred)
    return _safe_div(c.tp, c.tp + c.fp)


def brier(gold: Sequence[int], p: Sequence[float]) -> float:
    """Spec §18.4: mean((p_supported - gold)^2). Bounded, so saturation-safe."""
    if len(gold) != len(p):
        raise ValueError("gold and p must be the same length")
    if not gold:
        return float("nan")
    return sum((pi - gi) ** 2 for gi, pi in zip(gold, p, strict=True)) / len(gold)


def negative_log_loss(gold: Sequence[int], p: Sequence[float],
                      eps: float = DEFAULT_LOGLOSS_EPS) -> float:
    """Spec §18.4. `eps` clips away from 0/1; see the note at the top of the
    module -- with saturating probabilities this is load-bearing, not cosmetic."""
    if len(gold) != len(p):
        raise ValueError("gold and p must be the same length")
    if not gold:
        return float("nan")
    total = 0.0
    for g, pi in zip(gold, p, strict=True):
        q = min(max(pi, eps), 1.0 - eps)
        total += -(g * math.log(q) + (1 - g) * math.log(1.0 - q))
    return total / len(gold)


def equal_frequency_bins(p: Sequence[float], n_bins: int) -> list[list[int]]:
    """Indices grouped into n_bins of near-equal COUNT (spec §18.4 headline).

    Equal-frequency, not equal-width: with probabilities piled on 0.0 and 1.0,
    equal-width bins would leave most bins empty and make ECE meaningless.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    order = sorted(range(len(p)), key=lambda i: p[i])
    n = len(order)
    if n == 0:
        return []
    bins, start = [], 0
    for b in range(n_bins):
        end = round(n * (b + 1) / n_bins)
        if end > start:
            bins.append(order[start:end])
            start = end
    return bins


def ece(gold: Sequence[int], p: Sequence[float], n_bins: int = 10) -> float:
    """Expected Calibration Error over equal-frequency bins."""
    if len(gold) != len(p):
        raise ValueError("gold and p must be the same length")
    n = len(gold)
    if n == 0:
        return float("nan")
    total = 0.0
    for idx in equal_frequency_bins(p, n_bins):
        conf = sum(p[i] for i in idx) / len(idx)
        acc = sum(gold[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(acc - conf)
    return total


def auroc(gold: Sequence[int], p: Sequence[float]) -> float:
    """AUROC via the rank (Mann-Whitney U) identity, with tie correction."""
    pos = [pi for g, pi in zip(gold, p, strict=True) if g == 1]
    neg = [pi for g, pi in zip(gold, p, strict=True) if g == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank across the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, g in zip(ranks, gold, strict=True) if g == 1)
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def auprc(gold: Sequence[int], p: Sequence[float]) -> float:
    """Average precision, matching sklearn's `average_precision_score`:
    sum over thresholds of (recall_k - recall_{k-1}) * precision_k."""
    n_pos = sum(gold)
    if n_pos == 0:
        return float("nan")
    order = sorted(range(len(p)), key=lambda i: -p[i])
    tp = fp = 0
    prev_recall = 0.0
    total = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        for k in range(i, j + 1):  # consume the whole tie block at once
            if gold[order[k]] == 1:
                tp += 1
            else:
                fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        total += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1
    return total


def classification_report(gold: Sequence[int], pred: Sequence[int]) -> dict[str, float]:
    sp, sr, _ = precision_recall_f1(gold, pred, 1)
    up, ur, _ = precision_recall_f1(gold, pred, 0)
    return {
        "accuracy": accuracy(gold, pred),
        "balanced_accuracy": balanced_accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred),
        "mcc": mcc(gold, pred),
        "supported_precision": sp,
        "supported_recall": sr,
        "unsupported_precision": up,
        "unsupported_recall": ur,
        "false_verification_rate": false_verification_rate(gold, pred),
        "verified_precision": verified_precision(gold, pred),
    }

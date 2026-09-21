"""Hand-computed metric checks (spec §36).

Every expected value here is worked out by hand from a small confusion matrix,
so these tests are independent of both src/metrics.py and sklearn.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import metrics as M  # noqa: E402

# Deliberately imbalanced so BAcc and accuracy disagree.
#   gold: 1 1 1 1 0 0 0 0 0 0
#   pred: 1 1 1 0 1 0 0 0 0 0
# TP=3, FN=1, FP=1, TN=5
GOLD = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
PRED = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]


def test_confusion_matrix_by_hand():
    c = M.confusion(GOLD, PRED)
    assert (c.tp, c.fn, c.fp, c.tn) == (3, 1, 1, 5)
    assert c.n == 10


def test_accuracy_by_hand():
    assert M.accuracy(GOLD, PRED) == pytest.approx(8 / 10)


def test_balanced_accuracy_by_hand():
    # TPR = 3/4 = 0.75, TNR = 5/6 = 0.8333...
    assert M.balanced_accuracy(GOLD, PRED) == pytest.approx((0.75 + 5 / 6) / 2)


def test_balanced_accuracy_differs_from_accuracy_on_imbalance():
    assert M.balanced_accuracy(GOLD, PRED) != pytest.approx(M.accuracy(GOLD, PRED))


def test_false_verification_rate_by_hand():
    # FP / (FP + TN) = 1 / 6
    assert M.false_verification_rate(GOLD, PRED) == pytest.approx(1 / 6)


def test_verified_precision_by_hand():
    # TP / (TP + FP) = 3 / 4
    assert M.verified_precision(GOLD, PRED) == pytest.approx(3 / 4)


def test_fvr_is_zero_when_nothing_false_is_verified():
    assert M.false_verification_rate([0, 0, 1], [0, 0, 1]) == pytest.approx(0.0)


def test_fvr_is_one_when_everything_false_is_verified():
    assert M.false_verification_rate([0, 0], [1, 1]) == pytest.approx(1.0)


def test_precision_recall_f1_by_hand():
    p, r, f1 = M.precision_recall_f1(GOLD, PRED, 1)
    assert p == pytest.approx(3 / 4)
    assert r == pytest.approx(3 / 4)
    assert f1 == pytest.approx(3 / 4)
    p0, r0, f10 = M.precision_recall_f1(GOLD, PRED, 0)
    assert p0 == pytest.approx(5 / 6)
    assert r0 == pytest.approx(5 / 6)
    assert f10 == pytest.approx(5 / 6)


def test_macro_f1_by_hand():
    assert M.macro_f1(GOLD, PRED) == pytest.approx((3 / 4 + 5 / 6) / 2)


def test_mcc_by_hand():
    # (3*5 - 1*1) / sqrt(4*4*6*6) = 14 / 24
    assert M.mcc(GOLD, PRED) == pytest.approx(14 / 24)


def test_brier_by_hand():
    gold = [1, 0]
    p = [0.75, 0.25]
    assert M.brier(gold, p) == pytest.approx(((0.75 - 1) ** 2 + (0.25 - 0) ** 2) / 2)


def test_brier_is_bounded_under_saturation():
    """Unlike log loss, Brier survives p == 0.0 with gold == 1."""
    assert M.brier([1], [0.0]) == pytest.approx(1.0)
    assert M.brier([0], [1.0]) == pytest.approx(1.0)


def test_negative_log_loss_by_hand():
    gold = [1, 0]
    p = [0.75, 0.25]
    expected = -(math.log(0.75) + math.log(0.75)) / 2
    assert M.negative_log_loss(gold, p) == pytest.approx(expected)


def test_log_loss_is_finite_under_saturation():
    """The reason DEFAULT_LOGLOSS_EPS exists: Jev returns exact 0.0 and 1.0."""
    v = M.negative_log_loss([1, 0], [0.0, 1.0])
    assert math.isfinite(v)
    assert v > 0


def test_log_loss_without_clipping_would_be_infinite():
    with pytest.raises((ValueError, OverflowError)):
        math.log(0.0)


def test_perfect_prediction_scores():
    gold = [1, 1, 0, 0]
    assert M.balanced_accuracy(gold, gold) == pytest.approx(1.0)
    assert M.false_verification_rate(gold, gold) == pytest.approx(0.0)
    assert M.mcc(gold, gold) == pytest.approx(1.0)


def test_inverted_prediction_scores():
    gold = [1, 1, 0, 0]
    pred = [0, 0, 1, 1]
    assert M.balanced_accuracy(gold, pred) == pytest.approx(0.0)
    assert M.mcc(gold, pred) == pytest.approx(-1.0)


def test_mean_bacc_is_macro_not_pooled():
    """Spec §18.1. A huge dataset must not dominate the headline.

    Dataset 'big' has 100 rows at BAcc 0.5; 'small' has 2 rows at BAcc 1.0.
    Macro mean is 0.75; pooling would land near 0.5.
    """
    gold = [1, 0] * 50 + [1, 0]
    pred = [1, 1] * 50 + [1, 0]
    ds = ["big"] * 100 + ["small"] * 2
    per = M.balanced_accuracy_by_dataset(gold, pred, ds)
    assert per["big"] == pytest.approx(0.5)
    assert per["small"] == pytest.approx(1.0)
    assert M.mean_balanced_accuracy(per) == pytest.approx(0.75)
    assert M.balanced_accuracy(gold, pred) < 0.6


def test_equal_frequency_bins_are_balanced():
    p = [i / 100 for i in range(100)]
    bins = M.equal_frequency_bins(p, 10)
    assert len(bins) == 10
    assert all(len(b) == 10 for b in bins)
    assert sorted(i for b in bins for i in b) == list(range(100))


def test_equal_frequency_bins_survive_saturated_probabilities():
    """Equal-WIDTH bins would leave 8 of 10 bins empty here."""
    p = [0.0] * 50 + [1.0] * 50
    bins = M.equal_frequency_bins(p, 10)
    assert sum(len(b) for b in bins) == 100
    assert all(len(b) > 0 for b in bins)


def test_ece_is_zero_for_perfectly_calibrated_bins():
    gold = [1, 1, 0, 0]
    p = [1.0, 1.0, 0.0, 0.0]
    assert M.ece(gold, p, n_bins=2) == pytest.approx(0.0)


def test_ece_is_one_for_maximally_miscalibrated():
    assert M.ece([0, 0], [1.0, 1.0], n_bins=1) == pytest.approx(1.0)


def test_confusion_rejects_bad_labels():
    with pytest.raises(ValueError):
        M.confusion([2], [1])
    with pytest.raises(ValueError):
        M.confusion([1, 0], [1])


def test_classification_report_keys():
    r = M.classification_report(GOLD, PRED)
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "mcc",
              "supported_precision", "supported_recall",
              "unsupported_precision", "unsupported_recall",
              "false_verification_rate", "verified_precision"):
        assert k in r

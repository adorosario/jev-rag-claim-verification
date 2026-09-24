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


def test_clip_share_counts_only_the_items_the_clip_actually_prices():
    """Section 8 prints how much of the NLL is the clipping constant rather than the
    model, so the count has to be the items charged -log(eps) and nothing else. The
    confident-and-right items are clipped too, at about eps nats, and including them
    would inflate the share the paper reports."""
    eps = 1e-15
    gold = [1, 1, 0, 0]
    p = [0.0, 1.0, 1.0, 0.0]   # wrong-and-certain, right, wrong-and-certain, right
    d = M.negative_log_loss_clip_share(gold, p, eps=eps)
    assert d["n"] == 2
    assert d["nats_each"] == pytest.approx(-math.log(eps), rel=1e-4)
    assert d["total_nats"] / len(gold) == pytest.approx(
        M.negative_log_loss(gold, p, eps=eps))
    assert d["share"] == pytest.approx(d["clipped_nats"] / d["total_nats"])
    assert d["share"] == pytest.approx(1.0, abs=1e-12)


def test_clip_share_is_zero_when_no_probability_saturates():
    d = M.negative_log_loss_clip_share([1, 0], [0.75, 0.25])
    assert d["n"] == 0
    assert d["share"] == pytest.approx(0.0)


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


# --- what the clip costs, and at what resolution the vendor reports -----------------
# Both by hand. A saturating probability head makes the reported log loss depend on a
# constant the protocol picked rather than on the model, and the paper may not print the
# figure without printing that dependence, so the dependence is a measured quantity.
RES_P = [0.0, 0.0, 1.0, 0.87, 0.13, 0.5]


def test_probability_resolution_reads_the_step_off_the_values():
    r = M.probability_resolution(RES_P)
    assert r["decimals"] == 2
    assert r["step"] == pytest.approx(0.01)
    assert r["half_step"] == pytest.approx(0.005)
    assert r["n"] == 6 and r["n_distinct"] == 5
    assert r["n_at_zero"] == 2 and r["n_at_one"] == 1
    assert r["min_positive"] == pytest.approx(0.13)
    assert r["max_below_one"] == pytest.approx(0.87)


def test_probability_resolution_does_not_round_a_finer_head_to_two_places():
    r = M.probability_resolution([0.125, 0.5])
    assert r["decimals"] == 3 and r["step"] == pytest.approx(0.001)


def test_eps_sensitivity_reference_is_a_constant_one_half_forecast():
    # Two supported claims the model gets right at 0.9, one supported claim it calls
    # impossible and is charged the clip for. Below eps = 0.1 the clip touches only the
    # third item, so NLL(eps) = (2 * -log(0.9) - log(eps)) / 3. Above it the clip also
    # pulls the two confident items down to 1 - eps, and the crossing against log 2 lands
    # there: eps * (1 - eps)^2 = 1/8, whose root is (3 - sqrt 5) / 4 = 0.190983...
    # That is why the function scans the grid for a bracket before it bisects; a
    # bisection started on the small-eps formula would report 0.1543 and be wrong.
    gold, p = [1, 1, 1], [0.9, 0.9, 0.0]
    s = M.negative_log_loss_eps_sensitivity(gold, p, grid=(1e-9, 1e-6, 1e-3, 1e-1, 3e-1),
                                            eps=1e-9)
    assert s["reference_nll"] == pytest.approx(math.log(2))
    assert s["reported_nll"] == pytest.approx((2 * -math.log(0.9) - math.log(1e-9)) / 3)
    by_eps = {row["eps"]: row["nll"] for row in s["grid"]}
    assert by_eps[1e-3] == pytest.approx((2 * -math.log(0.9) - math.log(1e-3)) / 3)
    expected = (3 - math.sqrt(5)) / 4
    assert expected * (1 - expected) ** 2 == pytest.approx(0.125)
    assert s["crossover_eps"] == pytest.approx(expected, rel=1e-6)
    assert 1e-1 < s["crossover_eps"] < 3e-1


def test_eps_sensitivity_will_not_invent_a_crossing_the_grid_never_brackets():
    # The same data, on a grid that stops below the crossing. Reporting a constant the
    # grid never reached would be an extrapolation, and the paper prints this number.
    gold, p = [1, 1, 1], [0.9, 0.9, 0.0]
    s = M.negative_log_loss_eps_sensitivity(gold, p, grid=(1e-9, 1e-6, 1e-3), eps=1e-9)
    assert s["crossover_eps"] is None


def test_eps_sensitivity_reports_no_crossing_where_there_is_none():
    # A model that is simply wrong, at no boundary: no clip can rescue it, so there is
    # nothing to report and the function may not invent a constant.
    gold, p = [1, 1, 1, 0], [0.2, 0.2, 0.2, 0.8]
    s = M.negative_log_loss_eps_sensitivity(gold, p, grid=(1e-9, 1e-6, 1e-3, 1e-1))
    assert s["reported_exceeds_reference"]
    assert s["crossover_eps"] is None


def test_eps_sensitivity_agrees_with_the_log_loss_it_re_scores():
    gold = GOLD
    p = [0.9, 0.8, 1.0, 0.0, 1.0, 0.1, 0.0, 0.2, 0.3, 0.0]
    s = M.negative_log_loss_eps_sensitivity(gold, p, grid=(1e-12, 1e-6, 1e-2))
    for row in s["grid"]:
        assert row["nll"] == pytest.approx(
            M.negative_log_loss(gold, p, eps=row["eps"]))
        assert row["exceeds_reference"] == (row["nll"] > s["reference_nll"])

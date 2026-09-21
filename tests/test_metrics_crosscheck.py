"""Independent recomputation of every metric with sklearn/scipy (GOALS G0).

src/metrics.py is hand-written and imports no sklearn. This file recomputes the
same quantities with the reference libraries on randomized fixtures and requires
agreement to 1e-12. A bug therefore has to exist identically in two independent
implementations to go unnoticed.
"""

import random
import sys
from pathlib import Path

import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import metrics as M  # noqa: E402

TOL = 1e-12
N_FIXTURES = 40


def _fixtures():
    """Randomized cases, including the degenerate ones that break naive code."""
    rng = random.Random(20260918)  # seeded: failures are reproducible
    cases = []
    for _ in range(N_FIXTURES):
        n = rng.randint(20, 200)
        gold = [rng.randint(0, 1) for _ in range(n)]
        if len(set(gold)) == 1:  # keep both classes present
            gold[0] = 1 - gold[0]
        p = [rng.random() for _ in range(n)]
        pred = [1 if pi >= 0.5 else 0 for pi in p]
        cases.append((gold, pred, p))
    # saturated + heavily tied probabilities, i.e. what Jev actually returns
    gold = [rng.randint(0, 1) for _ in range(200)]
    gold[0], gold[1] = 0, 1
    p = [rng.choice([0.0, 0.01, 0.5, 0.93, 1.0]) for _ in range(200)]
    cases.append((gold, [1 if pi >= 0.5 else 0 for pi in p], p))
    return cases


FIXTURES = _fixtures()


@pytest.mark.parametrize("gold,pred,p", FIXTURES)
def test_balanced_accuracy_matches_sklearn(gold, pred, p):
    assert M.balanced_accuracy(gold, pred) == pytest.approx(
        balanced_accuracy_score(gold, pred), abs=TOL)


@pytest.mark.parametrize("gold,pred,p", FIXTURES)
def test_accuracy_matches_sklearn(gold, pred, p):
    assert M.accuracy(gold, pred) == pytest.approx(accuracy_score(gold, pred), abs=TOL)


@pytest.mark.parametrize("gold,pred,p", FIXTURES)
def test_macro_f1_matches_sklearn(gold, pred, p):
    assert M.macro_f1(gold, pred) == pytest.approx(
        f1_score(gold, pred, average="macro", zero_division=0), abs=TOL)


@pytest.mark.parametrize("gold,pred,p", FIXTURES)
def test_mcc_matches_sklearn(gold, pred, p):
    assert M.mcc(gold, pred) == pytest.approx(matthews_corrcoef(gold, pred), abs=TOL)


@pytest.mark.parametrize("gold,pred,p", FIXTURES)
def test_supported_precision_recall_match_sklearn(gold, pred, p):
    prec, rec, _ = M.precision_recall_f1(gold, pred, 1)
    assert prec == pytest.approx(precision_score(gold, pred, pos_label=1, zero_division=0), abs=TOL)
    assert rec == pytest.approx(recall_score(gold, pred, pos_label=1, zero_division=0), abs=TOL)


@pytest.mark.parametrize("gold,pred,p", FIXTURES)
def test_unsupported_precision_recall_match_sklearn(gold, pred, p):
    prec, rec, _ = M.precision_recall_f1(gold, pred, 0)
    assert prec == pytest.approx(precision_score(gold, pred, pos_label=0, zero_division=0), abs=TOL)
    assert rec == pytest.approx(recall_score(gold, pred, pos_label=0, zero_division=0), abs=TOL)


@pytest.mark.parametrize("gold,pred,p", FIXTURES)
def test_verified_precision_is_supported_precision(gold, pred, p):
    """§18.3 verified precision is precision on the SUPPORTED class."""
    assert M.verified_precision(gold, pred) == pytest.approx(
        precision_score(gold, pred, pos_label=1, zero_division=0), abs=TOL)


@pytest.mark.parametrize("gold,pred,p", FIXTURES)
def test_fvr_is_one_minus_unsupported_recall(gold, pred, p):
    """FVR = FP/(FP+TN) = 1 - TNR, and TNR is recall on the 0 class."""
    tnr = recall_score(gold, pred, pos_label=0, zero_division=0)
    assert M.false_verification_rate(gold, pred) == pytest.approx(1 - tnr, abs=TOL)


@pytest.mark.parametrize("gold,pred,p", FIXTURES)
def test_brier_matches_sklearn(gold, pred, p):
    assert M.brier(gold, p) == pytest.approx(brier_score_loss(gold, p), abs=TOL)


@pytest.mark.parametrize("gold,pred,p", FIXTURES)
def test_log_loss_matches_sklearn(gold, pred, p):
    expected = log_loss(gold, p, labels=[0, 1], eps=M.DEFAULT_LOGLOSS_EPS) \
        if "eps" in log_loss.__code__.co_varnames else None
    if expected is None:
        # newer sklearn dropped the eps kwarg; clip to match our semantics
        clipped = [min(max(pi, M.DEFAULT_LOGLOSS_EPS), 1 - M.DEFAULT_LOGLOSS_EPS) for pi in p]
        expected = log_loss(gold, clipped, labels=[0, 1])
    assert M.negative_log_loss(gold, p) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("gold,pred,p", FIXTURES)
def test_auroc_matches_sklearn(gold, pred, p):
    assert M.auroc(gold, p) == pytest.approx(roc_auc_score(gold, p), abs=TOL)


@pytest.mark.parametrize("gold,pred,p", FIXTURES)
def test_auprc_matches_sklearn(gold, pred, p):
    assert M.auprc(gold, p) == pytest.approx(average_precision_score(gold, p), abs=TOL)


def test_auroc_handles_ties_exactly():
    """Every probability identical: AUROC must be exactly 0.5, not rank noise."""
    gold = [1, 0, 1, 0]
    p = [0.5, 0.5, 0.5, 0.5]
    assert M.auroc(gold, p) == pytest.approx(roc_auc_score(gold, p), abs=TOL)
    assert M.auroc(gold, p) == pytest.approx(0.5, abs=TOL)


def test_per_dataset_bacc_matches_sklearn_per_group():
    rng = random.Random(7)
    gold, pred, ds = [], [], []
    for name in ("A", "B", "C"):
        for _ in range(60):
            gold.append(rng.randint(0, 1))
            pred.append(rng.randint(0, 1))
            ds.append(name)
    ours = M.balanced_accuracy_by_dataset(gold, pred, ds)
    for name in ("A", "B", "C"):
        g = [x for x, d in zip(gold, ds) if d == name]
        pr = [x for x, d in zip(pred, ds) if d == name]
        assert ours[name] == pytest.approx(balanced_accuracy_score(g, pr), abs=TOL)

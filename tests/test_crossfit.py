"""The cross-fitting in scripts/verify/paired_analysis.py, tested on the real 495.

Two properties, both of which the paper's section 10 leans on:

1. **Every fold contains all 11 datasets.** The headline is a macro average over
   datasets, so a fold that is missing one is scoring a different quantity from the
   one being reported. Stratifying by (dataset, gold) is what makes this hold, and
   a refactor that swapped in a plain KFold would break it silently, because the
   numbers would still come out looking plausible.

2. **The out-of-sample estimate does not exceed the in-sample estimate.** On this
   data the cascade threshold is selected on noise, so honest evaluation has to cost
   something. If cross-fitting ever reported MORE than the in-sample best, the fitting
   and the scoring would be leaking into each other and the whole selection-optimism
   result would be an artifact.

    docker compose run --rm dev uv run pytest tests/test_crossfit.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _export import repo_only, needs_gold_labels  # noqa: E402

# Cross-fitting is scored against the gold labels, which come from the gated dataset.
pytestmark = needs_gold_labels

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "verify"))

import paired_analysis as PA  # noqa: E402

from src import metrics as M  # noqa: E402

N_DATASETS = 11
SEED = 20260918


@pytest.fixture(scope="module")
def frame():
    df, _ = PA.build_frame()
    assert len(df) == 495, f"expected the locked 495 paired examples, got {len(df)}"
    assert df.ds.nunique() == N_DATASETS
    return df


def test_every_fold_contains_every_dataset(frame):
    """Across many shuffles, no fold may ever lose a dataset."""
    rng = np.random.default_rng(SEED)
    all_datasets = set(frame.ds.unique())
    for repeat in range(200):
        fold = PA.stratified_folds(frame, PA.K_FOLDS, rng)
        assert set(np.unique(fold)) == set(range(PA.K_FOLDS))
        for k in range(PA.K_FOLDS):
            present = set(frame.ds.to_numpy()[fold == k])
            missing = all_datasets - present
            assert not missing, (
                f"repeat {repeat}, fold {k} is missing {sorted(missing)}: macro balanced "
                "accuracy on that fold would not be the reported quantity"
            )


def test_every_training_set_keeps_both_gold_classes_per_dataset(frame):
    """What threshold selection actually needs, and it is weaker than it looks.

    The training set for each fold is the other four folds, and that is where the
    threshold is chosen, so every dataset there must still have both gold classes or it
    drops out of the training objective. That holds.

    The held-out fold is a different matter. RAGTruth carries four negatives in this
    495-item subsample, so with five folds at least one fold gets none of them. That is
    exactly why the cross-fitted cascade is scored over the assembled out-of-fold
    predictions instead of fold by fold, and test_single_fold_can_lose_a_class below
    pins the fact down rather than leaving it as folklore.
    """
    rng = np.random.default_rng(SEED + 1)
    ds, gold = frame.ds.to_numpy(), frame.gold.to_numpy()
    for repeat in range(50):
        fold = PA.stratified_folds(frame, PA.K_FOLDS, rng)
        for k in range(PA.K_FOLDS):
            train = fold != k
            for d in np.unique(ds):
                classes = set(gold[train & (ds == d)].tolist())
                assert classes == {0, 1}, (
                    f"repeat {repeat}, training set for fold {k}: dataset {d} holds only "
                    f"{classes}, so it silently leaves the threshold-selection objective"
                )


def test_single_fold_can_lose_a_class_which_is_why_scoring_is_assembled(frame):
    """A held-out fold really can lose a minority class on this data.

    If this ever stops being true the scoring could be simplified, so the test asserts
    the awkward fact rather than the convenient one. It also pins the cause: the
    thinnest dataset has fewer examples of its minority class than there are folds.
    """
    counts = frame.groupby("ds").gold.agg(["sum", "size"])
    thinnest = min(min(int(r["sum"]), int(r["size"] - r["sum"])) for _, r in counts.iterrows())
    assert thinnest < PA.K_FOLDS, (
        f"the thinnest minority class now holds {thinnest} examples, at least K_FOLDS="
        f"{PA.K_FOLDS}: per-fold scoring would now be defined, so revisit the assembled "
        "out-of-fold scoring and this test"
    )


def test_folds_partition_the_frame(frame):
    """No example is scored twice and none is dropped: each gets exactly one
    out-of-fold prediction, which is what lets the assembled 495 be scored as a whole."""
    rng = np.random.default_rng(SEED + 2)
    fold = PA.stratified_folds(frame, PA.K_FOLDS, rng)
    assert len(fold) == len(frame)
    counts = np.bincount(fold, minlength=PA.K_FOLDS)
    assert counts.sum() == len(frame)
    # round-robin inside each stratum keeps the folds within one example of each other
    assert counts.max() - counts.min() <= N_DATASETS * 2


def test_out_of_sample_does_not_exceed_in_sample(frame):
    """Selection optimism is non-negative on this data.

    A reduced repeat count keeps the test quick; the reported figure comes from the
    full 200 repeats in paired_analysis.main().
    """
    gold, ds = frame.gold.to_numpy(), frame.ds.to_numpy()
    in_sample = []
    for t in PA.THRESHOLDS:
        pred, _ = PA.cascade_pred(frame, t)
        in_sample.append(PA.macro_bacc_arr(gold, pred, ds))
    best_in = max(in_sample)

    xf = PA.crossfit_cascade(
        frame, lambda conf: 0.0, n_repeats=20, k_folds=PA.K_FOLDS, seed=SEED
    )
    oos, picked = xf.bacc, xf.picked
    assert len(picked) == 20 * PA.K_FOLDS
    # every claim carries exactly one out-of-fold prediction per repeat, which is what
    # the cascade-against-Astra bootstrap and McNemar in section 9 are computed over
    assert xf.oof_preds.shape == (20, len(frame))
    assert set(np.unique(xf.oof_preds)) <= {0, 1}
    assert oos.mean() <= best_in + 1e-12, (
        f"cross-fitted {oos.mean():.4f} exceeds in-sample best {best_in:.4f}: the "
        "threshold fit is leaking into the held-out fold"
    )
    # and it should not exceed it on any single repeat either, since the in-sample best
    # is the maximum over the same threshold grid evaluated on the same 495 examples
    assert oos.max() <= best_in + 1e-12


def test_crossfit_never_picks_a_threshold_outside_the_grid(frame):
    picked = PA.crossfit_cascade(
        frame, lambda conf: 0.0, n_repeats=10, k_folds=PA.K_FOLDS, seed=SEED + 3
    ).picked
    assert set(picked) <= set(PA.THRESHOLDS)


def test_crossfit_guard_metrics_are_out_of_fold_and_bracket_the_two_systems(frame):
    """The cascade's false-verification rate has to be cross-fitted too.

    Table 1 prints it beside the cross-fitted accuracy, so it must come out of the same
    out-of-fold predictions and not from a threshold fitted on the whole sample. Two
    properties pin it down: it is computed from the same assembled predictions as the
    accuracy in the same repeat, and because every claim is answered by either Jev or
    Astra it cannot fall outside the range the two systems set on their own.
    """
    gold = frame.gold.to_numpy().tolist()
    jev_fvr = M.false_verification_rate(gold, frame.jev.tolist())
    astra_fvr = M.false_verification_rate(gold, frame.astra.tolist())

    xf = PA.crossfit_cascade(
        frame, lambda conf: 0.0, n_repeats=20, k_folds=PA.K_FOLDS, seed=SEED + 4
    )
    fvr, vp = xf.fvr, xf.verified_precision
    assert len(fvr) == 20 and len(vp) == 20
    lo, hi = min(jev_fvr, astra_fvr), max(jev_fvr, astra_fvr)
    assert lo - 1e-12 <= fvr.min() and fvr.max() <= hi + 1e-12, (
        f"cross-fitted false-verification rates {fvr.min():.4f} to {fvr.max():.4f} fall "
        f"outside the range set by Jev ({jev_fvr:.4f}) and Astra ({astra_fvr:.4f})"
    )
    # and the composition is real routing, not one system in disguise
    assert fvr.mean() > min(jev_fvr, astra_fvr) + 1e-6


def test_fast_balanced_accuracy_matches_src_metrics(frame):
    """The loops use a numpy shortcut; src/metrics.py is the cross-checked definition."""
    PA.assert_fast_path_matches_metrics(frame, ["jev", "astra", "astra_high"])

#!/usr/bin/env python3
"""Every number in the arXiv paper, produced by one command.

    docker compose run --rm dev uv run python scripts/verify/paired_analysis.py

This file is the single source of the paper's evidence (ADR-006, CLAUDE.md rule 4).
It absorbs what used to live in three scripts:

  * the paired Jev / GPT-6 Astra comparison (this file, previously)
  * scripts/verify/crossfit_cascade.py   cross-fitted cascade and its controls
  * scripts/verify/minicheck_arm.py      the MiniCheck arm, scored and excluded

Both of those are now thin shims that call `main()` here, so there is exactly one
place a number can come from and exactly one command to regenerate all of them.

Outputs
-------
  arxiv/generated/numbers.json   everything, machine readable
  arxiv/generated/numbers.tex    one \\newcommand per reported value
  arxiv/generated/crossfit.json  the cascade view (unchanged schema, kept for links)
  arxiv/generated/minicheck_arm.json  the excluded arm (unchanged schema)
  medium/generated/numbers.json  the published article's evidence (schema frozen)
  arxiv/evidence/raw_hashes.json sha256 of every prediction file read

Rules this file enforces rather than assumes
--------------------------------------------
  * Calibration primitives come from `src.metrics`, which carries no sklearn and is
    cross-checked against sklearn and scipy to 1e-12 by tests/test_metrics_crosscheck.py.
    Nothing here reimplements Brier, NLL, ECE, AUROC or AUPRC. The fast numpy path
    used inside the bootstrap and cross-fitting loops is asserted equal to
    `src.metrics.balanced_accuracy` before any of it is reported.
  * Prices come from the hashed pricing snapshot, never from a literal in this file.
  * The log-loss clipping epsilon comes from configs/benchmark.yaml, not from a default.
  * MiniCheck is an EXCLUDED arm. Its numbers are computed so section 12 can disclose
    the attempt with evidence; every key that carries them says so.
  * No latency ratio is emitted. The bulk runs used concurrency 12, so per-request
    latency includes client-side queueing and is not a controlled measurement
    (CLAUDE.md rule 10, spec section 21).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.contingency_tables import mcnemar

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src import metrics as M  # noqa: E402
from src.data.load_aggrefact import load_dev_frame  # noqa: E402

# All runs below are spec section 28/29 compliant and use the P2b prompts (both models
# name entity errors). Superseded runs stay on disk and are never read here:
#   paired-astra-vs-jev-20260921      v1, discarded: our 16-token cap caused 84 failures
#   paired-astra-vs-jev-20260921-v2   pre-P2b prompt
#   paired-astra-highthink-20260921   pre-P2b prompt
#   runs/explore-jev-dev              exploratory Jev, no manifest, reduced schema
RUN_FILES = OrderedDict(
    jev=REPO / "runs/paired-jev-20260921/predictions_jev.jsonl",
    astra=REPO / "runs/paired-astra-lowthink-20260921-p2b/predictions_gpt6_astra.jsonl",
    astra_high=REPO / "runs/paired-astra-highthink-20260921-p2b/predictions_gpt6_astra.jsonl",
    minicheck=REPO / "runs/paired-minicheck-20260922/predictions_minicheck.jsonl",
)

ARXIV_JSON = REPO / "arxiv/generated/numbers.json"
ARXIV_TEX = REPO / "arxiv/generated/numbers.tex"
CROSSFIT_JSON = REPO / "arxiv/generated/crossfit.json"
MINICHECK_JSON = REPO / "arxiv/generated/minicheck_arm.json"
MEDIUM_JSON = REPO / "medium/generated/numbers.json"
ARXIV_HASHES_TEX = REPO / "arxiv/generated/hashes.tex"
EVIDENCE_JSON = REPO / "arxiv/evidence/raw_hashes.json"
INPUTS_JSON = REPO / "arxiv/evidence/input_hashes.json"

# The captured LLM-AggreFact leaderboard, which holds the published MiniCheck score our
# port was supposed to reproduce (threats-to-validity C4). Hashed with the other evidence.
LEADERBOARD_MANIFEST = "data/manifests/llm_aggrefact_published_leaderboard.json"

# The example-ID list that defines the evaluation sample. It predates the gated pipeline
# and no script in this repository regenerates the draw, so it is released and hashed as
# data rather than described as reproducible (section 12 says so in those words).
SAMPLE_ID_LIST = "runs/explore-jev-dev/jev_dev_sample.jsonl"

# The superseded frontier run behind every PromptParity* number in section 12. It is not
# part of the headline lock (raw_hashes.json is stamped and must not move), so it is
# hashed with the other auxiliary inputs and the paper says which file its numbers use.
PRE_PARITY_ASTRA = "runs/paired-astra-vs-jev-20260921-v2/predictions_gpt6_astra.jsonl"

# The first pass of that same configuration, discarded because our completion-token cap
# was consumed by the model's internal reasoning on 84 claims (appendix C). It carries the
# identical prompt hash, reasoning effort and model as PRE_PARITY_ASTRA and differs only
# in that cap, so the two together are the only repeat of the frontier arm this study has
# under an UNCHANGED prompt. Section 12 used to assert that no such repeat existed, which
# the released incident log contradicts, so the comparison is computed here instead.
CAPPED_ASTRA = "runs/paired-astra-vs-jev-20260921/predictions_gpt6_astra.jsonl"

# The SAME prompt edit, applied to the high-effort arm. One edit was made and both frontier
# configurations were re-run under it, so there are two pre-correction runs and not one.
# Only the low-effort pair was measured for a long time, and the sentence that grew out of
# that measurement ("no flip landed on a gold-unsupported claim, so the separated results
# are undisturbed") is false on this arm: one flip did, and it moves this arm's
# false-verification rate, which is a leg of one of the two Holm survivors. Measuring both
# arms is what the paper can defend; measuring one and generalising is not.
PRE_PARITY_ASTRA_HIGH = "runs/paired-astra-highthink-20260921/predictions_gpt6_astra.jsonl"

# Readable macro stems for the leaderboard rows. Derived names would come out as
# \PublishedgptFouroTwoZeroTwoFourZeroFiveOneThreeTestBAcc, which no one will type.
# A row present in the manifest and absent here fails loudly rather than silently.
_LEADERBOARD_MACRO = {
    "Bespoke-MiniCheck-7B": "BespokeMiniCheck",
    "Claude-3.5-Sonnet": "ClaudeThreeFiveSonnet",
    "gpt-4o-2024-05-13": "GptFourO",
    "MiniCheck-Flan-T5-L": "MiniCheckFlanTFiveL",
}

THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99)
N_REPEATS = 200
K_FOLDS = 5

# No digits in this string. Every quantity it refers to is computed and lives in
# configurations.minicheck, and the paper cites the macros, not this prose (rule 4).
MINICHECK_EXCLUSION = (
    "EXCLUDED as a comparison arm. Our port scores far below the other configurations, "
    "and the gap is not a threshold artifact: the best operating point over the whole "
    "decision-threshold sweep barely moves it (see minicheck_threshold_sweep), its AUROC "
    "is weak, and it calls SUPPORTED on far fewer claims than the gold rate. The scores do "
    "not separate, which points at our port (chunking or evidence truncation) rather than "
    "at the published model, so we do not publish a number we cannot stand behind. These "
    "values exist so section 12 can disclose the attempt with evidence. They are not a "
    "MiniCheck result and must never be presented as one."
)


# --------------------------------------------------------------------------------
# fast paths, each asserted against src.metrics before anything is reported
# --------------------------------------------------------------------------------
def bacc_fast(gold: np.ndarray, pred: np.ndarray):
    """Balanced accuracy, or None when a class is absent.

    A single-class group is never silently reweighted into a number.
    """
    pos = gold == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return None
    return 0.5 * (
        float((pred[pos] == 1).sum()) / npos + float((pred[~pos] == 0).sum()) / nneg
    )


def macro_bacc_arr(gold: np.ndarray, pred: np.ndarray, ds: np.ndarray):
    """Mean of per-dataset balanced accuracy: the LLM-AggreFact headline convention.

    Pooling instead would let RAGTruth, half of the dev split, decide the headline.
    """
    vals = []
    for d in np.unique(ds):
        m = ds == d
        b = bacc_fast(gold[m], pred[m])
        if b is not None:
            vals.append(b)
    return float(np.mean(vals)) if vals else None


def per_dataset_bacc(df: pd.DataFrame, col: str) -> dict[str, float]:
    """Reported per-dataset balanced accuracy, computed by src.metrics."""
    out = {}
    for d, t in df.groupby("ds"):
        gold, pred = t.gold.tolist(), t[col].tolist()
        if len(set(gold)) < 2:
            raise ValueError(f"dataset {d!r} has a single gold class; balanced accuracy is undefined")
        out[d] = M.balanced_accuracy(gold, pred)
    return out


def macro_bacc(df: pd.DataFrame, col: str) -> float:
    return M.mean_balanced_accuracy(per_dataset_bacc(df, col))


def assert_fast_path_matches_metrics(df: pd.DataFrame, cols) -> None:
    """The loops below cannot call src.metrics ten thousand times, so prove once that
    the numpy shortcut returns exactly what the cross-checked implementation returns."""
    for col in cols:
        for d, t in df.groupby("ds"):
            a = bacc_fast(t.gold.to_numpy(), t[col].to_numpy())
            b = M.balanced_accuracy(t.gold.tolist(), t[col].tolist())
            if a is None or abs(a - b) > 1e-12:
                raise AssertionError(
                    f"fast balanced accuracy disagrees with src.metrics on {col}/{d}: {a} vs {b}"
                )


# --------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------
def read_jsonl(path: Path) -> dict[str, dict]:
    """Append-only prediction files: a later row for the same example wins."""
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["example_id"]] = r
    return out


def write_if_changed(path: Path, text: str) -> bool:
    """Leave the file alone when the bytes would be identical.

    scripts/verify/check_article_numbers.py fails any figure older than
    medium/generated/numbers.json, which is the right check: a figure rendered before
    the data changed is a lie. But rewriting an unchanged file would bump its mtime and
    trip that check for nothing, so an unchanged output is left untouched and the gate
    keeps meaning what it says.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() == text:
        return False
    path.write_text(text)
    return True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reasoning_tokens(row: dict, eid: str) -> int:
    """Provider-reported internal reasoning tokens for one call.

    Refused rather than defaulted, because a missing field and a genuine zero are
    different facts and section 4 reports how often the count was genuinely zero. That
    count is the evidence behind the answer to "you crippled the frontier model"
    (threats-to-validity B1): the name of the effort setting says what we asked for, and
    this is a measurement of what came back.
    """
    extra = row.get("extra") or {}
    if extra.get("reasoning_tokens") is None:
        raise SystemExit(
            f"REFUSING to write numbers.json: {eid} carries no extra.reasoning_tokens, "
            "and section 4 reports that distribution rather than the name of the knob."
        )
    return int(extra["reasoning_tokens"])


def build_frame():
    """The paired frame, plus the bookkeeping the report needs about what was dropped.

    The headline frame is defined by Jev and Astra both returning a usable answer, which
    is what the published article scored. The high-effort ablation and the excluded
    MiniCheck arm are joined onto it; neither is allowed to shrink it.
    """
    dev = load_dev_frame()
    agg = dev.labels.groupby("example_id")["label"].agg(["nunique", "first"])
    # errata E18: a few (doc, claim) pairs appear twice in dev, some with conflicting
    # labels. Those are unscoreable, so they are excluded rather than last-wins.
    ambiguous = set(agg.index[agg["nunique"] > 1])
    gold = {e: int(v) for e, (nu, v) in agg[["nunique", "first"]].iterrows() if nu == 1}

    loaded = {k: read_jsonl(p) for k, p in RUN_FILES.items()}
    jev, astra = loaded["jev"], loaded["astra"]
    astra_high, minicheck = loaded["astra_high"], loaded["minicheck"]

    rows = []
    for eid, j in jev.items():
        a = astra.get(eid)
        if a is None or j["status"] != "ok" or a["status"] != "ok":
            continue
        if eid in ambiguous:
            continue
        if eid not in gold:
            raise KeyError(f"{eid} is absent from dev gold: wrong split or a stale run?")
        rows.append(
            dict(
                eid=eid,
                ds=j["dataset"],
                gold=gold[eid],
                jev=1 if j["label"] == "SUPPORTED" else 0,
                astra=1 if a["label"] == "SUPPORTED" else 0,
                jev_p=float(j["p_supported"]),
                jev_ms=float(j["latency_ms"]),
                astra_ms=float(a["latency_ms"]),
                jev_in=j["input_tokens"] or 0,
                jev_out=j["output_tokens"] or 0,
                astra_in=a["input_tokens"] or 0,
                astra_out=a["output_tokens"] or 0,
                astra_reason=reasoning_tokens(a, eid),
            )
        )
    df = pd.DataFrame(rows).reset_index(drop=True)  # positional index == label, for the cascade

    # errata E21: the high-effort ablation must cover the same items, or it is not an
    # ablation. Missing or failed rows are a refusal, never a quietly smaller sample.
    missing = [e for e in df.eid if e not in astra_high]
    if missing:
        raise SystemExit(
            f"REFUSING to write numbers.json: the high-effort ablation covers "
            f"{len(df) - len(missing)}/{len(df)} examples; errata E21 needs the same items."
        )
    bad = [e for e in df.eid if astra_high[e]["status"] != "ok"]
    if bad:
        raise SystemExit(f"REFUSING to write numbers.json: {len(bad)} high-effort rows are not ok.")
    df["astra_high"] = [1 if astra_high[e]["label"] == "SUPPORTED" else 0 for e in df.eid]
    df["astra_high_ms"] = [float(astra_high[e]["latency_ms"]) for e in df.eid]
    df["astra_high_in"] = [astra_high[e]["input_tokens"] or 0 for e in df.eid]
    df["astra_high_out"] = [astra_high[e]["output_tokens"] or 0 for e in df.eid]
    df["astra_high_reason"] = [reasoning_tokens(astra_high[e], e) for e in df.eid]

    # MiniCheck is excluded from the paper's arms, so a hole in it degrades its own
    # disclosure rather than blocking the paper.
    mc_missing = [e for e in df.eid if e not in minicheck or minicheck[e].get("status") != "ok"]
    minicheck_complete = not mc_missing
    if minicheck_complete:
        df["minicheck"] = [1 if minicheck[e]["label"] == "SUPPORTED" else 0 for e in df.eid]
        df["minicheck_p"] = [float(minicheck[e]["p"]) for e in df.eid]
        df["minicheck_ms"] = [float(minicheck[e]["ms"]) for e in df.eid]

    meta = dict(
        dev=dev,
        gold=gold,
        ambiguous=ambiguous,
        loaded=loaded,
        minicheck_complete=minicheck_complete,
        minicheck_missing=len(mc_missing),
    )
    return df, meta


# --------------------------------------------------------------------------------
# paired bootstrap
# --------------------------------------------------------------------------------
def paired_bootstrap(df: pd.DataFrame, cols, n_resamples: int, seed: int) -> dict[str, np.ndarray]:
    """One stratified resample scores every configuration, so differences stay paired.

    Resampling is within dataset (spec section 19). A resample that leaves a dataset
    single-class drops that dataset from that iteration's macro average, for every
    configuration alike, rather than inventing a balanced accuracy for it.
    """
    rng = np.random.default_rng(seed)
    groups = []
    for _, t in df.groupby("ds"):
        gold = t.gold.to_numpy()
        correct = {c: (t[c].to_numpy() == gold) for c in cols}
        groups.append((gold, correct, len(t)))

    draws = {c: np.empty(n_resamples) for c in cols}
    acc = {c: [] for c in cols}
    for i in range(n_resamples):
        for c in cols:
            acc[c].clear()
        for gold, correct, n in groups:
            idx = rng.integers(0, n, n)
            gg = gold[idx]
            pos = gg == 1
            npos = int(pos.sum())
            nneg = n - npos
            if npos == 0 or nneg == 0:
                continue
            neg = ~pos
            for c in cols:
                cc = correct[c][idx]
                acc[c].append(0.5 * (cc[pos].sum() / npos + cc[neg].sum() / nneg))
        for c in cols:
            draws[c][i] = float(np.mean(acc[c]))
    return draws


def pct_ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def fvr_gap_inference(df: pd.DataFrame, a: str, b: str, n_resamples: int, seed: int) -> dict:
    """Interval and test for the false-verification gap, on the same footing as the
    accuracy gap.

    The paper argues that false verification is the number an operator buys on, so it
    cannot be the one quantity reported as a bare point estimate. The estimand is the
    pooled rate over gold-unsupported claims, so the bootstrap resamples those claims
    within dataset, and the test is the exact McNemar on the claims where exactly one
    system waves the unsupported claim through.
    """
    neg = df[df.gold == 0]
    a_fv = neg[a].to_numpy() == 1
    b_fv = neg[b].to_numpy() == 1
    a_only, b_only = int((a_fv & ~b_fv).sum()), int((b_fv & ~a_fv).sum())
    both, neither = int((a_fv & b_fv).sum()), int((~a_fv & ~b_fv).sum())
    mc = mcnemar([[both, a_only], [b_only, neither]], exact=True)

    rng = np.random.default_rng(seed)
    groups = [(t[a].to_numpy() == 1, t[b].to_numpy() == 1, len(t)) for _, t in neg.groupby("ds")]
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        na = nb = n = 0
        for av, bv, m in groups:
            idx = rng.integers(0, m, m)
            na += int(av[idx].sum())
            nb += int(bv[idx].sum())
            n += m
        draws[i] = (na - nb) / n
    lo, hi = pct_ci(draws)
    return dict(
        n_gold_unsupported=int(len(neg)),
        a_only_false_verifies=a_only, b_only_false_verifies=b_only,
        both_false_verify=both, neither_false_verifies=neither,
        gap_pp=float((a_fv.mean() - b_fv.mean()) * 100),
        gap_boot_mean_pp=float(draws.mean() * 100),
        gap_ci_pp=[lo * 100, hi * 100],
        mcnemar_p=float(mc.pvalue), mcnemar_statistic=float(mc.statistic),
        note="paired stratified bootstrap over the gold-unsupported claims and an exact "
             "McNemar on the discordant false-verification pairs; the estimand is the "
             "pooled rate, which is what the configurations report",
    )


# --------------------------------------------------------------------------------
# cascade
# --------------------------------------------------------------------------------
def cascade_conf(df: pd.DataFrame, t: float) -> np.ndarray:
    """Jev decides when its confidence max(p, 1-p) reaches the threshold.

    Written this way on purpose. The algebraically identical `(p >= t) | (p <= 1 - t)`
    is not identical in binary floating point: 1 - 0.8 evaluates to
    0.19999999999999996, so a prediction at p = 0.2 exactly fails `p <= 1 - t` and gets
    escalated, while max(0.2, 0.8) >= 0.8 holds and keeps it. One example in this sample
    sits on that boundary, which was enough to make the escalation rate and the cascade
    cost printed in the paper unreproducible from the rule the paper states. The paper's
    rule is the one a reimplementer would write, so the code follows the paper rather
    than the reverse (docs/reference/incident-log.md, 2026-09-23).
    """
    p = df.jev_p.to_numpy()
    return np.maximum(p, 1.0 - p) >= t


def cascade_pred(df: pd.DataFrame, t: float):
    conf = cascade_conf(df, t)
    return np.where(conf, df.jev.to_numpy(), df.astra.to_numpy()), conf


def stratified_folds(df: pd.DataFrame, k: int, rng: np.random.Generator) -> np.ndarray:
    """Fold assignment stratified by (dataset, gold) so every fold keeps the mix.

    Round-robin inside each shuffled stratum, which is what makes every fold contain
    every dataset (tests/test_crossfit.py holds that property).
    """
    fold = np.empty(len(df), dtype=int)
    for _, idx in df.groupby(["ds", "gold"]).groups.items():
        pos = np.asarray(idx)
        rng.shuffle(pos)
        fold[pos] = np.arange(len(pos)) % k
    return fold


class CrossfitResult(NamedTuple):
    """What one cross-fitting run produced, named rather than positional.

    `oof_preds` is the field the rest of this file grew around: one assembled
    out-of-fold prediction per claim per repeat, which is what lets the cascade be
    compared with Astra using the same bootstrap and the same test as any other pair in
    the paper. Without it the only bracket the cascade carries is the spread over
    repeats, which says nothing about the 495 claims being a sample.
    """

    bacc: np.ndarray
    escalated: np.ndarray
    cost: np.ndarray
    vs_astra_pp: np.ndarray
    picked: list
    fvr: np.ndarray
    verified_precision: np.ndarray
    oof_preds: np.ndarray


def crossfit_cascade(df, cost_per_1k, n_repeats=N_REPEATS, k_folds=K_FOLDS, seed=None,
                     thresholds=THRESHOLDS):
    """Choose the escalation threshold on other folds, apply it to the held-out fold.

    Every example ends with exactly one out-of-fold prediction, and those are scored
    over the whole assembled set rather than per fold: a single fold has too few
    negatives in some datasets for macro balanced accuracy to be defined at all.
    """
    rng = np.random.default_rng(seed)
    gold_a, ds_a = df.gold.to_numpy(), df.ds.to_numpy()
    astra_a = df.astra.to_numpy()
    oos_bacc, oos_esc, oos_cost, oos_vs_astra, picked = [], [], [], [], []
    oos_fvr, oos_vp, oof_preds = [], [], []
    astra_bacc = macro_bacc_arr(gold_a, astra_a, ds_a)
    for _ in range(n_repeats):
        fold = stratified_folds(df, k_folds, rng)
        pred = np.empty(len(df), dtype=int)
        conf = np.empty(len(df), dtype=bool)
        for k in range(k_folds):
            tr, te = df.index[fold != k], df.index[fold == k]
            tr_df = df.loc[tr]
            tr_gold, tr_ds = tr_df.gold.to_numpy(), tr_df.ds.to_numpy()
            scored = []
            for t in thresholds:
                p, _ = cascade_pred(tr_df, t)
                b = macro_bacc_arr(tr_gold, p, tr_ds)
                if b is not None:
                    scored.append((b, t))
            t_star = max(scored)[1]
            picked.append(t_star)
            p_all, c_all = cascade_pred(df, t_star)
            pred[te] = p_all[te]
            conf[te] = c_all[te]
        b = macro_bacc_arr(gold_a, pred, ds_a)
        oos_bacc.append(b)
        oos_esc.append(float((~conf).mean()))
        oos_cost.append(cost_per_1k(conf))
        oos_vs_astra.append((b - astra_bacc) * 100)
        # The guard metric has to survive cross-fitting too. A cross-fitted accuracy
        # printed beside an in-sample false-verification rate would be the same
        # selection bug one column to the right, so both leave this loop together.
        oos_fvr.append(M.false_verification_rate(gold_a.tolist(), pred.tolist()))
        oos_vp.append(M.verified_precision(gold_a.tolist(), pred.tolist()))
        oof_preds.append(pred.copy())
    return CrossfitResult(
        bacc=np.array(oos_bacc), escalated=np.array(oos_esc), cost=np.array(oos_cost),
        vs_astra_pp=np.array(oos_vs_astra), picked=picked, fvr=np.array(oos_fvr),
        verified_precision=np.array(oos_vp), oof_preds=np.array(oof_preds),
    )


def cascade_vs_arm_inference(df, oof_preds: np.ndarray, n_resamples: int, seed: int,
                             arm: str = "astra") -> dict:
    """Sampling uncertainty and a test for the cascade against one single-system arm.

    The cross-fitting loop reports a spread over repeats. That spread carries fold
    assignment and threshold selection and says nothing about the claims being a sample,
    which is why the paper's own caption refuses to call it a confidence interval. A
    reader told the cascade cannot be separated from a comparator needs the other kind of
    uncertainty, so this resamples claims within dataset exactly as the single-system
    bootstrap does, cycling through the assembled out-of-fold prediction vectors so that
    fold-assignment noise is carried too rather than conditioned away.

    The test is the same exact McNemar the rest of the paper uses, on raw per-item
    accuracy. It is run once per repeat, because there is one assembled prediction
    vector per repeat and no single canonical one, and the range over repeats is
    reported alongside the median so that the answer does not depend on which repeat we
    chose to show.

    `arm` is a column of the paired frame. It is run against the frontier arm, which is
    the comparison the cost claim rests on, and against Jev alone, which is the
    comparison a reader has to see before believing a cascade is worth building: the
    cascade costs many times what Jev alone costs, so "is it better than Jev" is the
    unflattering question and it gets the same machinery as the flattering one.
    """
    gold, astra = df.gold.to_numpy(), df[arm].to_numpy()
    ds_a = df.ds.to_numpy()
    n_repeats = len(oof_preds)

    groups = []
    for _, t in df.groupby("ds"):
        pos = np.asarray(t.index)
        groups.append((gold[pos], astra[pos], pos, len(pos)))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        cas = oof_preds[i % n_repeats]
        acc_c, acc_a = [], []
        for g, a, pos, n in groups:
            idx = rng.integers(0, n, n)
            gg = g[idx]
            posm = gg == 1
            npos = int(posm.sum())
            nneg = n - npos
            if npos == 0 or nneg == 0:
                continue
            negm = ~posm
            cc = cas[pos][idx] == gg
            aa = a[idx] == gg
            acc_c.append(0.5 * (cc[posm].sum() / npos + cc[negm].sum() / nneg))
            acc_a.append(0.5 * (aa[posm].sum() / npos + aa[negm].sum() / nneg))
        draws[i] = float(np.mean(acc_c)) - float(np.mean(acc_a))
    lo, hi = pct_ci(draws)

    ps, stats, discordant, sig_favouring_cascade = [], [], [], 0
    mcnemar_favours_cascade = mcnemar_favours_arm = 0
    for cas in oof_preds:
        cc, ca = cas == gold, astra == gold
        a_only, b_only = int((cc & ~ca).sum()), int((ca & ~cc).sum())
        both, neither = int((cc & ca).sum()), int((~cc & ~ca).sum())
        mc = mcnemar([[both, a_only], [b_only, neither]], exact=True)
        ps.append(float(mc.pvalue))
        stats.append(float(mc.statistic))
        discordant.append(a_only + b_only)
        if a_only > b_only:
            mcnemar_favours_cascade += 1
        elif b_only > a_only:
            mcnemar_favours_arm += 1
        # Which way a separating repeat separates. A count of significant repeats with no
        # sign attached would let a reader assume the flattering direction.
        if mc.pvalue < 0.05 and a_only > b_only:
            sig_favouring_cascade += 1

    deltas = np.array([macro_bacc_arr(gold, cas, ds_a) for cas in oof_preds]) \
        - macro_bacc_arr(gold, astra, ds_a)

    # The SAME machinery on the guard metric. Section 9.3 argues that a claim two systems
    # differ, or do not, needs an interval and a test; false verification is the number
    # this paper tells operators to buy on, so it cannot be the one comparison carried as
    # a bare point estimate. Estimand and test match fvr_gap_inference exactly: a paired
    # stratified bootstrap over the gold-unsupported claims, and an exact McNemar on the
    # claims where exactly one of the two waves an unsupported claim through, once per
    # cross-fitting repeat.
    negm_all = gold == 0
    neg_pos = np.flatnonzero(negm_all)
    neg_ds = ds_a[neg_pos]
    arm_fv_all = astra[neg_pos] == 1
    neg_groups = []
    for d in np.unique(neg_ds):
        sel = np.flatnonzero(neg_ds == d)
        neg_groups.append((neg_pos[sel], arm_fv_all[sel], len(sel)))
    rng_fv = np.random.default_rng(seed + 1)
    fvr_draws = np.empty(n_resamples)
    for i in range(n_resamples):
        cas_fv = oof_preds[i % n_repeats][neg_pos] == 1
        num_c = num_a = den = 0
        for pos, a_fv, n in neg_groups:
            idx = rng_fv.integers(0, n, n)
            local = np.searchsorted(neg_pos, pos)
            num_c += int(cas_fv[local][idx].sum())
            num_a += int(a_fv[idx].sum())
            den += n
        fvr_draws[i] = (num_c - num_a) / den
    fvr_lo, fvr_hi = pct_ci(fvr_draws)

    fvr_ps, fvr_disc, fvr_favours_cascade, fvr_sig_favours_cascade = [], [], 0, 0
    for cas in oof_preds:
        c_fv = cas[neg_pos] == 1
        a_only = int((c_fv & ~arm_fv_all).sum())
        b_only = int((arm_fv_all & ~c_fv).sum())
        both = int((c_fv & arm_fv_all).sum())
        neither = int((~c_fv & ~arm_fv_all).sum())
        mcf = mcnemar([[both, a_only], [b_only, neither]], exact=True)
        fvr_ps.append(float(mcf.pvalue))
        fvr_disc.append(a_only + b_only)
        # "Favours the cascade" on this metric means the cascade false-verifies LESS,
        # which is b_only > a_only. The opposite sense from the accuracy block above, and
        # getting it backwards would invert the paper's guard claim.
        if b_only > a_only:
            fvr_favours_cascade += 1
            if mcf.pvalue < 0.05:
                fvr_sig_favours_cascade += 1
    fvr_deltas = np.array([
        float((cas[neg_pos] == 1).mean()) for cas in oof_preds]) - float(arm_fv_all.mean())

    fvr_inference = dict(
        n_gold_unsupported=int(len(neg_pos)),
        fvr_delta_pp=float(fvr_deltas.mean() * 100),
        fvr_delta_boot_mean_pp=float(fvr_draws.mean() * 100),
        fvr_delta_ci_pp=[fvr_lo * 100, fvr_hi * 100],
        mcnemar_p_median=float(np.median(fvr_ps)),
        mcnemar_p_min=float(min(fvr_ps)),
        mcnemar_p_max=float(max(fvr_ps)),
        n_discordant_median=float(np.median(fvr_disc)),
        n_repeats_p_below_05=int(sum(1 for p in fvr_ps if p < 0.05)),
        n_repeats_favouring_cascade=int(fvr_favours_cascade),
        n_repeats_p_below_05_favouring_cascade=int(fvr_sig_favours_cascade),
        note="paired stratified bootstrap over the gold-unsupported claims and an exact "
             "McNemar on the discordant false-verification pairs, one test per "
             "cross-fitting repeat. Negative fvr_delta_pp means the cascade false-verifies "
             "LESS than the comparison arm. Same estimand as complementarity."
             "fvr_gap_inference, which is the pooled rate.",
    )
    return dict(
        fvr_inference=fvr_inference,
        n_repeats=n_repeats,
        n_resamples=n_resamples,
        macro_bacc_delta_pp=float(deltas.mean() * 100),
        macro_bacc_delta_boot_mean_pp=float(draws.mean() * 100),
        macro_bacc_delta_ci_pp=[lo * 100, hi * 100],
        mcnemar_p_median=float(np.median(ps)),
        mcnemar_p_min=float(min(ps)),
        mcnemar_p_max=float(max(ps)),
        mcnemar_statistic_median=float(np.median(stats)),
        n_discordant_median=float(np.median(discordant)),
        # The median p-value alone would let a reader think no repeat separated the two.
        # Some do. Reporting the share that clear 0.05 is the difference between "the
        # comparison does not separate" and "the comparison does not separate at the
        # median, and here is how often the draw of folds says otherwise".
        n_repeats_p_below_05=int(sum(1 for p in ps if p < 0.05)),
        frac_repeats_p_below_05=float(sum(1 for p in ps if p < 0.05) / len(ps)),
        n_repeats_p_below_05_favouring_cascade=int(sig_favouring_cascade),
        # Two different "directions", and conflating them is how a paper ends up claiming
        # its point estimate never reverses when its own macro metric reverses in a third
        # of the repeats. Both counts are emitted so the prose has to name which it means.
        n_repeats_mcnemar_favours_cascade=int(mcnemar_favours_cascade),
        n_repeats_mcnemar_favours_arm=int(mcnemar_favours_arm),
        n_repeats_macro_below_arm=int((deltas < 0).sum()),
        frac_bootstrap_draws_negative=float((draws < 0).mean()),
        direction_note="n_repeats_mcnemar_favours_* counts the sign of the discordant split "
                       "per repeat (ties are in neither count); n_repeats_macro_below_arm "
                       "counts repeats whose MACRO balanced accuracy is below the arm. The "
                       "two answer different questions and a sentence about 'direction' has "
                       "to say which one it means.",
        interval_note="paired stratified bootstrap over the 495 claims, cycling through the "
                      "assembled out-of-fold cascade predictions so the interval carries both "
                      "sampling and fold-assignment noise. THIS is a confidence interval on "
                      "the difference; cascade.crossfit.minus_astra_pp_ci is not.",
        test_note="exact McNemar on RAW per-item accuracy pooled over datasets, one test per "
                  "cross-fitting repeat, NOT on the macro-averaged balanced accuracy",
    )


def arm_column_from_run(df, path: Path) -> list[int]:
    """The paired frame's verdict column, read from a run that is not a reported arm.

    Used to substitute a superseded run into a comparison the paper reports, so that
    "the correction did not manufacture this result" is a computation rather than a hope.
    """
    rows = read_jsonl(path)
    missing = [e for e in df.eid if e not in rows or rows[e].get("status") != "ok"]
    if missing:
        raise SystemExit(
            f"REFUSING to write numbers.json: {path.name} covers "
            f"{len(df) - len(missing)}/{len(df)} of the paired claims, so it cannot be "
            "substituted into a reported comparison.")
    return [1 if rows[e]["label"] == "SUPPORTED" else 0 for e in df.eid]


def prompt_parity_sensitivity(df, path: Path, post_col: str = "astra") -> dict:
    """The one prompt variation this study actually ran, measured instead of asserted.

    Appendix C describes a pre-publication correction: the frontier prompt was missing the
    entity-error trigger the decision model's criteria carried, so both frontier
    configurations were re-run under corrected wording and every frontier number in the
    paper comes from the corrected runs. The superseded runs are tracked and released, so
    "we varied the frontier prompt once and here is what moved" is a measurement available
    at zero cost, and section 12 states it rather than saying prompt sensitivity is
    entirely unmeasured.

    This is the only place in this file that reads a superseded run, and it never feeds a
    reported configuration: it is scored against the same gold vector and the same paired
    frame as everything else, and its output lives under its own key.
    """
    rows = {r["example_id"]: r
            for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}
    eids = list(df.eid)
    missing = [e for e in eids if e not in rows or rows[e].get("status") != "ok"]
    if missing:
        raise SystemExit(
            f"REFUSING to write numbers.json: the pre-correction frontier run {path.name} "
            f"covers {len(eids) - len(missing)}/{len(eids)} of the paired claims, so "
            "section 12 cannot report what the prompt change moved on that arm."
        )
    prior = np.array([1 if rows[e]["label"] == "SUPPORTED" else 0 for e in eids])
    gold, ds_a = df.gold.to_numpy(), df.ds.to_numpy()
    post = df[post_col].to_numpy()
    jev_fvr = M.false_verification_rate(gold.tolist(), df.jev.to_numpy().tolist())
    pre_fvr = M.false_verification_rate(gold.tolist(), prior.tolist())
    post_fvr = M.false_verification_rate(gold.tolist(), post.tolist())
    return dict(
        path=str(path.relative_to(REPO)),
        sha256=sha256_file(path),
        arm=post_col,
        n=len(eids),
        n_flips=int((prior != post).sum()),
        # An unchanged false-verification rate can mean no flip touched an unsupported
        # claim, or that flips on unsupported claims cancelled. The paper says which.
        n_flips_on_gold_unsupported=int(((prior != post) & (gold == 0)).sum()),
        n_flips_on_gold_supported=int(((prior != post) & (gold == 1)).sum()),
        # Which way the flips ran. The edit added a NOT_SUPPORTED trigger, so an effect
        # attributable to the prompt should push verdicts towards NOT_SUPPORTED. Most of
        # them went the other way, which is what a second sampling draw looks like and is
        # why section 12 reports this as a bound on combined movement rather than as a
        # prompt effect.
        n_flips_to_supported=int(((prior == 0) & (post == 1)).sum()),
        n_flips_to_unsupported=int(((prior == 1) & (post == 0)).sum()),
        pre_pred_supported_rate=float(prior.mean()),
        post_pred_supported_rate=float(post.mean()),
        pre_macro_bacc=macro_bacc_arr(gold, prior, ds_a),
        post_macro_bacc=macro_bacc_arr(gold, post, ds_a),
        pre_false_verification_rate=pre_fvr,
        post_false_verification_rate=post_fvr,
        pre_fvr_gap_pp=float((jev_fvr - pre_fvr) * 100),
        post_fvr_gap_pp=float((jev_fvr - post_fvr) * 100),
        fvr_gap_shift_pp=float(((jev_fvr - post_fvr) - (jev_fvr - pre_fvr)) * 100),
        note="the pre-correction frontier run (superseded, released, never a reported "
             "configuration) scored on the same paired frame and gold vector as the "
             "corrected run. One prompt change on one arm is a single observation of "
             "prompt sensitivity, not a sensitivity study.",
    )


def same_prompt_repeat(df, capped_path: Path, repeat_path: Path,
                       reported_path: Path) -> dict:
    """The one repeat of the frontier arm under an unchanged prompt, and where the parity
    flips landed.

    Section 12 said the frontier configurations were re-run only under the corrected
    prompt and never under an unchanged one, so nothing in the paper could bound their
    run-to-run movement. That was false against the project's own released incident log,
    which records the first paired pass being repeated with a larger completion-token cap
    two minutes later. Both passes carry the same prompt hash, the same reasoning effort
    and the same model id, and differ only in that cap, so on the claims the capped pass
    answered they are a same-prompt repeat and are reported as one.

    The comparison carries two selection effects that have to travel with it, and both are
    emitted here rather than left to the prose.

    First, the capped pass produced an answer only where the answer fitted inside the cap,
    so the rows it contributes are the shorter-answer ones and it is silent on the rest.
    Those rows are also, measurably, the rows on which this configuration reports no
    internal reasoning at all: the counts below give how many of the compared rows spent
    zero reasoning tokens and how many of the excluded rows spent some. A repeat taken
    where the model does no sampled reasoning cannot bound movement where it does, and
    every prompt-parity flip is in the excluded set. So this function reports where the
    flips fell as a description of where the movement sits, and the reasoning-token split
    that stops that description being read as an attribution. An earlier draft read the
    concentration as evidence that the flips were the prompt rather than a fresh draw;
    that inference is invalid in this direction, because concentration in the rows where
    the model reasons most is also what a sampling effect looks like.

    Second, both passes carry the SUPERSEDED pre-parity prompt, not the corrected prompt
    behind every frontier number in the paper. The hash of the reported run is emitted
    beside the hash of the repeat so the paper has to say so.
    """
    def labels(path: Path):
        return {r["example_id"]: r
                for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}

    capped, repeat = labels(capped_path), labels(repeat_path)
    eids = list(df.eid)
    for name, rows in (("capped", capped), ("unchanged-prompt", repeat)):
        absent = [e for e in eids if e not in rows]
        if absent:
            raise SystemExit(
                f"REFUSING to write numbers.json: the {name} frontier run is missing "
                f"{len(absent)} of the {len(eids)} paired claims, so section 12 cannot "
                "report what moved between two passes of the same prompt."
            )
    # Compared over the rows the capped pass actually answered. A failed row reports no
    # model id and no reasoning effort, so including them would make every run disagree
    # with itself; the comparison below is restricted to the same rows for the same reason.
    ok_eids = [e for e in eids if capped[e].get("status") == "ok"]
    if not ok_eids:
        raise SystemExit("REFUSING to write numbers.json: the capped pass answered nothing.")
    for field in ("prompt_hash", "model_id_reported"):
        seen = {capped[e][field] for e in ok_eids} | {repeat[e][field] for e in ok_eids}
        if len(seen) != 1:
            raise SystemExit(
                f"REFUSING to write numbers.json: the two passes disagree on {field} "
                f"({sorted(map(str, seen))}), so they are not a same-prompt repeat and "
                "section 12 must not call them one."
            )
    efforts = {capped[e]["extra"]["reasoning_effort"] for e in ok_eids} | \
              {repeat[e]["extra"]["reasoning_effort"] for e in ok_eids}
    if len(efforts) != 1:
        raise SystemExit(
            "REFUSING to write numbers.json: the two passes disagree on reasoning_effort."
        )

    lab = lambda r: 1 if r["label"] == "SUPPORTED" else 0
    answered = np.array([capped[e]["status"] == "ok" for e in eids])
    a_pred = np.array([lab(capped[e]) if capped[e]["status"] == "ok" else -1 for e in eids])
    b_pred = np.array([lab(repeat[e]) for e in eids])
    post = df.astra.to_numpy()
    gold, ds_a = df.gold.to_numpy(), df.ds.to_numpy()

    capped_macro = macro_bacc_arr(gold[answered], a_pred[answered], ds_a[answered])
    repeat_macro = macro_bacc_arr(gold[answered], b_pred[answered], ds_a[answered])
    parity_flips = b_pred != post
    n_flips = int(parity_flips.sum())
    n_unanswered = int((~answered).sum())

    # What the repeat can and cannot bound, from the provider's own reasoning-token count.
    # A row where the model reports zero reasoning tokens has almost nothing for a fresh
    # draw to move; a row where it reports fifty has. The split below is the difference
    # between "this configuration is deterministic" and "this configuration was measured
    # where it does not think", and the second is what this repeat supports.
    reported = labels(reported_path)
    def rtok(rows, e):
        return (rows[e].get("extra") or {}).get("reasoning_tokens")
    # Partitioned on the REPORTED run, because the sentence in Appendix E is about the
    # configuration the paper reports, and Section 4 gives that run's totals. Computing it
    # on the repeat instead put 397 and 80 beside a mean taken from the reported run, and
    # 397 + 4 is 401 where Section 4 says 407. Both are emitted now, with the file each
    # came from in the key, so the two can never be spliced into one sentence again:
    # reported 395 + 12 = 407 and 16 + 72 = 88, which is exactly what Section 4 prints.
    rep_ans = [rtok(reported, e) for e, ok in zip(eids, answered) if ok]
    rep_exc = [rtok(reported, e) for e, ok in zip(eids, answered) if not ok]
    repeat_ans = [rtok(repeat, e) for e, ok in zip(eids, answered) if ok]
    repeat_exc = [rtok(repeat, e) for e, ok in zip(eids, answered) if not ok]
    flip_eids = [e for e, f in zip(eids, parity_flips) if f]
    if any(v is None for v in rep_ans + rep_exc) or \
            any(rtok(reported, e) is None for e in flip_eids):
        raise SystemExit(
            "REFUSING to write numbers.json: a frontier row carries no reasoning-token "
            "count, so section 12 cannot say which rows the repeat could bound."
        )
    prompt_edit_flips_on_compared = int((parity_flips & answered).sum())

    return dict(
        capped_path=str(capped_path.relative_to(REPO)),
        repeat_path=str(repeat_path.relative_to(REPO)),
        capped_sha256=sha256_file(capped_path),
        prompt_hash=capped[ok_eids[0]]["prompt_hash"],
        reasoning_effort=str(sorted(map(str, efforts))[0]),
        capped_max_completion_tokens=int(json.loads(
            (capped_path.parent / "run_manifest.json").read_text()
        )["models"]["gpt6_astra"]["max_completion_tokens"]),
        repeat_max_completion_tokens=int(json.loads(
            (repeat_path.parent / "run_manifest.json").read_text()
        )["models"]["gpt6_astra"]["max_completion_tokens"]),
        n_paired=len(eids),
        n_compared=int(answered.sum()),
        n_capped_no_answer=n_unanswered,
        n_flips=int((a_pred[answered] != b_pred[answered]).sum()),
        agreement=float((a_pred[answered] == b_pred[answered]).mean()),
        capped_macro_bacc=capped_macro,
        repeat_macro_bacc=repeat_macro,
        macro_bacc_delta_pp=float((capped_macro - repeat_macro) * 100),
        # Where the prompt-parity flips fell. This locates the movement and nothing more.
        # Every flip is inside the rows the capped pass could not answer, which is exactly
        # the set this repeat says nothing about, so the concentration cannot be read as
        # evidence that the prompt rather than a fresh draw moved them. The expected count
        # under a uniform scatter is emitted so a reader can see how far from uniform it
        # is, not as a test of a hypothesis this design cannot test.
        parity_flips=n_flips,
        parity_flips_in_capped_no_answer=int((parity_flips & ~answered).sum()),
        parity_flips_expected_in_capped_no_answer=float(n_flips * n_unanswered / len(eids)),
        # The prompt edit moved nothing on the compared rows either, so that subset
        # separates neither hypothesis: it is silent on prompt and on draw alike.
        prompt_edit_flips_on_compared=prompt_edit_flips_on_compared,
        # The reasoning-token split that bounds what the repeat covers.
        compared_reasoning_zero_n_reported=int(sum(1 for v in rep_ans if v == 0)),
        compared_reasoning_mean_reported=float(sum(rep_ans) / len(rep_ans)),
        excluded_reasoning_nonzero_n_reported=int(sum(1 for v in rep_exc if v > 0)),
        compared_reasoning_zero_n_repeat=int(sum(1 for v in repeat_ans if v == 0)),
        compared_reasoning_mean_repeat=float(sum(repeat_ans) / len(repeat_ans)),
        excluded_reasoning_nonzero_n_repeat=int(sum(1 for v in repeat_exc if v > 0)),
        excluded_reasoning_n=len(rep_exc),
        parity_flip_reasoning_mean_repeat=float(
            sum(rtok(repeat, e) for e in flip_eids) / len(flip_eids)) if flip_eids else None,
        parity_flip_reasoning_mean_reported=float(
            sum(rtok(reported, e) for e in flip_eids) / len(flip_eids)) if flip_eids else None,
        # Both passes of the repeat carry the pre-parity prompt. No reported frontier
        # number does. The paper must not present this bound as measured on the prompt it
        # reports, so both hashes are emitted and the difference is stated in section 12.
        reported_path=str(reported_path.relative_to(REPO)),
        reported_prompt_hash=reported[ok_eids[0]]["prompt_hash"],
        note="two passes of the SAME frontier prompt, reasoning effort and model, "
             "differing only in our completion-token cap. That prompt is the SUPERSEDED "
             "pre-parity prompt, not the corrected prompt behind the reported frontier "
             "numbers. Restricted to the claims the capped pass answered, which are the "
             "shorter-answer rows and the rows on which the model reports no reasoning "
             "tokens, so it bounds run-to-run movement on those rows only. All of the "
             "prompt-parity flips are outside that set.",
    )


def _holm_adjust(members: dict) -> dict:
    """Holm-Bonferroni step-down over {name: raw p}, monotonicity enforced.

    Factored out of holm_bonferroni so the full family and every subset of it are adjusted
    by the same code. The family-size sensitivity below recomputes Holm on subsets, and an
    argument about how the answer moves with the family boundary is worth nothing if the
    subsets are adjusted by a second implementation.
    """
    ordered = sorted(members.items(), key=lambda kv: kv[1])
    n = len(ordered)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        # Step down, multiply by the number of remaining hypotheses, and enforce
        # monotonicity so an adjusted p never falls below one earlier in the sequence.
        running = max(running, min(1.0, p * (n - i)))
        adjusted[name] = running
    return adjusted


# Every cross-fitted comparator block in out["cascade"]["crossfit"], classified once, so
# that the count of separated results and the Holm family are drawn from one inventory.
#
# REPORTED_INFERENCE_BLOCKS are the comparisons the paper reports as results: the cascade
# against each of the three reported configurations. They are the blocks holm_bonferroni
# puts in its family by name, and the ones section 12 and appendix E enumerate.
#
# NONRESULT_INFERENCE_BLOCKS are computed for a disclosure or a sensitivity argument and
# are not results of the paper, so they are not counted among the comparisons that clear
# its two instruments and they are not members of the multiplicity family. Each carries
# the reason it is not a result, because the exclusion has to be arguable and not assumed.
REPORTED_INFERENCE_BLOCKS = ("vs_astra_inference", "vs_astra_high_inference",
                             "vs_jev_inference")
NONRESULT_INFERENCE_BLOCKS = {
    "vs_astra_high_pre_parity_inference":
        "the cascade measured against the SUPERSEDED pre-parity high-effort frontier run. "
        "It exists to show what the prompt-parity correction did to the comparison it "
        "replaced, so it is a sensitivity check on a retracted comparator, not a reported "
        "configuration. It is not a member of the Holm family either, and counting it "
        "among the separated results while dividing that count by the survivors of a "
        "family it does not belong to would state the paper's two multiplicity "
        "inventories over different universes.",
}


def _reported_inference_blocks(xf: dict) -> list[str]:
    """The cross-fitted comparator blocks the paper reports, named rather than globbed.

    A glob over `vs_*_inference` silently promotes every new block into whatever it feeds.
    That happened: the pre-parity sensitivity block was added as a comparator, the glob
    took it, and the paper printed a count of separated results one higher than the three
    its own sentences enumerate. Any block that is neither reported nor declared a
    non-result raises here, so the next one has to be classified before it can be counted.
    """
    unclassified = sorted(k for k in xf
                          if k.startswith("vs_") and k.endswith("_inference")
                          and k not in REPORTED_INFERENCE_BLOCKS
                          and k not in NONRESULT_INFERENCE_BLOCKS)
    if unclassified:
        raise AssertionError(
            f"unclassified cross-fitted comparator block(s): {unclassified}. Add each to "
            "REPORTED_INFERENCE_BLOCKS if the paper reports it as a result, which also "
            "means adding it to the holm_bonferroni family and to the prose that "
            "enumerates the separated comparisons, or to NONRESULT_INFERENCE_BLOCKS with "
            "the reason it is not a result.")
    return [k for k in REPORTED_INFERENCE_BLOCKS if k in xf]


def _n_clearing_uncorrected(out: dict) -> int:
    """How many comparisons clear BOTH instruments before any multiplicity adjustment.

    Section 5's criterion is per estimand: on false verification the bootstrap interval and
    the exact McNemar rest on the same gold-unsupported claims, so both are required; on
    macro-balanced accuracy they address different quantities, so the interval governs.
    This is that rule, counted once, so the five places the paper states the number all
    cite one macro instead of an English word that drifts. It drifted twice: a repair
    round updated three of the five and left two behind, and the word-number gate exempts
    spelled numbers below four, so nothing caught it.

    It then drifted a third time, the other way, which is why the comparators are named by
    _reported_inference_blocks: a glob over `vs_*_inference` counted the pre-parity
    sensitivity block, so the macro printed four while every sentence using it enumerated
    three, and the Holm family the sentence divides it into has thirteen members and no
    pre-parity one.
    """
    xf = out["cascade"]["crossfit"]
    gap = out["complementarity"]["fvr_gap_inference"]
    infs = _reported_inference_blocks(xf)
    # Reported configurations only. The two MiniCheck pairs are evidence for an exclusion,
    # not comparisons the paper reports, and section 12 counts them separately.
    acc = [v["macro_bacc_delta_ci_pp"] for k, v in out["pairs"].items()
           if "minicheck" not in k]
    acc += [xf[k]["macro_bacc_delta_ci_pp"] for k in infs]
    fvr = [(gap["gap_ci_pp"], gap["mcnemar_p"])]
    fvr += [(xf[k]["fvr_inference"]["fvr_delta_ci_pp"],
             xf[k]["fvr_inference"]["mcnemar_p_median"]) for k in infs]
    n = 0
    for ci in acc:
        lo, hi = sorted(ci)
        if lo > 0 or hi < 0:
            n += 1
    for ci, pv in fvr:
        lo, hi = sorted(ci)
        if (lo > 0 or hi < 0) and pv < 0.05:
            n += 1
    return n


def holm_bonferroni(out: dict) -> dict:
    """Holm-Bonferroni over every McNemar test the paper reports, as the spec requires.

    The spec's "Pairwise classification significance" section says to correct multiple
    comparisons with Holm-Bonferroni and to report raw and corrected p; nothing in the
    errata or the decision log overrides it, and the paper said twice that it applied no
    correction. This closes that gap.

    The family is defined here, in code, rather than chosen after seeing which member
    survives, because it decides the answer: the cascade's guard penalty against Astra
    has eight members at or above its raw p in the family below, which carries it just over
    0.05, and it would clear at any ten of these tests. The rule is every
    exact McNemar this paper reports as evidence about a comparison between
    configurations. That takes in the three single-system accuracy pairs, the two
    excluded-arm pairs (reported in Appendix E, and evidence for the exclusion), the
    pooled false-verification gap, the cascade against each neighbour on accuracy and on
    false verification, and the fixed-threshold guard comparison the discussion points at.
    tests/test_paper_evidence.py asserts that every McNemarP macro the paper prints
    belongs to this family, so the family cannot silently shrink.
    """
    xf = out["cascade"]["crossfit"]
    sweep99 = max(out["cascade"]["in_sample_sweep"], key=lambda r: r["threshold"])
    members = {
        "JevAstra": out["pairs"]["jev|astra"]["mcnemar_p"],
        "JevAstraHigh": out["pairs"]["jev|astra_high"]["mcnemar_p"],
        "AstraAstraHigh": out["pairs"]["astra|astra_high"]["mcnemar_p"],
        "FVRGap": out["complementarity"]["fvr_gap_inference"]["mcnemar_p"],
        "CascadeAstra": xf["vs_astra_inference"]["mcnemar_p_median"],
        "CascadeJev": xf["vs_jev_inference"]["mcnemar_p_median"],
        "CascadeAstraHigh": xf["vs_astra_high_inference"]["mcnemar_p_median"],
        "CascadeAstraFVR": xf["vs_astra_inference"]["fvr_inference"]["mcnemar_p_median"],
        "CascadeJevFVR": xf["vs_jev_inference"]["fvr_inference"]["mcnemar_p_median"],
        "CascadeAstraHighFVR":
            xf["vs_astra_high_inference"]["fvr_inference"]["mcnemar_p_median"],
        "CascadeAtNinetyNineFVRVsAstra": sweep99["fv_vs_astra_mcnemar_p"],
    }
    for key, name in (("jev|minicheck", "JevMiniCheck"),
                      ("astra|minicheck", "AstraMiniCheck")):
        if key in out["pairs"]:
            members[name] = out["pairs"][key]["mcnemar_p"]

    # Four members are not single exact tests. They are the MEDIAN of one exact McNemar
    # per cross-fitting repeat, computed on overlapping resamples of the same claims, so
    # they have no null distribution of their own and Holm over a family containing them
    # is descriptive rather than an error-rate guarantee. The paper has to say so, so the
    # set is named here rather than left for a reader to infer from the key names.
    median_members = sorted(
        k for k in members if k in {"CascadeAstra", "CascadeJev", "CascadeAstraHigh",
                                    "CascadeAstraFVR", "CascadeJevFVR",
                                    "CascadeAstraHighFVR"})

    adjusted = _holm_adjust(members)
    n = len(members)

    # The family size decides one of the two answers, and the only sensitivity the paper
    # reported was the one perturbation arithmetic guarantees cannot change it: the two
    # MiniCheck members are smaller than every other member, so removing them moves only
    # step-down positions. Removing members LARGER than the guard penalty does move it.
    # This computes how far, by removing them in decreasing raw p and recomputing Holm
    # from scratch at each size, and records the largest family at which the guard
    # penalty's adjusted p falls below 0.05 together with that value and the members that
    # had to go. It is the sensitivity that can change the answer, so it is the one the
    # paper has to print next to the boundary it fixed in advance.
    guard = "CascadeAstraFVR"
    guard_break_n = guard_break_p = None
    guard_break_dropped: list[str] = []
    trial = dict(members)
    for name in [None] + [k for k in sorted(members, key=lambda k: -members[k])
                          if k != guard]:
        if name is not None:
            del trial[name]
            guard_break_dropped.append(name)
        if _holm_adjust(trial)[guard] < 0.05:
            guard_break_n = len(trial)
            guard_break_p = float(_holm_adjust(trial)[guard])
            break
    # What Holm multiplies the guard penalty by is the number of members whose raw p is at
    # least its own, so the members ABOVE it in the step-down are the ones whose presence
    # decides it and the three below it are irrelevant to it. "A family of ten" is
    # therefore not sufficient on its own: the ten have to be ten that still include a
    # smaller member at every position below it. Rather than reason about that in prose,
    # this checks it, by dropping each larger member in turn and recomputing.
    guard_p = members[guard]
    above = [k for k, v in members.items() if k != guard and v >= guard_p]
    clears_dropping_any_one_above = all(
        _holm_adjust({k: v for k, v in members.items() if k != drop})[guard] < 0.05
        for drop in above) if above else False

    # And the number behind "dropping the two MiniCheck tests does not rescue it", so that
    # sentence rests on a recomputation rather than on an argument about orderings.
    without_minicheck = {k: v for k, v in members.items() if "MiniCheck" not in k}
    return dict(
        family_size=n,
        cascade_guard_holm_clears_at_family_n=guard_break_n,
        cascade_guard_holm_p_at_that_family=guard_break_p,
        cascade_guard_holm_break_dropped=guard_break_dropped,
        cascade_guard_holm_members_above_n=len(above),
        cascade_guard_holm_members_above=sorted(above),
        cascade_guard_holm_clears_dropping_any_one_above=bool(
            clears_dropping_any_one_above),
        cascade_guard_holm_p_excluding_minicheck=float(
            _holm_adjust(without_minicheck)[guard]) if guard in without_minicheck else None,
        cascade_guard_holm_note="the guard penalty's adjusted p depends on how many "
             "members sit at or above its raw p, so the family boundary decides it. "
             "Removing the two MiniCheck members cannot: both are smaller than every "
             "other member. Removing any one of the larger members can, and "
             "cascade_guard_holm_clears_at_family_n is the largest family at which it "
             "clears 0.05. The paper reports the family it fixed in code, and this bound "
             "beside it.",
        method="Holm-Bonferroni, step-down, monotonicity enforced",
        raw={k: float(v) for k, v in members.items()},
        adjusted={k: float(v) for k, v in adjusted.items()},
        survives_at_05=sorted(k for k, v in adjusted.items() if v < 0.05),
        # The comparisons between the three reported configurations and the cascade, i.e.
        # excluding the two MiniCheck pairs, which are evidence for an exclusion rather
        # than results. Reported separately so a reader can see that dropping them from
        # the family does not change which of the others survive: both MiniCheck p-values
        # are smaller than every other member, so they only shift the step-down indices.
        survives_at_05_excluding_minicheck=sorted(
            k for k, v in adjusted.items() if v < 0.05 and "MiniCheck" not in k),
        # The UNCORRECTED count, emitted because the paper states it in five places and
        # has twice printed two different values in the same appendix: a repair round
        # updated three of the five and left two behind, and the word-number gate exempts
        # spelled numbers below four so nothing caught it. It is a macro now.
        n_clearing_uncorrected=_n_clearing_uncorrected(out),
        median_members=median_members,
        # How wide the family would have to be before the one result the paper claims
        # stopped surviving. Holm's largest possible multiplier for any member is the
        # family size, so this is the conservative break point and it answers "you chose
        # the family that lets your result through" with a number rather than a promise.
        fvr_gap_raw_p=float(members["FVRGap"]),
        fvr_gap_holm_break_family_n=int(math.ceil(0.05 / members["FVRGap"])),
        median_members_note="these members are the median of one exact McNemar per "
             "cross-fitting repeat over overlapping resamples of the same claims, not "
             "single exact tests; the adjustment on them is descriptive and the paper "
             "says so in section 5 and appendix E.",
        note="the family is every exact McNemar this paper reports as evidence about a "
             "comparison between configurations, fixed in code before the adjustment is "
             "computed. A smaller family would let more results through and the paper "
             "does not get to choose one after the fact.",
    )


def error_correlation(df, a: str, b: str) -> dict:
    """How correlated the two systems' errors actually are.

    Section 7 opened the subsection that licenses the whole cascade with "two systems with
    correlated errors are substitutes; these two are not". On the locked predictions they
    ARE correlated, and the paper reported no measure of it. Composition does not need
    zero correlation, it needs imperfect correlation plus a signal for which of the two is
    about to be wrong, so the honest version rests on the discordant cells and on the
    joint-failure excess rather than on a denial the data refutes.
    """
    ca = (df[a].to_numpy() == df.gold.to_numpy())
    cb = (df[b].to_numpy() == df.gold.to_numpy())
    n = len(df)
    both = int((ca & cb).sum())
    neither = int((~ca & ~cb).sum())
    a_only = int((ca & ~cb).sum())
    b_only = int((~ca & cb).sum())
    po = (both + neither) / n
    pe = ((both + a_only) * (both + b_only) + (neither + b_only) * (neither + a_only)) / n**2
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    num = both * neither - a_only * b_only
    den = ((both + a_only) * (neither + b_only) * (both + b_only) * (neither + a_only)) ** 0.5
    phi = num / den if den else float("nan")
    # What independence would predict for the cell where both are wrong.
    expected_neither = (1 - ca.mean()) * (1 - cb.mean()) * n
    return dict(
        n=n, both_correct=both, neither_correct=neither,
        a_only_correct=a_only, b_only_correct=b_only,
        cohens_kappa=float(kappa), phi=float(phi),
        neither_correct_expected_if_independent=float(expected_neither),
        joint_failure_excess_ratio=float(neither / expected_neither) if expected_neither else None,
        note="per-item correctness agreement between the two arms. Positive kappa means "
             "the errors are correlated, which they are; the cascade rests on the "
             "discordant cells being large, not on the errors being independent.",
    )


def per_dataset_concentration(configs: dict, a: str, b: str) -> dict:
    """How much of the macro difference between two arms comes from its widest cells.

    The draft said the two extreme datasets "largely cancel", which is checkable and is
    the wrong way round: they nearly are the whole macro difference, and the remaining
    datasets sum to close to nothing. A macro average over eleven datasets of about
    forty-five claims can be almost entirely two cells, and a reader deciding how much
    to trust a tie needs that number rather than a reassurance.
    """
    per_a, per_b = configs[a]["per_dataset"], configs[b]["per_dataset"]
    deltas = {d: (per_a[d] - per_b[d]) * 100 for d in per_a}
    ranked = sorted(deltas.items(), key=lambda kv: -abs(kv[1]))
    top2 = ranked[:2]
    rest = ranked[2:]
    total = float(sum(deltas.values()))
    top2_sum = float(sum(v for _, v in top2))
    return dict(
        n_datasets=len(deltas),
        per_dataset_delta_pp=deltas,
        macro_delta_pp=float(total / len(deltas)),
        sum_of_deltas_pp=total,
        widest_two=[dict(dataset=d, delta_pp=v) for d, v in top2],
        widest_two_sum_pp=top2_sum,
        remaining_sum_pp=float(sum(v for _, v in rest)),
        n_remaining=len(rest),
        widest_two_share_of_sum_pct=float(top2_sum / total * 100) if total else None,
        # Signed sums and absolute sums are different facts, and an earlier draft drew the
        # opposite conclusion from confusing them. Nine cells that SUM to nothing because
        # they have opposite signs are not nine cells that ARE nothing. Both are reported,
        # with the share of each group's total magnitude that its own signs cancel.
        sum_abs_deltas_pp=float(sum(abs(v) for v in deltas.values())),
        widest_two_sum_abs_pp=float(sum(abs(v) for _, v in top2)),
        remaining_sum_abs_pp=float(sum(abs(v) for _, v in rest)),
        max_abs_remaining_pp=float(max(abs(v) for _, v in rest)) if rest else None,
        mean_abs_remaining_pp=float(np.mean([abs(v) for _, v in rest])) if rest else None,
        widest_two_cancellation_pct=float(
            (1 - abs(top2_sum) / sum(abs(v) for _, v in top2)) * 100),
        remaining_cancellation_pct=float(
            (1 - abs(sum(v for _, v in rest)) / sum(abs(v) for _, v in rest)) * 100),
        note="signed per-dataset balanced-accuracy differences in percentage points, "
             f"{a} minus {b}. The macro difference is their mean. 'widest two' are the two "
             "largest in absolute value, which is a description of this sample and not a "
             "selected subgroup result.",
    )


# Clipping constants the log loss is re-scored at, half a decade apart from the
# configured 1e-15 up to a twentieth. configs/benchmark.yaml marks the configured value
# "PROVISIONAL: preregister the value at G2A" and G2A never ran for this study, so the
# reported NLL rests on a constant nobody fixed in advance. The grid is dense enough to
# bracket the crossing against a constant one-half forecast rather than extrapolate it,
# and the resolution the vendor actually reports on is added to it per arm.
LOGLOSS_EPS_GRID = tuple(10.0 ** (-k / 2.0) for k in range(2, 31))

def jev_rerun_stability(df, path: Path) -> dict:
    """The same verifier, the same claims, an earlier run: how much moved.

    Threats-to-validity D3 is "your single run was lucky", and the honest answer at this
    scale is not a repeatability study but a disclosure: a second run of the cheap system
    over the identical example IDs is already on disk, and the paper reports how far
    apart the two are rather than letting a reader find the other file in the release.
    The comparison is made on the paired frame, so it uses the same gold vector and the
    same exclusions as every other number here.
    """
    rows = {r["example_id"]: r
            for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}
    eids = list(df.eid)
    missing = [e for e in eids if e not in rows or rows[e].get("status") != "ok"]
    if missing:
        raise SystemExit(
            f"REFUSING to write numbers.json: the earlier Jev run covers "
            f"{len(eids) - len(missing)}/{len(eids)} of the paired claims, so section 12 "
            "cannot report run-to-run movement on the same items."
        )
    prior = np.array([1 if rows[e]["label"] == "SUPPORTED" else 0 for e in eids])
    gold, ds_a = df.gold.to_numpy(), df.ds.to_numpy()
    headline = df.jev.to_numpy()
    flips = prior != headline
    conf = np.abs(df.jev_p.to_numpy() - 0.5) + 0.5
    prior_macro = macro_bacc_arr(gold, prior, ds_a)
    headline_macro = macro_bacc_arr(gold, headline, ds_a)
    per, per_fv = {}, {}
    for name, t in df.groupby("ds"):
        pos = np.asarray(t.index)
        per[name] = bacc_fast(gold[pos], prior[pos])
        # Section 7.4 reads a dataset's specificity off the reported run and calls it
        # pinned at the top of the scale. Whether the earlier pass of the same model
        # keeps it there is a count, not an adjective, so the count is carried here per
        # dataset and the prose cites it.
        per_fv[name] = int(((gold[pos] == 0) & (prior[pos] == 1)).sum())
    return dict(
        path=str(path.relative_to(REPO)),
        sha256=sha256_file(path),
        n=int(len(eids)),
        n_flips=int(flips.sum()),
        agreement=float((~flips).mean()),
        macro_bacc=float(prior_macro),
        macro_bacc_delta_pp=float((prior_macro - headline_macro) * 100),
        false_verification_rate=float(M.false_verification_rate(gold.tolist(), prior.tolist())),
        per_dataset=per,
        per_dataset_false_verified=per_fv,
        max_flip_confidence=float(conf[flips].max()) if flips.any() else 0.0,
        note="an EARLIER exploratory run of the same model version over the same example "
             "IDs. It carries no run manifest and a reduced row schema, so the two runs "
             "cannot be proven to share a configuration; it is reported as a lower bound "
             "on run-to-run movement, never as a second measurement of the headline.",
    )


# --------------------------------------------------------------------------------
# LaTeX macros
# --------------------------------------------------------------------------------
_DIGIT_WORDS = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
}
_THRESHOLD_WORDS = {
    0.5: "Fifty", 0.7: "Seventy", 0.8: "Eighty", 0.9: "Ninety",
    0.95: "NinetyFive", 0.97: "NinetySeven", 0.99: "NinetyNine",
}


def macro_token(text: str) -> str:
    """A LaTeX-legal command name: letters only, so digits become words."""
    out = []
    for ch in text:
        if ch.isalpha():
            out.append(ch)
        elif ch in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[ch])
    return "".join(out)


class Macros:
    """Collects \\newcommand definitions and refuses anything LaTeX would mangle."""

    def __init__(self) -> None:
        self._m: "OrderedDict[str, str]" = OrderedDict()
        self._sections: list[tuple[str, str]] = []

    def section(self, title: str) -> None:
        self._sections.append((title, ""))
        self._m[f"__section__{len(self._sections)}"] = title

    def add(self, name: str, value, fmt: str = "{:.1f}") -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            raise ValueError(f"macro {name}: refusing to emit a missing or non-finite value")
        token = macro_token(name)
        if token != name:
            raise ValueError(f"macro {name!r} is not letters-only; use {token!r}")
        if name in self._m:
            raise ValueError(f"macro {name!r} defined twice")
        s = fmt.format(value)
        if any(ch not in "0123456789.,+-" for ch in s):
            raise ValueError(
                f"macro {name!r} formats to {s!r}; macro bodies carry bare numbers only "
                "(a percent sign would start a LaTeX comment, a dollar sign would open math)"
            )
        self._m[name] = s
        return s

    def render(self) -> str:
        lines = [
            "% arxiv/generated/numbers.tex",
            "% GENERATED by scripts/verify/paired_analysis.py. Do not edit by hand.",
            "% Every number in the paper is one of these macros (ADR-006, CLAUDE.md rule 4),",
            "% so a hand-typed digit in main.tex is a build failure rather than a judgement call.",
            "% Macro bodies are bare numbers: no units, no percent signs, no dollar signs.",
            "% A negative body carries an ASCII hyphen, so wrap those in math mode to get a",
            "% real minus sign: $\\JevAstraDeltaCILo$, not \\JevAstraDeltaCILo.",
            "",
        ]
        for name, value in self._m.items():
            if name.startswith("__section__"):
                # "=" padding, not "-": a run of hyphens anywhere in this file would trip
                # the em dash gate below, which is deliberately blunt.
                lines += ["", f"% ==== {value} " + "=" * max(0, 68 - len(value)), ""]
            else:
                lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
        lines.append("")
        return "\n".join(lines)

    def count(self) -> int:
        return sum(1 for k in self._m if not k.startswith("__section__"))


def add_scientific(m: "Macros", name: str, value: float, sig: int = 2) -> None:
    """Emit <name>Mantissa and <name>Exponent for a value too small for fixed decimals.

    A macro body carries bare digits (Macros.add), so 1.6e-16 cannot be one macro. The
    prose composes the two as $\\<name>Mantissa\\times10^{-\\<name>Exponent}$. Only
    positive values below one are expected here; anything else is a caller error rather
    than something to render badly.
    """
    if not (0 < value < 1):
        raise ValueError(f"macro {name}: scientific form expects 0 < value < 1, got {value}")
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10.0 ** exponent)
    # Rounding the mantissa can carry it to 10.0; renormalise rather than print "10.0e-16".
    if round(mantissa, sig - 1) >= 10.0:
        mantissa, exponent = mantissa / 10.0, exponent + 1
    m.add(f"{name}Mantissa", mantissa, f"{{:.{sig - 1}f}}")
    m.add(f"{name}Exponent", -exponent, "{:d}")


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
def main() -> int:
    cfg = yaml.safe_load((REPO / "configs/benchmark.yaml").read_text())
    seed = int(cfg["random_seed"])
    ece_bins = int(cfg["metrics"]["ece_bins"])
    ece_sens_bins = int(cfg["metrics"]["ece_sensitivity_bins"])
    logloss_eps = float(cfg["metrics"]["logloss_epsilon"])
    n_boot = int(cfg["metrics"]["bootstrap_resamples"])

    df, meta = build_frame()
    loaded = meta["loaded"]
    minicheck_ok = meta["minicheck_complete"]

    arms = ["jev", "astra", "astra_high"] + (["minicheck"] if minicheck_ok else [])
    assert_fast_path_matches_metrics(df, arms)

    prices = yaml.safe_load((REPO / "configs/pricing_2026-09-20.yaml").read_text())["prices_per_mtok_usd"]
    jev_in_price = float(prices["jev-1.13.0"]["input"])
    a_in, a_out = float(prices["gpt-6-astra"]["input"]), float(prices["gpt-6-astra"]["output"])
    assert float(prices["jev-1.13.0"]["output"]) == 0.0, "Jev output pricing changed; update the cost model"

    def cost_1k(in_tok, out_tok, p_in, p_out):
        return float(np.mean(in_tok) * 1000 / 1e6 * p_in + np.mean(out_tok) * 1000 / 1e6 * p_out)

    jev_cost = cost_1k(df.jev_in, df.jev_out, jev_in_price, 0.0)
    astra_cost = cost_1k(df.astra_in, df.astra_out, a_in, a_out)
    astra_high_cost = cost_1k(df.astra_high_in, df.astra_high_out, a_in, a_out)

    # Where the headline cost ratio actually comes from. The draft attributed it to the
    # vendor billing input only, which is checkable and wrong: re-pricing Jev's output
    # tokens at its own input rate moves the ratio by a few per cent, because the
    # response is tiny. The rest is the per-token input price. This block exists so the
    # paper can name the driver from computed evidence rather than from an intuition.
    jev_cost_if_output_billed = cost_1k(df.jev_in, df.jev_out, jev_in_price, jev_in_price)
    ratio_if_output_billed = float(astra_cost / jev_cost_if_output_billed)
    ratio_headline = float(astra_cost / jev_cost)
    # The ratio is an exact product of three measured factors, and the paper prints all
    # three because a decomposition that does not close is worse than none:
    #   input price ratio  /  input token ratio  x  frontier output uplift  =  headline
    # The middle factor runs the wrong way (the decision model sends the longer request)
    # and an earlier draft left it out, which made 238 appear to explain 187.
    input_price_ratio = float(a_in / jev_in_price)
    input_token_ratio = float(df.jev_in.mean() / df.astra_in.mean())
    astra_output_uplift = float(
        astra_cost / cost_1k(df.astra_in, df.astra_out, a_in, 0.0))
    identity = input_price_ratio / input_token_ratio * astra_output_uplift
    assert abs(identity - ratio_headline) < 1e-9, (
        f"cost decomposition does not close: {identity} != {ratio_headline}")
    cost_decomposition = dict(
        input_token_ratio_jev_over_astra=input_token_ratio,
        astra_output_token_uplift=astra_output_uplift,
        ratio_from_price_and_tokens_only=float(input_price_ratio / input_token_ratio),
        identity_check=identity,
        jev_input_price_per_mtok_usd=jev_in_price,
        jev_output_price_per_mtok_usd=float(prices["jev-1.13.0"]["output"]),
        astra_input_price_per_mtok_usd=a_in,
        astra_output_price_per_mtok_usd=a_out,
        input_price_ratio=input_price_ratio,
        jev_cost_per_1k_usd=jev_cost,
        jev_cost_per_1k_usd_if_output_billed_at_input_rate=jev_cost_if_output_billed,
        cost_ratio_headline=ratio_headline,
        cost_ratio_if_output_billed_at_input_rate=ratio_if_output_billed,
        free_output_share_of_ratio_pct=float(
            (ratio_headline - ratio_if_output_billed) / ratio_headline * 100),
        note="The headline cost ratio is an exact product of three measured factors: the "
             "input-price ratio, divided by the ratio of mean input tokens (which runs "
             "the other way, because the decision model sends the longer request), times "
             "the frontier arm's output-token uplift. identity_check reproduces "
             "cost_ratio_headline to machine precision and the assertion in this file "
             "fails the run if it ever stops doing so. The vendor charging for input only "
             "is NOT the driver: billing the decision model's measured output tokens at "
             "its own input rate leaves the ratio almost unchanged. Both prices are list "
             "prices from one snapshot of two "
             "vendors' public pages on a single day, and free output tokens are a pricing "
             "policy, not a property of the model (threats-to-validity B6).",
    )

    def cascade_cost_per_1k(conf: np.ndarray) -> float:
        """Every claim pays Jev; escalated claims additionally pay Astra on their own
        (longer) token profile, not on the average claim's."""
        jev_c = df.jev_in.to_numpy() * jev_in_price / 1e6
        astra_c = np.where(
            ~conf, df.astra_in.to_numpy() * a_in / 1e6 + df.astra_out.to_numpy() * a_out / 1e6, 0.0
        )
        return float((jev_c + astra_c).mean() * 1000)

    # ---- per configuration ------------------------------------------------------
    gold_l = df.gold.tolist()
    boot = paired_bootstrap(df, arms, n_boot, seed)

    PROB_COL = {"jev": "jev_p", "minicheck": "minicheck_p"}
    LAT_COL = {"jev": "jev_ms", "astra": "astra_ms", "astra_high": "astra_high_ms",
               "minicheck": "minicheck_ms"}
    COSTS = {"jev": jev_cost, "astra": astra_cost, "astra_high": astra_high_cost, "minicheck": 0.0}

    configs: dict[str, dict] = {}
    for arm in arms:
        pred_l = df[arm].tolist()
        per_ds = per_dataset_bacc(df, arm)
        lo, hi = pct_ci(boot[arm])
        lat = df[LAT_COL[arm]].to_numpy()
        c = dict(
            macro_bacc=M.mean_balanced_accuracy(per_ds),
            macro_bacc_ci=[lo, hi],
            per_dataset=per_ds,
            raw_accuracy=M.accuracy(gold_l, pred_l),
            pooled_balanced_accuracy=M.balanced_accuracy(gold_l, pred_l),
            false_verification_rate=M.false_verification_rate(gold_l, pred_l),
            # The same quantity as a claim count. A guard rate this paper asks operators
            # to buy on is a share of 198 claims, and a difference of a few tenths of a
            # point is a difference of one claim; the count makes that unhideable.
            n_false_verified=int(sum(1 for g, q in zip(gold_l, pred_l) if g == 0 and q == 1)),
            verified_precision=M.verified_precision(gold_l, pred_l),
            macro_f1=M.macro_f1(gold_l, pred_l),
            mcc=M.mcc(gold_l, pred_l),
            pred_supported_rate=float(np.mean(pred_l)),
            cost_per_1k_usd=COSTS[arm],
            latency_ms=dict(p50=float(np.percentile(lat, 50)), p90=float(np.percentile(lat, 90)),
                            p95=float(np.percentile(lat, 95)), p99=float(np.percentile(lat, 99)),
                            mean=float(lat.mean())),
        )
        if arm in ("jev", "astra", "astra_high"):
            in_col = {"jev": "jev_in", "astra": "astra_in", "astra_high": "astra_high_in"}[arm]
            out_col = {"jev": "jev_out", "astra": "astra_out", "astra_high": "astra_high_out"}[arm]
            c["mean_input_tokens"] = float(df[in_col].mean())
            c["mean_output_tokens"] = float(df[out_col].mean())
            # Threat A6 is "your sample quietly dropped the long documents a 32k verifier
            # cannot take". The answer is the largest input any arm actually saw, next to
            # the count of rows that failed for any reason, which is zero. Both go in the
            # evidence so section 12 can state it rather than assert it.
            c["max_input_tokens"] = int(df[in_col].max())
            c["p95_input_tokens"] = float(np.percentile(df[in_col].to_numpy(), 95))
            # The paper calls the decision model non-generative and prints its mean output
            # tokens in the same table row, where the two read as a contradiction: 35.4
            # billed output tokens against a text generator's 10.0. The spread is what
            # settles it. A generator's output length tracks the claim; a fixed
            # serialisation envelope around one enumerated value does not, and this arm's
            # count takes two values over every claim in the study while the frontier arm's
            # spans two orders of magnitude. Reported per arm so the reconciliation in the
            # table caption cites a measurement rather than an assurance.
            c["min_output_tokens"] = int(df[out_col].min())
            c["max_output_tokens"] = int(df[out_col].max())
            c["n_distinct_output_tokens"] = int(df[out_col].nunique())
        # Threats-to-validity B1 ("you crippled the frontier model") is answered by the
        # reasoning-effort ablation, and an ablation is only an answer if the knob moved
        # something. What the provider billed as internal reasoning is the measurement of
        # that, so it is reported rather than left in the JSONL for a reviewer to find.
        if arm in ("astra", "astra_high"):
            r = df[{"astra": "astra_reason", "astra_high": "astra_high_reason"}[arm]].to_numpy()
            c["reasoning_tokens"] = dict(
                median=float(np.median(r)), mean=float(r.mean()),
                max=int(r.max()), n_zero=int((r == 0).sum()), n=int(len(r)),
                note="provider-reported internal reasoning tokens per call, from "
                     "extra.reasoning_tokens in the released predictions. The effort "
                     "setting is what we asked for; this is what came back.",
            )
        # ---- calibration, entirely from src.metrics ------------------------------
        if arm in PROB_COL:
            p = df[PROB_COL[arm]].tolist()
            bins = M.equal_frequency_bins(p, ece_bins)
            # Two different AUROCs, kept apart on purpose because the abstract's claim
            # is about the second one and they are easy to conflate:
            #   auroc            does p_supported separate supported from unsupported
            #   selective_auroc  does the model's own confidence, max(p, 1-p), rank the
            #                    claims it gets RIGHT above the ones it gets wrong
            # Only the second one is the routing signal a cascade actually uses.
            conf_score = [max(pi, 1.0 - pi) for pi in p]
            own_correct = [int(a == b) for a, b in zip(pred_l, gold_l, strict=True)]
            c["calibration"] = dict(
                probability_provenance="native",
                selective_auroc=M.auroc(own_correct, conf_score),
                selective_auprc=M.auprc(own_correct, conf_score),
                selective_note="confidence max(p, 1-p) scored against the model's own "
                               "correctness: this is the quantity that justifies routing, "
                               "and it is NOT the same as `auroc`",
                brier=M.brier(gold_l, p),
                negative_log_loss=M.negative_log_loss(gold_l, p, eps=logloss_eps),
                logloss_epsilon=logloss_eps,
                # What the clip pays for, kept beside the figure it decides: with a
                # saturating probability head a few confidently wrong items are charged
                # -log(eps) each, and section 8 prints that share rather than letting the
                # NLL read as a property of the model alone.
                logloss_clip=M.negative_log_loss_clip_share(gold_l, p, eps=logloss_eps),
                # The clip's share says most of the NLL is a constant of ours. The next
                # question a reader asks is what the figure would be at a different
                # constant, and section 8 may not print the first without the second:
                # the reported NLL exceeds a constant one-half forecast's only for clips
                # below `crossover_eps`, far below the resolution the vendor reports on.
                probability_resolution=M.probability_resolution(p),
                logloss_eps_sensitivity=M.negative_log_loss_eps_sensitivity(
                    gold_l, p,
                    grid=LOGLOSS_EPS_GRID + (M.probability_resolution(p)["half_step"],),
                    eps=logloss_eps),
                # The bounded score of the same constant forecast, which no clip of ours
                # can move. This is the reference section 8 reads its verdict against.
                constant_half_brier=M.brier(gold_l, [0.5] * len(p)),
                ece=M.ece(gold_l, p, n_bins=ece_bins),
                ece_bins=ece_bins,
                ece_sensitivity=M.ece(gold_l, p, n_bins=ece_sens_bins),
                ece_sensitivity_bins=ece_sens_bins,
                auroc=M.auroc(gold_l, p),
                auprc=M.auprc(gold_l, p),
                reliability=[
                    dict(bin=i, n=len(idx),
                         mean_p=float(np.mean([p[k] for k in idx])),
                         observed_supported_rate=float(np.mean([gold_l[k] for k in idx])),
                         lo=float(min(p[k] for k in idx)), hi=float(max(p[k] for k in idx)))
                    for i, idx in enumerate(bins)
                ],
            )
            # Where the miscalibration actually is. Section 8 said "at the confident ends
            # of the scale" for three drafts and a test pinned the phrase; the bins say
            # something more specific, so the shape is computed here and the prose cites
            # it rather than asserting a direction from memory.
            rel = c["calibration"]["reliability"]
            gaps = [(b["bin"], b["mean_p"], b["observed_supported_rate"],
                     b["observed_supported_rate"] - b["mean_p"], b["n"]) for b in rel]
            total_w = sum(abs(g[3]) * g[4] for g in gaps)
            worst = sorted(gaps, key=lambda g: -abs(g[3]))[:2]
            best = min(gaps, key=lambda g: abs(g[3]))
            res_native = c["calibration"]["probability_resolution"]
            # The block of predictions the vendor reports as exactly 0.00. This used to be
            # computed as "the bins whose MEAN is 0.0", which can only ever find bin 0 and
            # so returned that bin's size (50) while the comment claimed it was counting
            # the zeros (86). Section 8 printed the 50 as the number of predictions
            # reported as zero and Appendix E printed the 86, two values for one quantity.
            # The count now comes from the one place that measures it, and the share of it
            # that equal-count binning puts in the lowest bin is carried beside it so the
            # prose can say what the binning does without inventing a second total.
            n_at_zero = int(res_native["n_at_zero"])
            lowest = [b for b in rel if b["lo"] == 0.0 and b["hi"] == 0.0]
            interior_hi = 0.95

            # Confidence as section 3 defines it for routing, max(p, 1-p), which is what
            # makes a predicted 0.05 a CONFIDENT prediction and not an uncertain one.
            def _conf(p: float) -> float:
                return max(p, 1.0 - p)

            c["calibration"]["shape"] = dict(
                worst_bins=[dict(bin=b, mean_p=mp, observed=ob, gap=gp, n=n,
                                 confidence=_conf(mp))
                            for b, mp, ob, gp, n in worst],
                best_bin=dict(bin=best[0], mean_p=best[1], observed=best[2], gap=best[3],
                              n=best[4], confidence=_conf(best[1])),
                # The HIGHEST bin whose observed rate clears the diagonal by more than the
                # tolerance, so the region it bounds INCLUDES that bin. It was called
                # under_confident_below, section 8 printed "below" in front of it, and the
                # sentence then offered as its second example the very bin the macro is
                # computed from. The name carries the boundary now, and
                # tests/test_paper_evidence.py checks that every bin at or under this value
                # is under-confident while the first bin above it is not.
                under_confident_at_or_below=float(max(
                    (g[1] for g in gaps if g[3] > 0.02), default=0.0)),
                over_confident_band=[
                    float(min((g[1] for g in gaps if g[3] < -0.02), default=0.0)),
                    float(max((g[1] for g in gaps if g[3] < -0.02), default=0.0))],
                over_confident_band_tolerance=0.02,
                # The lowest predicted probability at which the curve is at or below the
                # diagonal at all. The band above is where the shortfall clears the
                # tolerance; this is where it starts, and the two are not the same number.
                first_at_or_below_diagonal_p=float(min(
                    (g[1] for g in gaps if g[3] <= 0.0), default=1.0)),
                n_bins_below_diagonal=int(sum(1 for g in gaps if g[3] < 0.0)),
                interior_share_of_ece_pct=float(
                    sum(abs(g[3]) * g[4] for g in gaps if 0.0 < g[1] < interior_hi)
                    / total_w * 100) if total_w else None,
                # Named so the prose cannot print the share without printing what it
                # counts: a predicted 0.05 is interior on this definition and confident on
                # the routing one, and a reader has to be able to see that.
                interior_definition_hi=interior_hi,
                n_tied_at_zero=n_at_zero,
                n_tied_at_zero_in_lowest_bin=int(sum(b["n"] for b in lowest)),
                note="signed gap is observed minus predicted, so positive means the model "
                     "is under-confident there. The claim 'miscalibrated at the confident "
                     "ends' is false as a statement about predicted probability and true "
                     "as one about max(p, 1-p): the two widest gaps are at predicted 0.05 "
                     "and 0.29, and the first of those is a confident prediction. This "
                     "block exists so the prose has to cite where the deviation is rather "
                     "than assert where it feels.",
            )
        else:
            c["calibration"] = dict(
                probability_provenance="none",
                # No internal document is named here. This string is plotted inside
                # figure F4 and printed in section 8, where it stands as the paper's
                # authority for a factual claim about a vendor's API, and a reader of the
                # released package cannot obtain the project's errata register. The
                # statement is the evidence; the pointer was not.
                note="the provider exposes no token logprobs for this model, so this "
                     "configuration cannot supply a routing signal at all",
            )
        if arm == "minicheck":
            c["excluded"] = True
            c["exclusion_reason"] = MINICHECK_EXCLUSION
        if arm == "astra_high":
            c["reasoning_effort"] = "high"
            c["max_completion_tokens"] = 4096
        if arm == "astra":
            c["reasoning_effort"] = "low"
        configs[arm] = c

    # ---- paired differences -----------------------------------------------------
    def pair_stats(a: str, b: str) -> dict:
        ca = df[a].to_numpy() == df.gold.to_numpy()
        cb = df[b].to_numpy() == df.gold.to_numpy()
        a_only, b_only = int((ca & ~cb).sum()), int((cb & ~ca).sum())
        both, neither = int((ca & cb).sum()), int((~ca & ~cb).sum())
        mc = mcnemar([[both, a_only], [b_only, neither]], exact=True)
        d = boot[a] - boot[b]
        dlo, dhi = pct_ci(d)
        return dict(
            agreement=float((df[a].to_numpy() == df[b].to_numpy()).mean()),
            disagreement=float((df[a].to_numpy() != df[b].to_numpy()).mean()),
            a_only_correct=a_only, b_only_correct=b_only,
            both_correct=both, neither_correct=neither,
            macro_bacc_delta_pp=float((configs[a]["macro_bacc"] - configs[b]["macro_bacc"]) * 100),
            macro_bacc_delta_boot_mean_pp=float(d.mean() * 100),
            macro_bacc_delta_ci_pp=[dlo * 100, dhi * 100],
            mcnemar_p=float(mc.pvalue), mcnemar_statistic=float(mc.statistic),
            mcnemar_tests="RAW per-item accuracy pooled over datasets, NOT the macro-averaged "
                          "balanced accuracy reported in macro_bacc_delta_pp",
            raw_accuracy_favors=(a if a_only > b_only else b if b_only > a_only else "tie"),
        )

    pairs = {"jev|astra": pair_stats("jev", "astra"),
             "jev|astra_high": pair_stats("jev", "astra_high"),
             "astra|astra_high": pair_stats("astra", "astra_high")}
    if minicheck_ok:
        pairs["jev|minicheck"] = pair_stats("jev", "minicheck")
        pairs["astra|minicheck"] = pair_stats("astra", "minicheck")

    # ---- operational view (CLAUDE.md rule 6) ------------------------------------
    # Any example a model failed on is scored WRONG for that model, never dropped.
    jev_raw, astra_raw = loaded["jev"], loaded["astra"]
    op_rows = []
    for eid, j in jev_raw.items():
        a = astra_raw.get(eid)
        if a is None or eid in meta["ambiguous"] or eid not in meta["gold"]:
            continue
        g = meta["gold"][eid]
        op_rows.append(dict(
            ds=j["dataset"], gold=g,
            jev=(1 if j["label"] == "SUPPORTED" else 0) if j["status"] == "ok" else 1 - g,
            astra=(1 if a["label"] == "SUPPORTED" else 0) if a["status"] == "ok" else 1 - g,
        ))
    op = pd.DataFrame(op_rows)
    operational = dict(n=int(len(op)),
                       jev_macro_bacc=macro_bacc(op, "jev"),
                       astra_macro_bacc=macro_bacc(op, "astra"),
                       note="model-caused failures scored as wrong; identical to the "
                            "scored view here because no run had a model-caused failure")

    # ---- gating sweep: Jev alone, abstaining when unsure -------------------------
    gating = []
    for t in THRESHOLDS:
        conf = cascade_conf(df, t)
        acc = df[conf]
        if len(acc) == 0:
            continue
        per, dropped = [], []
        for name, x in acc.groupby("ds"):
            b = bacc_fast(x.gold.to_numpy(), x["jev"].to_numpy())
            (per if b is not None else dropped).append(b if b is not None else name)
        g_l, p_l = acc.gold.tolist(), acc.jev.tolist()
        # A tight gate can leave a dataset with no claims of one gold class, at which
        # point balanced accuracy is undefined there and the dataset leaves the macro
        # average. That changes the denominator, so the denominator is reported.
        gating.append(dict(
            threshold=t, coverage=float(len(acc) / len(df)), n=int(len(acc)),
            macro_bacc=float(np.mean(per)) if per else None,
            n_datasets_scored=len(per),
            n_datasets_dropped=len(dropped),
            datasets_dropped=sorted(dropped),
            pooled_balanced_accuracy=M.balanced_accuracy(g_l, p_l) if len(set(g_l)) > 1 else None,
            verified_precision=M.verified_precision(g_l, p_l),
            false_verification_rate=M.false_verification_rate(g_l, p_l),
        ))

    # ---- cascade: in sample -----------------------------------------------------
    gold_a, ds_a = df.gold.to_numpy(), df.ds.to_numpy()
    astra_a = df.astra.to_numpy()
    # The guard metric on every row of this sweep, in claim COUNTS and with the same
    # exact McNemar the paper demands of every other false-verification contrast. A
    # rounded rate hides how few claims a difference rests on: at the tightest threshold
    # the cascade and the frontier arm are one claim apart, which reads as half a point of
    # FVR and separates from nothing. Section 9.3 argues that a claim two systems differ,
    # or do not, needs a test; a row of this table is a point estimate someone will quote,
    # so it carries the counts and the test alongside the rate.
    neg_m = gold_a == 0
    astra_fv_m = neg_m & (astra_a == 1)
    astra_bacc_in = macro_bacc_arr(gold_a, astra_a, ds_a)
    in_sample = []
    for t in THRESHOLDS:
        pred, conf = cascade_pred(df, t)
        cas_fv_m = neg_m & (pred == 1)
        cas_only = int((cas_fv_m & ~astra_fv_m).sum())
        astra_only = int((astra_fv_m & ~cas_fv_m).sum())
        both = int((cas_fv_m & astra_fv_m).sum())
        neither = int((neg_m & ~cas_fv_m & ~astra_fv_m).sum())
        mc_fv = mcnemar([[both, cas_only], [astra_only, neither]], exact=True)
        bacc_t = macro_bacc_arr(gold_a, pred, ds_a)
        in_sample.append(dict(
            threshold=t, macro_bacc=bacc_t,
            escalated=float((~conf).mean()), cost_per_1k_usd=cascade_cost_per_1k(conf),
            false_verification_rate=M.false_verification_rate(gold_l, pred.tolist()),
            verified_precision=M.verified_precision(gold_l, pred.tolist()),
            n_gold_unsupported=int(neg_m.sum()),
            n_false_verified=int(cas_fv_m.sum()),
            astra_n_false_verified=int(astra_fv_m.sum()),
            fv_vs_astra_cascade_only=cas_only,
            fv_vs_astra_astra_only=astra_only,
            fv_vs_astra_mcnemar_p=float(mc_fv.pvalue),
            macro_bacc_minus_astra_pp=float((bacc_t - astra_bacc_in) * 100),
        ))
    best_in = max(in_sample, key=lambda r: r["macro_bacc"])

    # ---- cascade: cross fitted --------------------------------------------------
    xf = crossfit_cascade(df, cascade_cost_per_1k, seed=seed)
    oos_b, oos_e, oos_c = xf.bacc, xf.escalated, xf.cost
    oos_vs_astra, picked, oos_fvr, oos_vp = xf.vs_astra_pp, xf.picked, xf.fvr, xf.verified_precision
    cf_lo, cf_hi = pct_ci(oos_b)
    fvr_lo, fvr_hi = pct_ci(oos_fvr)
    crossfit = dict(
        n_repeats=N_REPEATS, k_folds=K_FOLDS, seed=seed, thresholds=list(THRESHOLDS),
        macro_bacc_mean=float(oos_b.mean()), macro_bacc_ci=[cf_lo, cf_hi],
        escalated_mean=float(oos_e.mean()), cost_per_1k_usd_mean=float(oos_c.mean()),
        minus_astra_pp_mean=float(oos_vs_astra.mean()),
        minus_astra_pp_ci=[float(np.percentile(oos_vs_astra, 2.5)),
                           float(np.percentile(oos_vs_astra, 97.5))],
        minus_astra_interval_note="spread over cross-fitting repeats only: it carries the "
                                  "threshold-selection noise, not the sampling uncertainty of the "
                                  "495 examples, so it is not a confidence interval on the "
                                  "population difference. The confidence interval and the test "
                                  "for this comparison are in vs_astra_inference.",
        vs_astra_inference=cascade_vs_arm_inference(df, xf.oof_preds, n_boot, seed, "astra"),
        # The comparison a cascade paper is tempted to leave out. The cascade costs many
        # times what Jev alone costs, so a reader deciding whether to build one needs the
        # cascade measured against the cheap system with the same instruments it is
        # measured against the expensive one.
        vs_jev_inference=cascade_vs_arm_inference(df, xf.oof_preds, n_boot, seed, "jev"),
        # The neighbour the cascade sits beside in Table 4 and lost to. The high-effort
        # frontier arm is the row directly under the cross-fitted cascade there, it is
        # ahead of it on accuracy and well ahead on the guard metric, and for one draft the
        # only prose about the two of them together explained a rounding coincidence in the
        # caption. A table that prints a neighbour prints the difference against it, with
        # the same bootstrap and the same exact McNemar as the other two neighbours.
        vs_astra_high_inference=cascade_vs_arm_inference(
            df, xf.oof_preds, n_boot, seed, "astra_high"),
        # The same comparison against the SUPERSEDED high-effort run. The prompt
        # correction moved one verdict on a gold-unsupported claim on this arm, which
        # moves this arm's false-verification rate, which is one leg of this comparison,
        # which is one of the two results that survive the Holm adjustment. Section 12
        # used to certify the separated results as undisturbed by the correction on the
        # strength of the low-effort arm alone, where no flip touched an unsupported
        # claim. This recomputes the survivor against the run the correction replaced so
        # the certification is a measurement. The cascade's own predictions do not depend
        # on this arm, so only the comparator column moves.
        vs_astra_high_pre_parity_inference=cascade_vs_arm_inference(
            df.assign(astra_high=arm_column_from_run(df, REPO / PRE_PARITY_ASTRA_HIGH)),
            xf.oof_preds, n_boot, seed, "astra_high"),
        cost_ratio_crossfit_over_jev=float(oos_c.mean() / jev_cost),
        minus_jev_pp_mean=float((oos_b.mean() - configs["jev"]["macro_bacc"]) * 100),
        minus_jev_fvr_pp_mean=float(
            (float(np.array(oos_fvr).mean()) - configs["jev"]["false_verification_rate"]) * 100),
        threshold_selection_counts={str(t): int((np.array(picked) == t).sum()) for t in THRESHOLDS},
        n_threshold_fits=len(picked),
        selection_optimism_pp=float((best_in["macro_bacc"] - oos_b.mean()) * 100),
        # Monte Carlo error of the MEANS this paper prints. Every cross-fitted headline is
        # one draw of N_REPEATS fold assignments at one seed, and the spread over repeats
        # says nothing about how precisely the MEAN of that spread is pinned down. These
        # are the across-repeat standard errors, so a reader can see that the cost mean is
        # carried to about a cent and the escalation share to about a third of a point,
        # and that printing more digits than that would be false precision.
        monte_carlo_sem=dict(
            macro_bacc_pp=float(oos_b.std(ddof=1) / np.sqrt(len(oos_b)) * 100),
            cost_per_1k_usd=float(oos_c.std(ddof=1) / np.sqrt(len(oos_c))),
            escalated_pp=float(oos_e.std(ddof=1) / np.sqrt(len(oos_e)) * 100),
            fvr_pp=float(np.array(oos_fvr).std(ddof=1) / np.sqrt(len(oos_fvr)) * 100),
            note="standard error of the mean ACROSS cross-fitting repeats, at this seed. It "
                 "bounds how much each printed cross-fitted mean would move under a "
                 "different draw of the fold assignment. It is not sampling uncertainty "
                 "over the claims; that is in vs_astra_inference and vs_jev_inference.",
        ),
        # Why the top of macro_bacc_ci sits exactly on the in-sample best. It is not a
        # property of the procedure: it holds only while more than 2.5% of repeats pick the
        # in-sample threshold in every fold and so reconstruct the in-sample cascade
        # exactly. At this seed that share is n_repeats_reproducing_in_sample / n_repeats.
        # Nothing here pins that share where it is, and nothing in this package varies the
        # seed to find out where it goes, so the paper reports the count for this draw of
        # folds and makes no claim about other seeds. An earlier version of this comment
        # asserted that neighbouring seeds drop below 2.5%; that was never measured, and a
        # reviewer who did measure it found the opposite at most of the seeds tried.
        n_repeats_reproducing_in_sample=int(
            np.isclose(oos_b, best_in["macro_bacc"], rtol=0, atol=1e-12).sum()),
        cost_ratio_astra_over_crossfit=float(astra_cost / float(oos_c.mean())),
        false_verification_rate_mean=float(oos_fvr.mean()),
        false_verification_rate_ci=[fvr_lo, fvr_hi],
        verified_precision_mean=float(oos_vp.mean()),
        minus_astra_fvr_pp_mean=float((oos_fvr.mean()
                                       - configs["astra"]["false_verification_rate"]) * 100),
        guard_note="the cross-fitted cascade is a MORE permissive guard than Astra alone: "
                   "it inherits Jev's verdict on the claims it does not escalate, and Jev "
                   "false-verifies at a higher rate. The cascade's contribution is cost at "
                   "matched accuracy, and this is what that costs on the guard metric.",
    )

    # ---- controls: is the SIGNAL doing the work, or just the extra escalation? ---
    # Confidence gating has to be compared against spending the same escalation budget at
    # random. Without this control an apparent cascade gain could be nothing but extra
    # traffic sent to the stronger model.
    jev_a, astra_a = df.jev.to_numpy(), df.astra.to_numpy()
    orc_conf = jev_a == gold_a
    orc_pred = np.where(orc_conf, jev_a, astra_a)
    oracle = dict(
        macro_bacc=macro_bacc_arr(gold_a, orc_pred, ds_a),
        escalated=float((~orc_conf).mean()),
        cost_per_1k_usd=cascade_cost_per_1k(orc_conf),
        false_verification_rate=M.false_verification_rate(gold_l, orc_pred.tolist()),
        verified_precision=M.verified_precision(gold_l, orc_pred.tolist()),
        headroom_over_jev_pp=float((macro_bacc_arr(gold_a, orc_pred, ds_a)
                                    - configs["jev"]["macro_bacc"]) * 100),
        note="escalate exactly the claims Jev gets wrong: an upper bound no router can pass, "
             "not an achievable operating point",
    )
    controls = []
    for g in in_sample:
        rate = g["escalated"]
        if rate == 0.0:
            continue
        rng_c = np.random.default_rng(seed)
        rand_b, rand_c = [], []
        for _ in range(N_REPEATS):
            conf = rng_c.random(len(df)) >= rate     # escalate a random `rate` share
            rand_b.append(macro_bacc_arr(gold_a, np.where(conf, jev_a, astra_a), ds_a))
            rand_c.append(cascade_cost_per_1k(conf))
        rand_b = np.array(rand_b)
        controls.append(dict(
            threshold=g["threshold"], escalated=rate,
            confidence_macro_bacc=g["macro_bacc"],
            random_macro_bacc_mean=float(rand_b.mean()),
            random_macro_bacc_ci=[float(np.percentile(rand_b, 2.5)), float(np.percentile(rand_b, 97.5))],
            random_cost_per_1k_usd_mean=float(np.mean(rand_c)),
            confidence_minus_random_pp=float((g["macro_bacc"] - rand_b.mean()) * 100),
            clears_random_ci=bool(g["macro_bacc"] > float(np.percentile(rand_b, 97.5))),
        ))

    # ---- complementarity headline ------------------------------------------------
    ja = pairs["jev|astra"]
    complementarity = dict(
        agreement=ja["agreement"], disagreement=ja["disagreement"],
        jev_only_correct=ja["a_only_correct"], astra_only_correct=ja["b_only_correct"],
        both_correct=ja["both_correct"], neither_correct=ja["neither_correct"],
        datasets_jev_higher=int(sum(1 for d in configs["jev"]["per_dataset"]
                                    if configs["jev"]["per_dataset"][d] > configs["astra"]["per_dataset"][d])),
        datasets_astra_higher=int(sum(1 for d in configs["jev"]["per_dataset"]
                                      if configs["astra"]["per_dataset"][d] > configs["jev"]["per_dataset"][d])),
        fvr_gap_pp=float((configs["jev"]["false_verification_rate"]
                          - configs["astra"]["false_verification_rate"]) * 100),
        fvr_gap_inference=fvr_gap_inference(df, "jev", "astra", n_boot, seed),
        per_dataset_concentration=per_dataset_concentration(configs, "jev", "astra"),
        error_correlation=error_correlation(df, "jev", "astra"),
    )

    # ---- MiniCheck decision-threshold sweep (EXCLUDED arm, disclosure only) --------
    # The exclusion rests on "the scores do not separate", and the obvious objection is
    # that we simply used the wrong cut point. So sweep every cut point its own scores
    # allow and report the best one: if the ceiling over the whole sweep is still far
    # below the other arms, the cut point was never the problem.
    minicheck_sweep = None
    if minicheck_ok:
        p_mc = df.minicheck_p.to_numpy()
        cuts = sorted(set(np.round(np.unique(p_mc), 6).tolist()) | {0.5})
        swept = []
        for cut in cuts:
            b = macro_bacc_arr(gold_a, (p_mc >= cut).astype(int), ds_a)
            if b is not None:
                swept.append((b, float(cut)))
        best_b, best_cut = max(swept)
        minicheck_sweep = dict(
            n_cutpoints=len(swept),
            default_cut=0.5,
            default_macro_bacc=configs["minicheck"]["macro_bacc"],
            best_cut=best_cut,
            best_macro_bacc=best_b,
            headroom_from_retuning_pp=float((best_b - configs["minicheck"]["macro_bacc"]) * 100),
            gap_to_jev_at_best_cut_pp=float((configs["jev"]["macro_bacc"] - best_b) * 100),
            note="EXCLUDED arm. Retuning the cut point is the best case for our port and it "
                 "still does not reach the other configurations, so the exclusion is not a "
                 "threshold artifact.",
        )
        configs["minicheck"]["threshold_sweep"] = minicheck_sweep

        # C4 in docs/explanation/threats-to-validity.md designates MiniCheck as the
        # EXTERNAL ANCHOR for harness validity, so the score it was supposed to hit is
        # evidence, not trivia. It is read from the captured leaderboard manifest, never
        # typed here. Note the two are not the same measurement: the published figure is
        # a macro average over the full TEST sets, ours is 495 dev claims.
        lb = json.loads((REPO / LEADERBOARD_MANIFEST).read_text())
        pub = float(lb["models"]["MiniCheck-Flan-T5-L"]["scores"][
            lb["columns"].index("Average")]) / 100.0
        configs["minicheck"]["published_reference"] = dict(
            manifest=LEADERBOARD_MANIFEST,
            source=lb["source"],
            captured_utc=lb["captured_utc"],
            model_row="MiniCheck-Flan-T5-L",
            split=lb["split"],
            macro_bacc=pub,
            our_macro_bacc=configs["minicheck"]["macro_bacc"],
            shortfall_pp=float((pub - configs["minicheck"]["macro_bacc"]) * 100),
            harness_validation_target=lb["harness_validation_target"],
            comparability_note="the published figure is a macro average over the full "
                               "LLM-AggreFact test sets; ours is the same model scored on "
                               "the 495 dev claims of this study, so a gap of a few points "
                               "would be a split effect. A gap of this size is not.",
            role="external harness-validity anchor (threats-to-validity C4): the arm exists "
                 "to show that the loader, the label join and the balanced-accuracy code "
                 "reproduce a published number. It did not, and the arm is excluded.",
        )

    # The same captured leaderboard carries rows for systems this study did not run, and
    # three of them sit above every configuration we report. Quoting the MiniCheck row as
    # a harness anchor while passing over the rest would be exactly the selective reading
    # this paper spends a section objecting to, so every row in the manifest is emitted.
    lb_all = json.loads((REPO / LEADERBOARD_MANIFEST).read_text())
    _avg = lb_all["columns"].index("Average")
    _unnamed = set(lb_all["models"]) - set(_LEADERBOARD_MACRO)
    if _unnamed:
        raise SystemExit(
            f"REFUSING to write numbers.json: leaderboard rows {sorted(_unnamed)} have no "
            "macro stem in _LEADERBOARD_MACRO. A row that exists in the evidence and not "
            "in the paper is the selective reading this paper objects to."
        )
    published_leaderboard = dict(
        manifest=LEADERBOARD_MANIFEST,
        source=lb_all["source"],
        captured_utc=lb_all["captured_utc"],
        split=lb_all["split"],
        rows={name: float(row["scores"][_avg]) / 100.0
              for name, row in lb_all["models"].items()},
        comparability_note="These are macro balanced accuracies over the FULL LLM-AggreFact "
                           "TEST sets, reported by their authors under their own prompts and "
                           "harnesses. Every number this paper reports is a macro over a "
                           "495-claim DEV sample under one shared prompt. The two are not on "
                           "the same footing and no difference between them is a measurement "
                           "of anything. They are printed so a reader has an external scale "
                           "for the task rather than only the two systems we happened to run.",
        # The manifest is a PARTIAL capture, taken to pin the MiniCheck harness anchor.
        # The live leaderboard lists many more systems, at scores that fall between the
        # captured rows and below the lowest of them; the highest captured row is also the
        # highest on the leaderboard, so "spanning scores above" would have been false.
        # Printing these four rows as though they were the field would be the selective
        # reading this paper objects to, one level up, so the scope is recorded here and
        # the paper states it.
        capture_scope="PARTIAL. This file was captured to pin the published MiniCheck-Flan-T5-L "
                      "row as the harness-validity anchor of threats-to-validity C4; the other "
                      "rows were kept as context and were not selected by any stated rule. The "
                      "public leaderboard at the source URL lists many more systems, at scores "
                      "that fall between the rows captured here and below the lowest of them. "
                      "Nothing in this paper may describe these four rows as the field, or as "
                      "bounding it.",
    )

    # ---- per-dataset support ------------------------------------------------------
    # The macro headline weights every dataset equally, so the thinnest minority class
    # decides how much any one of those eleven numbers can be trusted. On this 495-item
    # subsample RAGTruth carries four negatives, which is the honest answer to "your best
    # Jev dataset, how many negatives is that resting on?" and section 12 has to say it.
    support = {}
    for d, t in df.groupby("ds"):
        npos, nneg = int(t.gold.sum()), int((t.gold == 0).sum())
        support[d] = dict(n=int(len(t)), n_supported=npos, n_unsupported=nneg,
                          min_class=min(npos, nneg))
    thinnest = min(support.items(), key=lambda kv: kv[1]["min_class"])
    support_summary = dict(
        per_dataset=support,
        thinnest_dataset=thinnest[0],
        thinnest_minority_class_n=thinnest[1]["min_class"],
        note="every dataset keeps both gold classes in the full 495, so macro balanced "
             "accuracy is defined everywhere. A single held-out cross-fitting fold can "
             "still lose the thinnest minority class, which is why the cross-fitted "
             "cascade is scored over the assembled out-of-fold predictions rather than "
             "fold by fold.",
    )

    # ---- provenance --------------------------------------------------------------
    file_hashes = {str(p.relative_to(REPO)): sha256_file(p) for p in RUN_FILES.values()}
    dev = meta["dev"]
    provenance = dict(
        dataset="lytang/LLM-AggreFact dev split",
        parquet_sha256=dev.parquet_sha256,
        runs={k: str(p.relative_to(REPO)) for k, p in RUN_FILES.items()},
        prediction_file_sha256=file_hashes,
        pricing_snapshot="configs/pricing_2026-09-20.yaml",
        config="configs/benchmark.yaml",
        sample_id_list=dict(
            path=SAMPLE_ID_LIST,
            sha256=sha256_file(REPO / SAMPLE_ID_LIST),
            n=sum(1 for line in (REPO / SAMPLE_ID_LIST).read_text().splitlines() if line.strip()),
            note="the example IDs every run script reads. The draw itself predates this "
                 "repository's gated pipeline and is not regenerated by any code here, so "
                 "the list is released and hashed as data. Its per-dataset composition and "
                 "both gold classes per dataset are checkable from the list alone.",
        ),
        leaderboard_manifest=LEADERBOARD_MANIFEST,
        generator="scripts/verify/paired_analysis.py",
        seed=seed,
        bootstrap_resamples=n_boot,
    )

    out = dict(
        schema_version=2,
        provenance=provenance,
        jev_rerun=jev_rerun_stability(df, REPO / SAMPLE_ID_LIST),
        prompt_parity=prompt_parity_sensitivity(
            df, REPO / PRE_PARITY_ASTRA),
        prompt_parity_high=prompt_parity_sensitivity(
            df, REPO / PRE_PARITY_ASTRA_HIGH, post_col="astra_high"),
        same_prompt_repeat=same_prompt_repeat(
            df, REPO / CAPPED_ASTRA, REPO / PRE_PARITY_ASTRA, RUN_FILES["astra"]),
        n_paired=int(len(df)), n_datasets=int(df.ds.nunique()),
        n_gold_supported=int(df.gold.sum()), n_gold_unsupported=int((df.gold == 0).sum()),
        gold_supported_rate=float(df.gold.mean()),
        valid_rate={k: float(sum(1 for r in v.values() if r.get("status") == "ok") / len(v))
                    for k, v in loaded.items()},
        rows_read={k: len(v) for k, v in loaded.items()},
        dataset_support=support_summary,
        minicheck_threshold_sweep=minicheck_sweep,
        configurations=configs,
        excluded_arms=(["minicheck"] if minicheck_ok else []),
        minicheck_exclusion=MINICHECK_EXCLUSION,
        minicheck_scored=minicheck_ok,
        minicheck_missing_rows=meta["minicheck_missing"],
        published_leaderboard=published_leaderboard,
        pairs=pairs,
        complementarity=complementarity,
        operational=operational,
        gating=gating,
        cascade=dict(in_sample_sweep=in_sample, in_sample_best=best_in,
                     crossfit=crossfit, random_control=controls, oracle=oracle),
        ratios=dict(cost_astra_over_jev=float(astra_cost / jev_cost),
                    cost_astra_high_over_jev=float(astra_high_cost / jev_cost),
                    cost_astra_over_crossfit_cascade=crossfit["cost_ratio_astra_over_crossfit"]),
        cost_decomposition=cost_decomposition,
        latency_note="Both bulk runs used concurrency 12, so per-request latency includes "
                     "client-side queueing. These are descriptive, NOT the controlled us-west-2 "
                     "measurement of spec section 21, and no latency ratio is reported "
                     "(CLAUDE.md rule 10).",
    )

    # Computed last, because it reads the p-values every other block produced.
    out["multiplicity"] = holm_bonferroni(out)

    write_if_changed(ARXIV_JSON, json.dumps(out, indent=2, sort_keys=True, default=float) + "\n")

    write_tex(out)
    write_hashes_tex(out)
    write_compat_views(out, df)
    write_evidence(file_hashes, out)
    print_summary(out)
    return 0


# --------------------------------------------------------------------------------
def write_tex(out: dict) -> None:
    m = Macros()
    NAME = {"jev": "Jev", "astra": "Astra", "astra_high": "AstraHigh", "minicheck": "MiniCheck"}
    cfg = out["configurations"]

    m.section("sample")
    m.add("NPaired", out["n_paired"], "{:d}")
    m.add("NDatasets", out["n_datasets"], "{:d}")
    m.add("NGoldSupported", out["n_gold_supported"], "{:d}")
    m.add("NGoldUnsupported", out["n_gold_unsupported"], "{:d}")
    m.add("GoldSupportedRatePct", out["gold_supported_rate"] * 100)
    m.add("BootstrapResamples", out["provenance"]["bootstrap_resamples"], "{:d}")

    m.section("per dataset support (the thinnest minority class bounds the macro headline)")
    for d, s in sorted(out["dataset_support"]["per_dataset"].items()):
        m.add(f"Support{macro_token(d)}N", s["n"], "{:d}")
        m.add(f"Support{macro_token(d)}Pos", s["n_supported"], "{:d}")
        m.add(f"Support{macro_token(d)}Neg", s["n_unsupported"], "{:d}")
    m.add("ThinnestMinorityClassN", out["dataset_support"]["thinnest_minority_class_n"], "{:d}")

    for arm, label in NAME.items():
        if arm not in cfg:
            continue
        c = cfg[arm]
        m.section(f"{label}" + ("  (EXCLUDED ARM: disclosure only, never a result)"
                                if c.get("excluded") else ""))
        m.add(f"{label}BAcc", c["macro_bacc"] * 100)
        # A two decimal form of every headline score, for the sentences and tables that
        # print a DIFFERENCE beside its two operands. At one decimal the printed operands
        # do not reproduce the printed difference (73.8 minus 73.3 is 0.5, the difference
        # is 0.56), and a reader with a pencil is entitled to reproduce it.
        # Wherever the paper prints a difference in the same paragraph as both of its
        # operands, scripts/verify/check_paper_macros.py checks that the three printed
        # numbers subtract exactly, at the difference's own precision.
        m.add(f"{label}BAccTwoDP", c["macro_bacc"] * 100, "{:.2f}")
        m.add(f"{label}BAccCILo", c["macro_bacc_ci"][0] * 100)
        m.add(f"{label}BAccCIHi", c["macro_bacc_ci"][1] * 100)
        # Derived from the two bounds immediately above so that prose describing how wide
        # this interval is cites a macro rather than an adjective rounded by hand. Section
        # 12 bounds a possible split effect on the MiniCheck anchor by this half-width,
        # which is the only place a typed "about five points" used to stand.
        m.add(f"{label}BAccCIHalfWidthPP",
              (c["macro_bacc_ci"][1] - c["macro_bacc_ci"][0]) * 100 / 2)
        m.add(f"{label}RawAcc", c["raw_accuracy"] * 100)
        m.add(f"{label}PooledBAcc", c["pooled_balanced_accuracy"] * 100)
        m.add(f"{label}FVR", c["false_verification_rate"] * 100)
        # Two decimal form, for the sentences that print this rate beside a DIFFERENCE of
        # two such rates. See the triple table in scripts/verify/check_paper_macros.py.
        m.add(f"{label}FVRTwoDP", c["false_verification_rate"] * 100, "{:.2f}")
        m.add(f"{label}FVRCount", c["n_false_verified"], "{:d}")
        m.add(f"{label}VerifiedPrecision", c["verified_precision"] * 100)
        m.add(f"{label}MacroFOne", c["macro_f1"] * 100)
        m.add(f"{label}MCC", c["mcc"], "{:.3f}")
        m.add(f"{label}PredSupportedRate", c["pred_supported_rate"] * 100)
        # Five decimals for the decision model, because every cost RATIO in the paper is
        # printed beside this figure and the frontier figure. At three decimals a reader
        # dividing 8.93 by 0.048 gets 186 where the paper prints 187, which is the
        # rounding of the denominator and not a disagreement about the measurement.
        m.add(f"{label}CostPerK", c["cost_per_1k_usd"], "{:.5f}" if arm == "jev" else "{:.2f}")
        m.add(f"{label}ValidRatePct", out["valid_rate"][arm] * 100)
        # The macro NAME carries the caveat. These come from concurrency-12 bulk runs,
        # not the controlled protocol of spec section 21, so rule 10 forbids quoting them
        # as a speed comparison. A macro called \\JevLatencyPFifty is trivial to drop into
        # a sentence without its caveat; one called \\JevUncontrolledLatencyPFifty is not.
        for k, word in (("p50", "PFifty"), ("p95", "PNinetyFive"), ("p99", "PNinetyNine")):
            m.add(f"{label}UncontrolledLatency{word}", c["latency_ms"][k], "{:,.0f}")
        if "mean_input_tokens" in c:
            m.add(f"{label}MeanInputTokens", c["mean_input_tokens"], "{:,.0f}")
            m.add(f"{label}MeanOutputTokens", c["mean_output_tokens"], "{:,.1f}")
            m.add(f"{label}MaxInputTokens", c["max_input_tokens"], "{:,.0f}")
            m.add(f"{label}PNinetyFiveInputTokens", c["p95_input_tokens"], "{:,.0f}")
            # The spread of the output-token count, which is what distinguishes a fixed
            # typed envelope from prose. Table 2's caption reconciles its "no text" column
            # with its token column out of these three.
            m.add(f"{label}MinOutputTokens", c["min_output_tokens"], "{:,.0f}")
            m.add(f"{label}MaxOutputTokens", c["max_output_tokens"], "{:,.0f}")
            m.add(f"{label}DistinctOutputTokenCounts", c["n_distinct_output_tokens"], "{:d}")
        if "reasoning_tokens" in c:
            rt = c["reasoning_tokens"]
            m.add(f"{label}ReasoningMedian", rt["median"], "{:.0f}")
            m.add(f"{label}ReasoningMean", rt["mean"], "{:.1f}")
            m.add(f"{label}ReasoningMax", rt["max"], "{:d}")
            m.add(f"{label}ReasoningZeroN", rt["n_zero"], "{:d}")
        cal = c["calibration"]
        if cal["probability_provenance"] == "native":
            m.add(f"{label}Brier", cal["brier"], "{:.3f}")
            m.add(f"{label}NLL", cal["negative_log_loss"], "{:.3f}")
            # The NLL alone reads as a property of the model. It is not: with a
            # probability that saturates, the items the verifier is confidently wrong
            # about are charged the clipping constant, and section 8 prints how much of
            # the figure that is so a reader can price the rest.
            clip = cal["logloss_clip"]
            m.add(f"{label}NLLClipN", clip["n"], "{:d}")
            m.add(f"{label}NLLClipSharePct", clip["share"] * 100, "{:.0f}")
        if "shape" in cal:
            sh = cal["shape"]
            w0, w1 = sh["worst_bins"]
            m.add(f"{label}CalWorstP", w0["mean_p"], "{:.2f}")
            m.add(f"{label}CalWorstObserved", w0["observed"], "{:.2f}")
            m.add(f"{label}CalSecondP", w1["mean_p"], "{:.2f}")
            m.add(f"{label}CalSecondObserved", w1["observed"], "{:.2f}")
            m.add(f"{label}CalUnderConfidentAtOrBelow",
                  sh["under_confident_at_or_below"], "{:.2f}")
            m.add(f"{label}CalOverConfidentLo", sh["over_confident_band"][0], "{:.2f}")
            m.add(f"{label}CalOverConfidentHi", sh["over_confident_band"][1], "{:.2f}")
            m.add(f"{label}CalInteriorShareOfECEPct", sh["interior_share_of_ece_pct"], "{:.0f}")
            # What "interior" counts. The share is a fact about predicted probability and
            # section 8 may not print it without the bound, because on the routing
            # definition of confidence a predicted 0.05 is interior AND confident.
            m.add(f"{label}CalInteriorHi", sh["interior_definition_hi"], "{:.2f}")
            # The widest gap, priced in the confidence the cascade routes by rather than in
            # the predicted probability, and the flattest bin beside it.
            m.add(f"{label}CalWorstConfidence", sh["worst_bins"][0]["confidence"], "{:.2f}")
            m.add(f"{label}CalBestP", sh["best_bin"]["mean_p"], "{:.2f}")
            m.add(f"{label}CalBestConfidence", sh["best_bin"]["confidence"], "{:.2f}")
            m.add(f"{label}CalFirstAtOrBelowP", sh["first_at_or_below_diagonal_p"], "{:.2f}")
            m.add(f"{label}CalBinsBelowDiagonalN", sh["n_bins_below_diagonal"], "{:d}")
            # The same count as {label}ProbAtZeroN below, by construction: both are the
            # predictions the vendor reports as exactly 0.00. They were two different
            # numbers for one quantity until the shape block stopped deriving this from
            # bin means, and tests/test_paper_evidence.py now holds them equal.
            m.add(f"{label}CalTiedAtZeroN", sh["n_tied_at_zero"], "{:d}")
            m.add(f"{label}CalZeroInLowestBinN", sh["n_tied_at_zero_in_lowest_bin"], "{:d}")
            m.add(f"{label}NLLClipNatsEach", clip["nats_each"], "{:.1f}")
            add_scientific(m, f"{label}NLLClipEps", clip["eps"])
            # What "certain" means, in the vendor's own resolution. A reported 0.00 is a
            # probability below half a step, not a zero, so the -log(eps) charged against
            # it is a choice of ours. The paper prints the step beside the charge.
            res = cal["probability_resolution"]
            m.add(f"{label}ProbDecimals", res["decimals"], "{:d}")
            m.add(f"{label}ProbDistinct", res["n_distinct"], "{:d}")
            m.add(f"{label}ProbAtZeroN", res["n_at_zero"], "{:d}")
            m.add(f"{label}ProbAtOneN", res["n_at_one"], "{:d}")
            m.add(f"{label}ProbStep", res["step"], f"{{:.{res['decimals']}f}}")
            m.add(f"{label}ProbHalfStep", res["half_step"], f"{{:.{res['decimals'] + 1}f}}")
            # And the sensitivity itself: the same predictions scored at a clip the size
            # of that resolution, the reference a constant one-half forecast scores, and
            # the clip at which the two cross.
            sens = cal["logloss_eps_sensitivity"]
            m.add(f"{label}NLLConstantHalf", sens["reference_nll"], "{:.3f}")
            m.add(f"{label}BrierConstantHalf", cal["constant_half_brier"], "{:.3f}")
            by_eps = {row["eps"]: row["nll"] for row in sens["grid"]}
            m.add(f"{label}NLLAtHalfStepClip", by_eps[res["half_step"]], "{:.3f}")
            for eps_value, token in ((1e-6, "Micro"), (1e-3, "Milli")):
                match = [v for k, v in by_eps.items() if abs(k / eps_value - 1.0) < 1e-9]
                if match:
                    m.add(f"{label}NLLAtEps{token}", match[0], "{:.3f}")
                    # The constant itself, so the appendix table prints a macro in the
                    # column that names the constant rather than a typed exponent.
                    add_scientific(m, f"{label}EpsGrid{token}", eps_value)
            if sens["crossover_eps"] is not None:
                add_scientific(m, f"{label}NLLCrossoverEps", sens["crossover_eps"])
            m.add(f"{label}ECE", cal["ece"], "{:.3f}")
            m.add(f"{label}ECESensitivity", cal["ece_sensitivity"], "{:.3f}")
            m.add(f"{label}AUROC", cal["auroc"], "{:.3f}")
            m.add(f"{label}AUPRC", cal["auprc"], "{:.3f}")
            m.add(f"{label}SelectiveAUROC", cal["selective_auroc"], "{:.3f}")
            m.add(f"{label}SelectiveAUPRC", cal["selective_auprc"], "{:.3f}")
        for d, v in sorted(c["per_dataset"].items()):
            m.add(f"{label}BAcc{macro_token(d)}", v * 100)
        if "published_reference" in c:
            pr = c["published_reference"]
            m.add(f"{label}PublishedTestBAcc", pr["macro_bacc"] * 100)
            m.add(f"{label}PublishedTestBAccTwoDP", pr["macro_bacc"] * 100, "{:.2f}")
            m.add(f"{label}ShortfallVsPublishedPP", pr["shortfall_pp"])
        if "threshold_sweep" in c:
            sw = c["threshold_sweep"]
            m.add(f"{label}BestCutBAcc", sw["best_macro_bacc"] * 100)
            m.add(f"{label}BestCutBAccTwoDP", sw["best_macro_bacc"] * 100, "{:.2f}")
            m.add(f"{label}BestCut", sw["best_cut"], "{:.3f}")
            m.add(f"{label}RetuningHeadroomPP", sw["headroom_from_retuning_pp"], "{:.1f}")
            m.add(f"{label}GapToJevAtBestCutPP", sw["gap_to_jev_at_best_cut_pp"], "{:.1f}")
            m.add(f"{label}CutpointsSwept", sw["n_cutpoints"], "{:d}")

    m.section("paired comparisons")
    for key, label in (("jev|astra", "JevAstra"), ("jev|astra_high", "JevAstraHigh"),
                       ("astra|astra_high", "AstraAstraHigh")):
        p = out["pairs"][key]
        m.add(f"{label}DeltaPP", abs(p["macro_bacc_delta_pp"]), "{:.1f}")
        m.add(f"{label}DeltaSignedPP", p["macro_bacc_delta_pp"], "{:.1f}")
        m.add(f"{label}DeltaSignedTwoDP", p["macro_bacc_delta_pp"], "{:.2f}")
        m.add(f"{label}DeltaCILo", p["macro_bacc_delta_ci_pp"][0], "{:.1f}")
        m.add(f"{label}DeltaCIHi", p["macro_bacc_delta_ci_pp"][1], "{:.1f}")
        m.add(f"{label}McNemarP", p["mcnemar_p"], "{:.2f}")
        m.add(f"{label}McNemarStat", p["mcnemar_statistic"], "{:.0f}")

    comp = out["complementarity"]
    m.section("complementarity")
    m.add("AgreementPct", comp["agreement"] * 100)
    m.add("DisagreementPct", comp["disagreement"] * 100)
    m.add("JevOnlyCorrect", comp["jev_only_correct"], "{:d}")
    m.add("AstraOnlyCorrect", comp["astra_only_correct"], "{:d}")
    m.add("BothCorrect", comp["both_correct"], "{:d}")
    m.add("NeitherCorrect", comp["neither_correct"], "{:d}")
    ec = comp["error_correlation"]
    m.add("ErrorKappa", ec["cohens_kappa"], "{:.2f}")
    m.add("ErrorPhi", ec["phi"], "{:.2f}")
    m.add("NeitherExpectedIndependent", ec["neither_correct_expected_if_independent"],
          "{:.1f}")  # one decimal so 66 / 21.6 rounds to the 3.1 the paper prints;
          # at "{:.0f}" it printed 22 and a reader with a pencil got 3.0
    m.add("JointFailureExcess", ec["joint_failure_excess_ratio"], "{:.1f}")
    m.add("DatasetsJevHigher", comp["datasets_jev_higher"], "{:d}")
    m.add("DatasetsAstraHigher", comp["datasets_astra_higher"], "{:d}")
    m.add("FVRGapPP", comp["fvr_gap_pp"])
    m.add("FVRGapTwoDP", comp["fvr_gap_pp"], "{:.2f}")
    fi = comp["fvr_gap_inference"]
    m.add("FVRGapCILo", fi["gap_ci_pp"][0], "{:.2f}")
    m.add("FVRGapCIHi", fi["gap_ci_pp"][1], "{:.2f}")
    # Six decimals, not five: appendix E tells the reader this raw p stays below 0.05
    # for any family smaller than FVRGapHolmBreakFamilyN, and a reader who checks that
    # multiplies the printed digits. At five decimals the printed 0.00053 x 94 lands at
    # 0.04982 and contradicts the sentence, while the value itself does not. A printed
    # digit that cannot reproduce the claim beside it is the defect, not the claim.
    m.add("FVRGapMcNemarP", fi["mcnemar_p"], "{:.6f}")
    m.add("FVRJevOnlyFalseVerifies", fi["a_only_false_verifies"], "{:d}")
    m.add("FVRAstraOnlyFalseVerifies", fi["b_only_false_verifies"], "{:d}")
    m.add("FVRBothFalseVerify", fi["both_false_verify"], "{:d}")
    # The exact McNemar for the FVR gap is computed on exactly these items, so the count
    # belongs next to the p-value rather than left for a reader to add up.
    m.add("FVRGapDiscordant",
          fi["a_only_false_verifies"] + fi["b_only_false_verifies"], "{:d}")
    pc = comp["per_dataset_concentration"]
    m.add("WidestTwoSumPP", pc["widest_two_sum_pp"], "{:.2f}")
    m.add("RemainingNineSumPP", pc["remaining_sum_pp"], "{:.2f}")
    m.add("NRemainingDatasets", pc["n_remaining"], "{:d}")
    m.add("WidestTwoShareOfSumPct", pc["widest_two_share_of_sum_pct"], "{:.0f}")
    m.add("SumAbsDeltasPP", pc["sum_abs_deltas_pp"], "{:.1f}")
    # The signed residue the absolute sum cancels down to. Section 7.3 set the absolute
    # sum against the macro average and omitted the two steps between them, so a reader
    # with a pencil got a different answer: the cancellation leaves this, and the macro
    # average is this divided by the number of datasets. Two decimals, because that is
    # the precision at which the division reproduces the headline.
    m.add("SignedSumDeltasPP", pc["sum_of_deltas_pp"], "{:.2f}")
    m.add("WidestTwoSumAbsPP", pc["widest_two_sum_abs_pp"], "{:.1f}")
    m.add("RemainingSumAbsPP", pc["remaining_sum_abs_pp"], "{:.1f}")
    m.add("MaxAbsRemainingPP", pc["max_abs_remaining_pp"], "{:.1f}")
    m.add("MeanAbsRemainingPP", pc["mean_abs_remaining_pp"], "{:.1f}")
    m.add("WidestTwoCancellationPct", pc["widest_two_cancellation_pct"], "{:.0f}")
    m.add("RemainingCancellationPct", pc["remaining_cancellation_pct"], "{:.0f}")
    # Two decimals, because section 7 prints these two cells' difference and then their
    # sum: at one decimal the two printed cells sum to a different number than the
    # printed total does.
    m.add("DeltaAggreFactCNNPP", pc["per_dataset_delta_pp"]["AggreFact-CNN"], "{:.2f}")
    m.add("DeltaExpertQAPP", pc["per_dataset_delta_pp"]["ExpertQA"], "{:.2f}")

    r = out["jev_rerun"]
    m.section("Jev run to run movement (an earlier run of the same model, section 12)")
    m.add("JevRerunFlips", r["n_flips"], "{:d}")
    m.add("JevRerunAgreementPct", r["agreement"] * 100, "{:.2f}")
    m.add("JevRerunBAcc", r["macro_bacc"] * 100)
    m.add("JevRerunBAccTwoDP", r["macro_bacc"] * 100, "{:.2f}")
    # The MAGNITUDE of the run-to-run movement, not its sign. macro_bacc_delta_pp is
    # stored earlier-minus-reported, and all three sentences that print it describe the
    # movement the other way round ("moved from the earlier to the reported"), so the
    # signed spelling put a minus in front of a rise and one of them then read a reversal
    # out of it. The direction is a word in the prose now and the macro carries no sign,
    # so there is nothing left for a sentence to contradict. check_paper_macros.py checks
    # it as an absdiff against the two levels, which are both printed.
    m.add("JevRerunDeltaPP", abs(r["macro_bacc_delta_pp"]), "{:.1f}")
    m.add("JevRerunFVR", r["false_verification_rate"] * 100)
    m.add("JevRerunMaxFlipConfidence", r["max_flip_confidence"], "{:.2f}")
    for d, v in sorted(r["per_dataset_false_verified"].items()):
        m.add(f"JevRerunFalseVerified{macro_token(d)}", v, "{:d}")
    for d, v in sorted(r["per_dataset"].items()):
        m.add(f"JevRerunBAcc{macro_token(d)}", v * 100)

    pp = out["prompt_parity"]
    m.section("the one frontier prompt change we made, measured (section 12, appendix C)")
    m.add("PromptParityFlips", pp["n_flips"], "{:d}")
    m.add("PromptParityPreBAcc", pp["pre_macro_bacc"] * 100, "{:.2f}")
    m.add("PromptParityPostBAcc", pp["post_macro_bacc"] * 100, "{:.2f}")
    m.add("PromptParityPreFVR", pp["pre_false_verification_rate"] * 100)
    m.add("PromptParityPostFVR", pp["post_false_verification_rate"] * 100)
    m.add("PromptParityFVRGapShiftPP", pp["fvr_gap_shift_pp"], "{:.2f}")
    m.add("PromptParityFlipsOnUnsupported", pp["n_flips_on_gold_unsupported"], "{:d}")
    m.add("PromptParityFlipsOnSupported", pp["n_flips_on_gold_supported"], "{:d}")
    m.add("PromptParityFlipsToSupported", pp["n_flips_to_supported"], "{:d}")
    m.add("PromptParityFlipsToUnsupported", pp["n_flips_to_unsupported"], "{:d}")

    # The same edit on the other frontier arm. Reported because the low-effort arm's
    # result does not generalise to it: there no flip landed on a gold-unsupported claim,
    # here one did, and this arm's false-verification rate is a leg of a Holm survivor.
    ph = out["prompt_parity_high"]
    m.add("PromptParityHighFlips", ph["n_flips"], "{:d}")
    m.add("PromptParityHighPreBAcc", ph["pre_macro_bacc"] * 100, "{:.2f}")
    m.add("PromptParityHighPostBAcc", ph["post_macro_bacc"] * 100, "{:.2f}")
    m.add("PromptParityHighPreFVR", ph["pre_false_verification_rate"] * 100)
    m.add("PromptParityHighPostFVR", ph["post_false_verification_rate"] * 100)
    m.add("PromptParityHighFlipsOnUnsupported", ph["n_flips_on_gold_unsupported"], "{:d}")
    m.add("PromptParityHighFlipsOnSupported", ph["n_flips_on_gold_supported"], "{:d}")
    m.add("PromptParityHighFlipsToSupported", ph["n_flips_to_supported"], "{:d}")
    m.add("PromptParityHighFlipsToUnsupported", ph["n_flips_to_unsupported"], "{:d}")
    # And what substituting the superseded run does to the survivor that rests on it.
    pph = out["cascade"]["crossfit"]["vs_astra_high_pre_parity_inference"]["fvr_inference"]
    m.add("CascadeAstraHighFVRPreParityDeltaPP", pph["fvr_delta_pp"], "{:.2f}")
    m.add("CascadeAstraHighFVRPreParityMcNemarPMedian", pph["mcnemar_p_median"], "{:.5f}")
    m.add("CascadeAstraHighFVRPreParityRepeatsPBelowFive",
          pph["n_repeats_p_below_05"], "{:d}")

    sp = out["same_prompt_repeat"]
    m.section("the frontier arm run twice under an UNCHANGED prompt (section 12)")
    m.add("SamePromptRepeatCompared", sp["n_compared"], "{:d}")
    m.add("SamePromptRepeatExcluded", sp["n_capped_no_answer"], "{:d}")
    m.add("SamePromptRepeatFlips", sp["n_flips"], "{:d}")
    m.add("SamePromptRepeatAgreementPct", sp["agreement"] * 100, "{:.2f}")
    m.add("SamePromptRepeatBAcc", sp["capped_macro_bacc"] * 100, "{:.2f}")
    m.add("SamePromptRepeatBAccOther", sp["repeat_macro_bacc"] * 100, "{:.2f}")
    m.add("SamePromptRepeatDeltaPP", sp["macro_bacc_delta_pp"], "{:.2f}")
    m.add("SamePromptRepeatCappedTokenCap", sp["capped_max_completion_tokens"], "{:d}")
    m.add("SamePromptRepeatTokenCap", sp["repeat_max_completion_tokens"], "{:d}")
    # What the repeat covers. These are the macros that stop the zero-flip result being
    # read as a determinism result for the reported configuration: the compared rows are
    # the rows on which this configuration reports no reasoning tokens, and every
    # prompt-parity flip is outside them.
    m.add("SamePromptRepeatComparedReasoningZero",
          sp["compared_reasoning_zero_n_reported"], "{:d}")
    m.add("SamePromptRepeatComparedReasoningMean",
          sp["compared_reasoning_mean_reported"], "{:.2f}")
    m.add("SamePromptRepeatExcludedReasoningNonzero",
          sp["excluded_reasoning_nonzero_n_reported"], "{:d}")
    m.add("SamePromptRepeatPromptEditFlipsOnCompared",
          sp["prompt_edit_flips_on_compared"], "{:d}")
    # Where the parity flips landed: a description of where the movement sits. It is NOT
    # evidence that the prompt rather than the draw moved them, because it sits entirely
    # inside the rows the repeat could not reach, and because concentration in the rows
    # where the model reasons most is also what a sampling effect looks like.
    m.add("PromptParityFlipsInCappedNoAnswer",
          sp["parity_flips_in_capped_no_answer"], "{:d}")
    m.add("PromptParityFlipsExpectedInCappedNoAnswer",
          sp["parity_flips_expected_in_capped_no_answer"], "{:.1f}")
    m.add("PromptParityFlipsReasoningMean",
          sp["parity_flip_reasoning_mean_reported"], "{:.1f}")

    m.section("cost ratios")
    m.add("CostRatioAstraOverJev", out["ratios"]["cost_astra_over_jev"], "{:.0f}")
    m.add("CostRatioAstraHighOverJev", out["ratios"]["cost_astra_high_over_jev"], "{:.0f}")
    # Two decimals: at one, "3.1" does not reproduce from the two costs printed beside it
    # (8.93 / 2.92 = 3.06), and the Monte Carlo error of the denominator is larger than the
    # gap between 3.0 and 3.1, so the extra digit is honest precision and not false.
    m.add("CostRatioAstraOverCascade", out["ratios"]["cost_astra_over_crossfit_cascade"], "{:.2f}")

    m.section("what the headline cost ratio is actually made of")
    cd = out["cost_decomposition"]
    m.add("JevInputPricePerMtok", cd["jev_input_price_per_mtok_usd"], "{:.3f}")
    m.add("AstraInputPricePerMtok", cd["astra_input_price_per_mtok_usd"], "{:.2f}")
    m.add("AstraOutputPricePerMtok", cd["astra_output_price_per_mtok_usd"], "{:.2f}")
    # Section 4 prints this decomposition as a chain a reader is invited to multiply out,
    # so every link is carried at a precision that reproduces the next one. At the coarser
    # formats it used to carry, 238 divided by 1.35 and multiplied by 1.059 came to 186
    # where the paper printed the 187 headline.
    m.add("InputPriceRatio", cd["input_price_ratio"], "{:.1f}")
    m.add("InputTokenRatioJevOverAstra", cd["input_token_ratio_jev_over_astra"], "{:.3f}")
    m.add("AstraOutputUplift", cd["astra_output_token_uplift"], "{:.3f}")
    m.add("RatioFromPriceAndTokens", cd["ratio_from_price_and_tokens_only"], "{:.1f}")
    m.add("JevCostPerKIfOutputBilled",
          cd["jev_cost_per_1k_usd_if_output_billed_at_input_rate"], "{:.5f}")
    m.add("CostRatioIfOutputBilled", cd["cost_ratio_if_output_billed_at_input_rate"], "{:.0f}")
    m.add("FreeOutputShareOfRatioPct", cd["free_output_share_of_ratio_pct"], "{:.0f}")

    m.section("published LLM-AggreFact reference rows (test split, other harnesses)")
    for name, val in sorted(out["published_leaderboard"]["rows"].items()):
        m.add(f"Published{_LEADERBOARD_MACRO[name]}TestBAcc", val * 100)
    m.add("PublishedLeaderboardRows", len(out["published_leaderboard"]["rows"]), "{:d}")

    m.section("confidence gating sweep, Jev alone")
    for g in out["gating"]:
        t = _THRESHOLD_WORDS[g["threshold"]]
        m.add(f"GateAt{t}Coverage", g["coverage"] * 100)
        m.add(f"GateAt{t}BAcc", g["macro_bacc"] * 100)
        m.add(f"GateAt{t}VerifiedPrecision", g["verified_precision"] * 100)
        m.add(f"GateAt{t}FVR", g["false_verification_rate"] * 100)
        # The macro average at a tight gate is not always over all eleven datasets, so
        # the count goes in a column of its own rather than in a footnote nobody writes.
        m.add(f"GateAt{t}NDatasets", g["n_datasets_scored"], "{:d}")
        m.add(f"GateAt{t}NDatasetsDropped", g["n_datasets_dropped"], "{:d}")

    m.section("cascade, in sample (SELECTION CONTAMINATED: section 10 only)")
    for c in out["cascade"]["in_sample_sweep"]:
        t = _THRESHOLD_WORDS[c["threshold"]]
        m.add(f"CascadeAt{t}BAcc", c["macro_bacc"] * 100)
        m.add(f"CascadeAt{t}BAccTwoDP", c["macro_bacc"] * 100, "{:.2f}")
        m.add(f"CascadeAt{t}Escalated", c["escalated"] * 100)
        m.add(f"CascadeAt{t}Cost", c["cost_per_1k_usd"], "{:.2f}")
        m.add(f"CascadeAt{t}FVR", c["false_verification_rate"] * 100)
        # The guard metric of this row as claims rather than a rate, with the frontier
        # arm's count beside it and the exact McNemar between the two. A reader quoting
        # one of these rows against the frontier arm needs to know the margin is one
        # claim before quoting it, so the counts and the test are macros too.
        m.add(f"CascadeAt{t}FVRCount", c["n_false_verified"], "{:d}")
        m.add(f"CascadeAt{t}FVRVsAstraCascadeOnly", c["fv_vs_astra_cascade_only"], "{:d}")
        m.add(f"CascadeAt{t}FVRVsAstraAstraOnly", c["fv_vs_astra_astra_only"], "{:d}")
        m.add(f"CascadeAt{t}FVRVsAstraMcNemarP", c["fv_vs_astra_mcnemar_p"], "{:.2f}")
        m.add(f"CascadeAt{t}MinusAstraPP", c["macro_bacc_minus_astra_pp"])
    b = out["cascade"]["in_sample_best"]
    m.add("CascadeInSampleBAcc", b["macro_bacc"] * 100)
    m.add("CascadeInSampleBAccTwoDP", b["macro_bacc"] * 100, "{:.2f}")
    m.add("CascadeInSampleEscalated", b["escalated"] * 100)
    m.add("CascadeInSampleCost", b["cost_per_1k_usd"], "{:.2f}")
    m.add("CascadeInSampleThreshold", b["threshold"], "{:.2f}")

    cf = out["cascade"]["crossfit"]
    m.section("cascade, cross fitted (the reportable operating point)")
    m.add("CascadeCrossfitBAcc", cf["macro_bacc_mean"] * 100)
    m.add("CascadeCrossfitCILo", cf["macro_bacc_ci"][0] * 100)
    m.add("CascadeCrossfitCIHi", cf["macro_bacc_ci"][1] * 100)
    m.add("CascadeCrossfitEscalated", cf["escalated_mean"] * 100)
    m.add("CascadeCrossfitCost", cf["cost_per_1k_usd_mean"], "{:.2f}")
    m.add("CascadeCrossfitMinusAstraPP", cf["minus_astra_pp_mean"], "{:.2f}")
    m.add("CascadeCrossfitFVR", cf["false_verification_rate_mean"] * 100)
    m.add("CascadeCrossfitFVRTwoDP", cf["false_verification_rate_mean"] * 100, "{:.2f}")
    m.add("CascadeCrossfitFVRCILo", cf["false_verification_rate_ci"][0] * 100)
    m.add("CascadeCrossfitFVRCIHi", cf["false_verification_rate_ci"][1] * 100)
    m.add("CascadeCrossfitVerifiedPrecision", cf["verified_precision_mean"] * 100)
    m.add("CascadeCrossfitMinusAstraFVRPP", cf["minus_astra_fvr_pp_mean"], "{:.1f}")
    # The comparison the paper's headline rests on, with an interval that is a confidence
    # interval and a test that exists. \CascadeCrossfitMinusAstraPP above is the same
    # point estimate carrying only the spread over repeats, so these are the ones any
    # sentence about separability has to cite.
    vi = cf["vs_astra_inference"]
    m.add("CascadeAstraDeltaPP", vi["macro_bacc_delta_pp"], "{:.2f}")
    m.add("CascadeAstraDeltaCILo", vi["macro_bacc_delta_ci_pp"][0], "{:.2f}")
    m.add("CascadeAstraDeltaCIHi", vi["macro_bacc_delta_ci_pp"][1], "{:.2f}")
    # Section 9 says this interval is too wide to resolve the comparison. How wide it is
    # is a reported quantity and therefore a macro, not an adjective rounded by hand.
    m.add("CascadeAstraDeltaCIWidthPP",
          vi["macro_bacc_delta_ci_pp"][1] - vi["macro_bacc_delta_ci_pp"][0], "{:.2f}")
    m.add("CascadeAstraMcNemarPMedian", vi["mcnemar_p_median"], "{:.2f}")
    m.add("CascadeAstraMcNemarPMin", vi["mcnemar_p_min"], "{:.2f}")
    m.add("CascadeAstraMcNemarPMax", vi["mcnemar_p_max"], "{:.2f}")
    m.add("CascadeAstraDiscordantMedian", vi["n_discordant_median"], "{:.0f}")
    m.add("CascadeAstraRepeatsPBelowFive", vi["n_repeats_p_below_05"], "{:d}")
    m.add("CascadeAstraRepeatsPBelowFivePct", vi["frac_repeats_p_below_05"] * 100)
    m.add("CascadeAstraRepeatsPBelowFiveForCascade",
          vi["n_repeats_p_below_05_favouring_cascade"], "{:d}")
    # The two directions that are not the same direction (see the inference function).
    m.add("CascadeAstraMcNemarFavoursCascade",
          vi["n_repeats_mcnemar_favours_cascade"], "{:d}")
    m.add("CascadeAstraMcNemarFavoursAstra",
          vi["n_repeats_mcnemar_favours_arm"], "{:d}")
    m.add("CascadeAstraRepeatsMacroBelowAstra", vi["n_repeats_macro_below_arm"], "{:d}")
    m.add("CascadeAstraBootstrapDrawsNegativePct",
          vi["frac_bootstrap_draws_negative"] * 100)

    m.section("cascade against the high-effort frontier arm, its neighbour in table 4")
    # The row directly under the cross-fitted cascade in Table 4 is ahead of it on accuracy
    # and six points ahead on the guard metric. The paper prints that row, so it prints the
    # difference against it too, from the same bootstrap and the same exact McNemar.
    vh = cf["vs_astra_high_inference"]
    m.add("CascadeAstraHighDeltaPP", vh["macro_bacc_delta_pp"], "{:.2f}")
    m.add("CascadeAstraHighDeltaCILo", vh["macro_bacc_delta_ci_pp"][0], "{:.2f}")
    m.add("CascadeAstraHighDeltaCIHi", vh["macro_bacc_delta_ci_pp"][1], "{:.2f}")
    m.add("CascadeAstraHighMcNemarPMedian", vh["mcnemar_p_median"], "{:.2f}")
    m.add("CascadeAstraHighMcNemarPMin", vh["mcnemar_p_min"], "{:.2f}")
    m.add("CascadeAstraHighMcNemarPMax", vh["mcnemar_p_max"], "{:.2f}")
    m.add("CascadeAstraHighDiscordantMedian", vh["n_discordant_median"], "{:.0f}")
    m.add("CascadeAstraHighRepeatsPBelowFive", vh["n_repeats_p_below_05"], "{:d}")
    m.add("CascadeAstraHighRepeatsMacroBelowArm",
          vh["n_repeats_macro_below_arm"], "{:d}")

    m.section("cascade against Jev alone, the comparison a cascade paper wants to skip")
    m.add("CostRatioCascadeOverJev", cf["cost_ratio_crossfit_over_jev"], "{:.0f}")
    m.add("CascadeCrossfitMinusJevPP", cf["minus_jev_pp_mean"], "{:.2f}")
    m.add("CascadeCrossfitMinusJevFVRPP", cf["minus_jev_fvr_pp_mean"], "{:.1f}")
    vj = cf["vs_jev_inference"]
    m.add("CascadeJevDeltaPP", vj["macro_bacc_delta_pp"], "{:.2f}")
    m.add("CascadeJevDeltaCILo", vj["macro_bacc_delta_ci_pp"][0], "{:.2f}")
    m.add("CascadeJevDeltaCIHi", vj["macro_bacc_delta_ci_pp"][1], "{:.2f}")
    m.add("CascadeJevMcNemarPMedian", vj["mcnemar_p_median"], "{:.2f}")
    m.add("CascadeJevMcNemarPMin", vj["mcnemar_p_min"], "{:.2f}")
    m.add("CascadeJevMcNemarPMax", vj["mcnemar_p_max"], "{:.2f}")
    m.add("CascadeJevDiscordantMedian", vj["n_discordant_median"], "{:.0f}")
    m.add("CascadeJevRepeatsPBelowFive", vj["n_repeats_p_below_05"], "{:d}")
    m.add("CascadeJevRepeatsPBelowFiveForCascade",
          vj["n_repeats_p_below_05_favouring_cascade"], "{:d}")
    m.add("CascadeJevRepeatsMacroBelowJev", vj["n_repeats_macro_below_arm"], "{:d}")
    m.add("CascadeJevBootstrapDrawsNegativePct",
          vj["frac_bootstrap_draws_negative"] * 100)
    m.add("SelectionOptimismPP", cf["selection_optimism_pp"], "{:.2f}")
    m.add("CrossfitRepeats", cf["n_repeats"], "{:d}")
    m.add("CrossfitFolds", cf["k_folds"], "{:d}")
    m.add("CrossfitThresholdFits", cf["n_threshold_fits"], "{:d}")

    # The guard metric on the same footing as the accuracy metric, for both neighbours.
    mp = out["multiplicity"]
    m.section("Holm-Bonferroni over every McNemar the paper reports (spec requirement)")
    m.add("MultiplicityFamilyN", mp["family_size"], "{:d}")
    m.add("NClearingUncorrected", mp["n_clearing_uncorrected"], "{:d}")
    m.add("MultiplicitySurvivingN",
          len(mp["survives_at_05_excluding_minicheck"]), "{:d}")
    m.add("MultiplicityMedianMembersN", len(mp["median_members"]), "{:d}")
    m.add("FVRGapHolmBreakFamilyN", mp["fvr_gap_holm_break_family_n"], "{:d}")
    # The same boundary for the result that did NOT survive, which is the direction the
    # paper owed a reader: the family it fixed in code is the one that keeps the cascade's
    # guard penalty out, and this is the family size at which it would come in.
    m.add("CascadeGuardHolmClearsAtFamilyN",
          mp["cascade_guard_holm_clears_at_family_n"], "{:d}")
    m.add("CascadeGuardHolmPAtThatFamily",
          mp["cascade_guard_holm_p_at_that_family"], "{:.3f}")
    m.add("CascadeGuardHolmPExcludingMiniCheck",
          mp["cascade_guard_holm_p_excluding_minicheck"], "{:.3f}")
    m.add("CascadeGuardHolmMembersAboveN",
          mp["cascade_guard_holm_members_above_n"], "{:d}")
    # How many members have to leave before it clears. It was one while the family had
    # eleven members; adding the two comparisons against the high-effort arm made it two,
    # and the appendix sentence that said "drop any one of them" was true only at the old
    # size. The count is a macro so the sentence cannot go stale silently again.
    m.add("CascadeGuardHolmBreakDroppedN",
          len(mp["cascade_guard_holm_break_dropped"]), "{:d}")
    for name, adj in sorted(mp["adjusted"].items()):
        m.add(f"{name}McNemarPHolm", adj, "{:.3f}")
    # The two excluded-arm pairs are smaller than 1e-13, so three decimals prints them as
    # 0.000 raw and adjusted, which is a bound no probability satisfies and which appendix
    # E printed for a round. A macro body carries bare digits only (see Macros.add), so
    # scientific notation is emitted as a mantissa and an exponent the prose composes.
    for name in ("JevMiniCheck", "AstraMiniCheck"):
        for kind, source in (("", mp["raw"]), ("Holm", mp["adjusted"])):
            add_scientific(m, f"{name}McNemarP{kind}", source[name])

    m.section("the guard metric, with an interval and a test (both neighbours)")
    for label, block in (("CascadeAstra", vi["fvr_inference"]),
                         ("CascadeJev", vj["fvr_inference"]),
                         ("CascadeAstraHigh", vh["fvr_inference"])):
        m.add(f"{label}FVRDeltaPP", block["fvr_delta_pp"], "{:.2f}")
        m.add(f"{label}FVRDeltaCILo", block["fvr_delta_ci_pp"][0], "{:.2f}")
        m.add(f"{label}FVRDeltaCIHi", block["fvr_delta_ci_pp"][1], "{:.2f}")
        # Four decimals for the same reason as FVRGapMcNemarP: appendix E prints this
        # median beside its Holm adjustment and names the multiplier, so the printed
        # digits have to reproduce the printed adjustment. At three decimals 0.006 x 8
        # gives 0.048 against a printed 0.051, which inverts the answer the paper gives, and
        # at four it gives 0.0504 against that same 0.051. Five reproduces both the
        # adjustment and the family-of-ten sensitivity printed beside it.
        m.add(f"{label}FVRMcNemarPMedian", block["mcnemar_p_median"], "{:.5f}")
        m.add(f"{label}FVRMcNemarPMin", block["mcnemar_p_min"], "{:.3f}")
        m.add(f"{label}FVRMcNemarPMax", block["mcnemar_p_max"], "{:.2f}")
        m.add(f"{label}FVRDiscordantMedian", block["n_discordant_median"], "{:.0f}")
        m.add(f"{label}FVRRepeatsPBelowFive", block["n_repeats_p_below_05"], "{:d}")
        m.add(f"{label}FVRRepeatsFavouringCascade",
              block["n_repeats_favouring_cascade"], "{:d}")

    m.section("Monte Carlo error of the cross-fitted means, and the seed's fingerprint")
    mc = cf["monte_carlo_sem"]
    m.add("CrossfitSEMBAccPP", mc["macro_bacc_pp"], "{:.3f}")
    m.add("CrossfitSEMCost", mc["cost_per_1k_usd"], "{:.3f}")
    m.add("CrossfitSEMEscalatedPP", mc["escalated_pp"], "{:.3f}")
    m.add("CrossfitSEMFVRPP", mc["fvr_pp"], "{:.3f}")
    m.add("RepeatsReproducingInSample", cf["n_repeats_reproducing_in_sample"], "{:d}")
    m.add("RepeatsReproducingInSamplePct",
          cf["n_repeats_reproducing_in_sample"] / cf["n_repeats"] * 100, "{:.1f}")
    m.add("AnalysisSeed", cf["seed"], "{:d}")
    for t, n in cf["threshold_selection_counts"].items():
        m.add(f"ThresholdPicked{_THRESHOLD_WORDS[float(t)]}", n, "{:d}")

    # Two decimal forms exist for every headline score (see the per-arm loop above). This
    # is the cascade's, which two tables print beside \AstraHighBAccTwoDP because the two
    # are different quantities that round to the same value at one decimal.
    m.section("two decimal forms, for tables and for printed differences")
    m.add("CascadeCrossfitBAccTwoDP", cf["macro_bacc_mean"] * 100, "{:.2f}")
    m.add("OracleBAccTwoDP", out["cascade"]["oracle"]["macro_bacc"] * 100, "{:.2f}")

    m.section("random escalation control and oracle routing")
    for c in out["cascade"]["random_control"]:
        t = _THRESHOLD_WORDS[c["threshold"]]
        m.add(f"RandomAt{t}Escalated", c["escalated"] * 100)
        m.add(f"RandomAt{t}BAcc", c["random_macro_bacc_mean"] * 100)
        m.add(f"RandomAt{t}BAccTwoDP", c["random_macro_bacc_mean"] * 100, "{:.2f}")
        m.add(f"RandomAt{t}CILo", c["random_macro_bacc_ci"][0] * 100)
        m.add(f"RandomAt{t}CIHi", c["random_macro_bacc_ci"][1] * 100)
        m.add(f"RandomAt{t}CILoTwoDP", c["random_macro_bacc_ci"][0] * 100, "{:.2f}")
        m.add(f"RandomAt{t}CIHiTwoDP", c["random_macro_bacc_ci"][1] * 100, "{:.2f}")
        # Two decimals across the whole gain column, because it is the difference of the
        # two columns printed beside it and at one decimal the t=0.90 row did not
        # reproduce: 74.0 minus 73.4 is 0.6 and the gain printed 0.7. Every row reproduces
        # at two.
        m.add(f"ConfMinusRandomAt{t}PP", c["confidence_minus_random_pp"], "{:.2f}")
    m.add("NRandomBudgetsTested", len(out["cascade"]["random_control"]), "{:d}")
    m.add("NRandomBudgetsConfidenceWins",
          sum(1 for c in out["cascade"]["random_control"]
              if c["confidence_minus_random_pp"] > 0), "{:d}")
    m.add("ConfMinusRandomMinPP",
          min(c["confidence_minus_random_pp"] for c in out["cascade"]["random_control"]), "{:.2f}")
    m.add("ConfMinusRandomMaxPP",
          max(c["confidence_minus_random_pp"] for c in out["cascade"]["random_control"]), "{:.2f}")
    o = out["cascade"]["oracle"]
    m.add("OracleBAcc", o["macro_bacc"] * 100)
    m.add("OracleEscalated", o["escalated"] * 100)
    m.add("OracleCost", o["cost_per_1k_usd"], "{:.2f}")
    m.add("OracleHeadroomPP", o["headroom_over_jev_pp"])
    m.add("OracleFVR", o["false_verification_rate"] * 100)
    # Two decimal form so that the oracle row does not have to break the precision of a
    # column whose other cells are printed to two decimals for the differences the
    # surrounding section takes of them.
    m.add("OracleFVRTwoDP", o["false_verification_rate"] * 100, "{:.2f}")
    m.add("OracleVerifiedPrecision", o["verified_precision"] * 100)

    text = m.render()
    # chr() rather than the characters themselves, so that a grep for em and en dashes
    # across this repo stays honest and does not hit its own detector.
    if "---" in text or chr(0x2014) in text or chr(0x2013) in text:
        raise SystemExit(
            "REFUSING to write numbers.tex: it contains a dash run that renders as an em dash"
        )
    changed = write_if_changed(ARXIV_TEX, text)
    print(f"{'wrote' if changed else 'unchanged'} {ARXIV_TEX.relative_to(REPO)}  "
          f"({m.count()} macros)")


def write_hashes_tex(out: dict) -> None:
    """The digests, as a block the reproduction appendix inputs.

    The paper's provenance claim is that a reader can tell whether the file in front of
    them is the file that was analysed. A tag and a timestamp proof establish that in the
    repository that carries them, and establish nothing at all for a reader holding a
    download from somewhere else, so the digests are printed in the paper too. Generated
    here, never typed (CLAUDE.md rule 4).
    """
    prov = out["provenance"]
    entries = [(p, h) for p, h in sorted(prov["prediction_file_sha256"].items())]
    entries.append((prov["sample_id_list"]["path"], prov["sample_id_list"]["sha256"]))
    entries.append(("LLM-AggreFact dev parquet", prov["parquet_sha256"]))
    blocks = [f"{p.replace('_', chr(92) + '_')}\\\\\n{h}" for p, h in entries]
    text = "\n".join([
        "% arxiv/generated/hashes.tex",
        "% GENERATED by scripts/verify/paired_analysis.py. Do not edit by hand.",
        "% SHA-256 of every file the paper's numbers are computed from, so that the",
        "% provenance claim in appendix D is checkable from the paper alone.",
        "\\begin{flushleft}\\ttfamily\\footnotesize\\raggedright",
        "\\\\[4pt]\n".join(blocks),
        "\\end{flushleft}",
        "",
    ])
    if "---" in text:
        raise SystemExit("REFUSING to write hashes.tex: it contains a dash run")
    changed = write_if_changed(ARXIV_HASHES_TEX, text)
    print(f"{'wrote' if changed else 'unchanged'} {ARXIV_HASHES_TEX.relative_to(REPO)}  "
          f"({len(entries)} digests)")


# --------------------------------------------------------------------------------
def write_compat_views(out: dict, df: pd.DataFrame) -> None:
    """The two older JSON files, rebuilt from the same computation.

    arxiv/generated/crossfit.json is cited by arxiv/FINDING-cascade-out-of-sample.md and
    medium/generated/numbers.json is the published article's evidence, checked by
    scripts/verify/check_article_numbers.py. Both schemas stay frozen; they are now views
    of numbers.json rather than separate computations that could drift away from it.
    """
    cfg, cf = out["configurations"], out["cascade"]["crossfit"]
    crossfit_view = dict(
        note="A VIEW of arxiv/generated/numbers.json, written by scripts/verify/paired_analysis.py.",
        controls=[dict(threshold=c["threshold"], escalated=c["escalated"],
                       confidence_bacc=c["confidence_macro_bacc"],
                       random_bacc_mean=c["random_macro_bacc_mean"],
                       random_bacc_lo=c["random_macro_bacc_ci"][0],
                       random_bacc_hi=c["random_macro_bacc_ci"][1],
                       confidence_minus_random_pp=c["confidence_minus_random_pp"],
                       oracle_bacc=out["cascade"]["oracle"]["macro_bacc"],
                       oracle_escalated=out["cascade"]["oracle"]["escalated"])
                  for c in out["cascade"]["random_control"]],
        n=out["n_paired"], n_repeats=cf["n_repeats"], k_folds=cf["k_folds"], seed=cf["seed"],
        thresholds=cf["thresholds"],
        jev_only_bacc=cfg["jev"]["macro_bacc"], astra_only_bacc=cfg["astra"]["macro_bacc"],
        in_sample_sweep=[dict(threshold=c["threshold"], bacc=c["macro_bacc"],
                              escalated=c["escalated"], cost_per_1k=c["cost_per_1k_usd"])
                         for c in out["cascade"]["in_sample_sweep"]],
        in_sample_best=dict(threshold=out["cascade"]["in_sample_best"]["threshold"],
                            bacc=out["cascade"]["in_sample_best"]["macro_bacc"],
                            escalated=out["cascade"]["in_sample_best"]["escalated"],
                            cost_per_1k=out["cascade"]["in_sample_best"]["cost_per_1k_usd"]),
        crossfit=dict(bacc_mean=cf["macro_bacc_mean"], bacc_lo=cf["macro_bacc_ci"][0],
                      bacc_hi=cf["macro_bacc_ci"][1], escalated_mean=cf["escalated_mean"],
                      cost_per_1k_mean=cf["cost_per_1k_usd_mean"]),
        optimism_pp=cf["selection_optimism_pp"],
        threshold_selection_counts=cf["threshold_selection_counts"],
    )
    write_if_changed(CROSSFIT_JSON,
                     json.dumps(crossfit_view, indent=2, sort_keys=True, default=float) + "\n")

    if out["minicheck_scored"]:
        mc_view = dict(
            note="A VIEW of arxiv/generated/numbers.json. MiniCheck is an EXCLUDED arm.",
            excluded_arm="minicheck", exclusion_reason=out["minicheck_exclusion"],
            n=out["n_paired"],
            systems={a: dict(macro_bacc=cfg[a]["macro_bacc"], raw_accuracy=cfg[a]["raw_accuracy"],
                             fvr=cfg[a]["false_verification_rate"],
                             pred_supported_rate=cfg[a]["pred_supported_rate"],
                             per_dataset=cfg[a]["per_dataset"],
                             probability=cfg[a]["calibration"]["probability_provenance"],
                             **({k: cfg[a]["calibration"][k] for k in ("brier", "ece", "auroc")}
                                if cfg[a]["calibration"]["probability_provenance"] == "native" else {}))
                     for a in ("jev", "astra", "minicheck")},
            pairs={k: dict(agreement=v["agreement"],
                                             a_only_correct=v["a_only_correct"],
                                             b_only_correct=v["b_only_correct"],
                                             both_correct=v["both_correct"],
                                             neither_correct=v["neither_correct"])
                   for k, v in out["pairs"].items() if "minicheck" in k or k == "jev|astra"},
        )
        write_if_changed(MINICHECK_JSON,
                         json.dumps(mc_view, indent=2, sort_keys=True, default=float) + "\n")

    # medium/generated/numbers.json: the published article's frozen schema.
    j, a, h = cfg["jev"], cfg["astra"], cfg["astra_high"]
    medium = dict(
        astra_high_effort=dict(mean_bacc=h["macro_bacc"], per_dataset=h["per_dataset"],
                               cost_per_1k=h["cost_per_1k_usd"], p50_ms=h["latency_ms"]["p50"],
                               mean_output_tokens=h["mean_output_tokens"],
                               fvr=h["false_verification_rate"],
                               verified_precision=h["verified_precision"],
                               valid_rate=out["valid_rate"]["astra_high"],
                               reasoning_effort="high", max_completion_tokens=4096),
        generated_from=dict(jev=out["provenance"]["runs"]["jev"],
                            astra=out["provenance"]["runs"]["astra"],
                            dataset="lytang/LLM-AggreFact dev",
                            parquet_sha256=out["provenance"]["parquet_sha256"]),
        n_paired=out["n_paired"], n_datasets=out["n_datasets"],
        astra_rows=out["rows_read"]["astra"],
        astra_ok=int(round(out["valid_rate"]["astra"] * out["rows_read"]["astra"])),
        jev_rows=out["rows_read"]["jev"],
        jev_ok=int(round(out["valid_rate"]["jev"] * out["rows_read"]["jev"])),
        jev_valid_rate=out["valid_rate"]["jev"], astra_valid_rate=out["valid_rate"]["astra"],
        jev=dict(mean_bacc=j["macro_bacc"], ci=j["macro_bacc_ci"], per_dataset=j["per_dataset"],
                 fvr=j["false_verification_rate"], verified_precision=j["verified_precision"],
                 cost_per_1k=j["cost_per_1k_usd"], p50_ms=j["latency_ms"]["p50"],
                 p95_ms=j["latency_ms"]["p95"], mean_input_tokens=j["mean_input_tokens"]),
        astra=dict(mean_bacc=a["macro_bacc"], ci=a["macro_bacc_ci"], per_dataset=a["per_dataset"],
                   fvr=a["false_verification_rate"], verified_precision=a["verified_precision"],
                   cost_per_1k=a["cost_per_1k_usd"], p50_ms=a["latency_ms"]["p50"],
                   p95_ms=a["latency_ms"]["p95"], mean_input_tokens=a["mean_input_tokens"],
                   mean_output_tokens=a["mean_output_tokens"]),
        delta=dict(mean=out["pairs"]["jev|astra"]["macro_bacc_delta_boot_mean_pp"] / 100,
                   ci=[out["pairs"]["jev|astra"]["macro_bacc_delta_ci_pp"][0] / 100,
                       out["pairs"]["jev|astra"]["macro_bacc_delta_ci_pp"][1] / 100],
                   jev_minus_astra_pp=out["pairs"]["jev|astra"]["macro_bacc_delta_pp"]),
        paired=dict(agreement=out["complementarity"]["agreement"],
                    jev_only_correct=out["complementarity"]["jev_only_correct"],
                    astra_only_correct=out["complementarity"]["astra_only_correct"],
                    mcnemar_p=out["pairs"]["jev|astra"]["mcnemar_p"],
                    mcnemar_statistic=out["pairs"]["jev|astra"]["mcnemar_statistic"],
                    mcnemar_tests=out["pairs"]["jev|astra"]["mcnemar_tests"],
                    raw_accuracy_favors=out["pairs"]["jev|astra"]["raw_accuracy_favors"]),
        ratios=dict(cost=out["ratios"]["cost_astra_over_jev"]),
        raw_accuracy=dict(jev=j["raw_accuracy"], astra=a["raw_accuracy"]),
        operational=dict(n=out["operational"]["n"],
                         jev_mean_bacc=out["operational"]["jev_macro_bacc"],
                         astra_mean_bacc=out["operational"]["astra_macro_bacc"]),
        concurrency_note=out["latency_note"],
        gating=[dict(threshold=g["threshold"], coverage=g["coverage"], bacc=g["macro_bacc"],
                     bacc_pooled=g["pooled_balanced_accuracy"],
                     n_datasets_scored=g["n_datasets_scored"],
                     verified_precision=g["verified_precision"], n=g["n"])
                for g in out["gating"]],
        cascade=[dict(threshold=c["threshold"], escalated=c["escalated"], bacc=c["macro_bacc"],
                      cost_per_1k=c["cost_per_1k_usd"])
                 for c in out["cascade"]["in_sample_sweep"]],
    )
    write_if_changed(MEDIUM_JSON, json.dumps(medium, indent=2, sort_keys=True, default=float) + "\n")


def write_evidence(file_hashes: dict, out: dict) -> None:
    """Write the evidence lock, and refuse to silently rewrite a stamped one.

    raw_hashes.json is covered by an OpenTimestamps proof and by the pushed tag
    results-lock-arxiv-v1. Once stamped its bytes are fixed: editing it is what
    makes `ots verify` report "File does not match original!", which is exactly
    the failure a reader would hit when checking our provenance claim. A D3
    repair round did precisely that by adding two input hashes here, so the file
    is now written fail-closed and auxiliary hashes live in their own file.
    """
    payload = json.dumps(dict(
        lock="results-lock-arxiv-v1",
        what="sha256 of every prediction file the arXiv paper's numbers are computed from, "
             "plus the pinned dataset parquet. Lock before looking (spec section 43): if any "
             "hash here changes, every number in the paper has to be regenerated and the "
             "reason recorded in docs/reference/incident-log.md.",
        generator="scripts/verify/paired_analysis.py",
        algorithm="sha256",
        prediction_files=file_hashes,
        dataset_parquet_sha256=out["provenance"]["parquet_sha256"],
        configs={p: sha256_file(REPO / p) for p in
                 ("configs/benchmark.yaml", "configs/pricing_2026-09-20.yaml")},
        n_paired=out["n_paired"],
    ), indent=2, sort_keys=True) + "\n"

    if EVIDENCE_JSON.is_file() and EVIDENCE_JSON.read_text() != payload:
        raise SystemExit(
            f"REFUSING to rewrite {EVIDENCE_JSON.relative_to(REPO)}.\n"
            "It is stamped by arxiv/evidence/raw_hashes.json.ots and pinned by the pushed "
            "tag results-lock-arxiv-v1, so changing it breaks `ots verify` for every reader.\n"
            "If a PREDICTION hash moved, that is an incident: regenerate every affected number "
            "and record why in docs/reference/incident-log.md.\n"
            "If you only want to hash more inputs, add them to INPUTS_JSON instead."
        )
    write_if_changed(EVIDENCE_JSON, payload)

    # Auxiliary inputs. These are not covered by the timestamp proof and are free
    # to change, which is the whole reason they are not in the stamped file.
    write_if_changed(INPUTS_JSON, json.dumps(dict(
        what="sha256 of supporting inputs that are NOT part of the timestamped evidence lock. "
             "The prediction files and the dataset parquet are in raw_hashes.json, which is "
             "stamped and must not change; these are recorded for reproducibility only.",
        generator="scripts/verify/paired_analysis.py",
        algorithm="sha256",
        inputs={p: sha256_file(REPO / p)
                for p in (LEADERBOARD_MANIFEST, SAMPLE_ID_LIST, PRE_PARITY_ASTRA,
                          PRE_PARITY_ASTRA_HIGH, CAPPED_ASTRA)},
    ), indent=2, sort_keys=True) + "\n")
    print(f"evidence lock: {EVIDENCE_JSON.relative_to(REPO)} (stamped, unchanged)")
    print(f"aux inputs:    {INPUTS_JSON.relative_to(REPO)}")


def print_summary(out: dict) -> None:
    cfg = out["configurations"]
    p = lambda x: f"{x * 100:.1f}"
    print(f"\nn = {out['n_paired']} paired examples over {out['n_datasets']} datasets "
          f"({out['n_gold_supported']} supported, {out['n_gold_unsupported']} not)\n")
    print(f"  {'configuration':<12} {'BAcc':>6} {'95% CI':>16} {'raw':>6} {'FVR':>6} "
          f"{'VP':>6} {'$/1k':>8} {'AUROC':>6}")
    for arm in ("jev", "astra", "astra_high", "minicheck"):
        if arm not in cfg:
            continue
        c = cfg[arm]
        au = c["calibration"].get("auroc")
        tag = " EXCLUDED" if c.get("excluded") else ""
        print(f"  {arm:<12} {p(c['macro_bacc']):>6} "
              f"[{p(c['macro_bacc_ci'][0])},{p(c['macro_bacc_ci'][1])}] "
              f"{p(c['raw_accuracy']):>6} {p(c['false_verification_rate']):>6} "
              f"{p(c['verified_precision']):>6} {c['cost_per_1k_usd']:>8.3f} "
              f"{(f'{au:.3f}' if au else '  -  '):>6}{tag}")
    ja = out["pairs"]["jev|astra"]
    print(f"\n  jev - astra: {ja['macro_bacc_delta_pp']:+.1f} pp "
          f"[{ja['macro_bacc_delta_ci_pp'][0]:+.1f}, {ja['macro_bacc_delta_ci_pp'][1]:+.1f}], "
          f"McNemar p = {ja['mcnemar_p']:.2f}")
    comp = out["complementarity"]
    print(f"  disagree on {p(comp['disagreement'])}% of claims; jev only right {comp['jev_only_correct']}, "
          f"astra only right {comp['astra_only_correct']}")
    b, cf = out["cascade"]["in_sample_best"], out["cascade"]["crossfit"]
    print(f"\n  cascade in-sample   {p(b['macro_bacc'])} at t={b['threshold']}, "
          f"{b['escalated'] * 100:.0f}% escalated, ${b['cost_per_1k_usd']:.2f}/1k  (NOT reportable)")
    print(f"  cascade CROSS-FIT   {p(cf['macro_bacc_mean'])} "
          f"[{p(cf['macro_bacc_ci'][0])}, {p(cf['macro_bacc_ci'][1])}], "
          f"{cf['escalated_mean'] * 100:.0f}% escalated, ${cf['cost_per_1k_usd_mean']:.2f}/1k")
    print(f"  selection optimism  {cf['selection_optimism_pp']:.2f} pp")
    o = out["cascade"]["oracle"]
    print(f"  oracle routing      {p(o['macro_bacc'])} at {o['escalated'] * 100:.0f}% escalated "
          f"({o['headroom_over_jev_pp']:.1f} pp of headroom over Jev)")
    print(f"\nwrote {ARXIV_JSON.relative_to(REPO)}")
    print(f"wrote {CROSSFIT_JSON.relative_to(REPO)} and {MINICHECK_JSON.relative_to(REPO)} (views)")
    print(f"wrote {MEDIUM_JSON.relative_to(REPO)} (published article evidence, schema frozen)")


if __name__ == "__main__":
    raise SystemExit(main())

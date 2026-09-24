"""The claims the paper makes about its own evidence, checked against numbers.json.

Each test here corresponds to a sentence in the paper that a reviewer can check from the
released artifacts, and exists because the sentence was wrong once:

* the gating table quoted a macro average whose denominator had changed, with no column
  saying so;
* the headline table printed no false-verification rate for the cascade it recommends;
* the false-verification gap reached the abstract as a bare point estimate while the
  accuracy gap carried an interval and a test;
* the MiniCheck arm was reported without the published score it was meant to reproduce;
* the tightest cascade row was called better than the frontier arm on the guard metric on
  the strength of a one-claim margin with no interval and no test;
* section 12 attributed the prompt-parity flips to the prompt on the strength of a repeat
  that covers none of them;
* the abstract grew past the length arXiv will accept.

    docker compose run --rm dev uv run pytest tests/test_paper_evidence.py -v
"""

import math
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _export import repo_only  # noqa: E402

# Every test here reads arxiv/source/, arxiv/outline.md or scripts/verify/, none of
# which is in the released package. They ship so the file-by-file comparison in the
# builder stays true, and they skip there rather than fail. See tests/_export.py.
pytestmark = repo_only

REPO = Path(__file__).resolve().parents[1]
NUMBERS_JSON = REPO / "arxiv/generated/numbers.json"
NUMBERS_TEX = REPO / "arxiv/generated/numbers.tex"


@pytest.fixture(scope="module")
def out():
    return json.loads(NUMBERS_JSON.read_text())


@pytest.fixture(scope="module")
def macros():
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", NUMBERS_TEX.read_text()))


def test_every_gating_row_reports_the_denominator_of_its_macro_average(out, macros):
    """A tight gate can empty a gold class in a dataset, which drops it out of the macro
    average. The count is reported per row so the table can carry it as a column."""
    n_datasets = out["n_datasets"]
    words = {0.5: "Fifty", 0.7: "Seventy", 0.8: "Eighty", 0.9: "Ninety",
             0.95: "NinetyFive", 0.97: "NinetySeven", 0.99: "NinetyNine"}
    for g in out["gating"]:
        assert g["n_datasets_scored"] + g["n_datasets_dropped"] == n_datasets
        assert len(g["datasets_dropped"]) == g["n_datasets_dropped"]
        name = f"GateAt{words[g['threshold']]}NDatasets"
        assert macros[name] == str(g["n_datasets_scored"]), (
            f"{name} must travel with GateAt{words[g['threshold']]}BAcc so the table can "
            "print the denominator beside the number")


def test_a_gating_row_whose_denominator_changed_is_visible_in_the_evidence(out):
    """If this ever stops being true the caveat in section 8 can go, and not before."""
    changed = [g for g in out["gating"] if g["n_datasets_dropped"]]
    if not changed:
        pytest.skip("no gating row loses a dataset on the current data")
    for g in changed:
        assert g["n_datasets_scored"] < out["n_datasets"]
        assert g["macro_bacc"] is not None


def test_the_cascade_reports_a_cross_fitted_guard_metric(out, macros):
    cf = out["cascade"]["crossfit"]
    for key in ("false_verification_rate_mean", "false_verification_rate_ci",
                "verified_precision_mean"):
        assert key in cf, f"the cross-fitted cascade must carry {key}"
    lo, hi = cf["false_verification_rate_ci"]
    assert lo <= cf["false_verification_rate_mean"] <= hi
    assert macros["CascadeCrossfitFVR"] == f"{cf['false_verification_rate_mean'] * 100:.1f}"


def test_the_cascade_is_the_more_permissive_guard_and_the_paper_says_so(out):
    """The cascade inherits the cheap system's verdicts where it does not escalate, so
    its false-verification rate sits above the frontier arm's. The sign of this is a
    claim in sections 9, 11 and 13, so it is asserted rather than assumed."""
    cf = out["cascade"]["crossfit"]
    astra_fvr = out["configurations"]["astra"]["false_verification_rate"]
    jev_fvr = out["configurations"]["jev"]["false_verification_rate"]
    assert astra_fvr < cf["false_verification_rate_mean"] < jev_fvr
    assert cf["minus_astra_fvr_pp_mean"] > 0


def test_the_false_verification_gap_carries_an_interval_and_a_test(out, macros):
    fi = out["complementarity"]["fvr_gap_inference"]
    assert fi["n_gold_unsupported"] == out["n_gold_unsupported"]
    # the four cells of the paired table account for every gold-unsupported claim
    total = (fi["a_only_false_verifies"] + fi["b_only_false_verifies"]
             + fi["both_false_verify"] + fi["neither_false_verifies"])
    assert total == fi["n_gold_unsupported"]
    # and the cells reproduce each system's reported rate
    jev = out["configurations"]["jev"]["false_verification_rate"]
    astra = out["configurations"]["astra"]["false_verification_rate"]
    n = fi["n_gold_unsupported"]
    assert (fi["a_only_false_verifies"] + fi["both_false_verify"]) / n == pytest.approx(jev)
    assert (fi["b_only_false_verifies"] + fi["both_false_verify"]) / n == pytest.approx(astra)
    lo, hi = fi["gap_ci_pp"]
    assert lo < fi["gap_pp"] < hi
    assert 0.0 <= fi["mcnemar_p"] <= 1.0
    for name in ("FVRGapCILo", "FVRGapCIHi", "FVRGapMcNemarP"):
        assert name in macros, f"the paper needs \\{name} to report the gap honestly"


def test_the_minicheck_arm_carries_the_published_score_it_failed_to_reach(out):
    """MiniCheck is the external harness-validity anchor (threats-to-validity C4), so the
    number it was supposed to reproduce is evidence and is read from the manifest."""
    if not out["minicheck_scored"]:
        pytest.skip("MiniCheck arm not scored in this run")
    pr = out["configurations"]["minicheck"]["published_reference"]
    manifest = json.loads((REPO / pr["manifest"]).read_text())
    row = manifest["models"][pr["model_row"]]
    assert pr["macro_bacc"] * 100 == pytest.approx(row["scores"][manifest["columns"].index("Average")])
    assert str(round(pr["macro_bacc"] * 100, 1)) in manifest["harness_validation_target"]
    assert pr["shortfall_pp"] == pytest.approx(
        (pr["macro_bacc"] - out["configurations"]["minicheck"]["macro_bacc"]) * 100)


def test_the_sample_id_list_is_named_and_hashed(out):
    """Section 5 and section 12 say the sample is a released list rather than a draw any
    command regenerates, so the list has to be identified and hashed in the evidence."""
    s = out["provenance"]["sample_id_list"]
    assert (REPO / s["path"]).is_file()
    assert re.fullmatch(r"[0-9a-f]{64}", s["sha256"])
    assert s["n"] == out["n_paired"]
    # The hash lives in input_hashes.json, not in raw_hashes.json. raw_hashes.json is
    # covered by an OpenTimestamps proof and by the pushed tag results-lock-arxiv-v1, so
    # adding anything to it invalidates the proof; that happened once and is in
    # docs/reference/incident-log.md. Auxiliary inputs are hashed in their own file, and
    # this test holds the separation as well as the hash.
    aux = json.loads((REPO / "arxiv/evidence/input_hashes.json").read_text())
    assert aux["inputs"][s["path"]] == s["sha256"]
    lock = json.loads((REPO / "arxiv/evidence/raw_hashes.json").read_text())
    assert s["path"] not in lock.get("configs", {})
    assert s["path"] not in lock.get("prediction_files", {})


def test_the_tightest_cascade_row_does_not_separate_on_the_guard_metric(out, macros):
    """The paper points at the tightest sweep row when it narrows its own safety caveat, so
    that row's guard margin has to be reported as what it is. Section 9.3 requires an
    interval and a test before a difference on this metric may be believed, and the same
    requirement binds a difference the paper likes."""
    row = max(out["cascade"]["in_sample_sweep"], key=lambda r: r["threshold"])
    astra = out["configurations"]["astra"]
    n = out["n_gold_unsupported"]
    assert row["n_gold_unsupported"] == n
    assert row["astra_n_false_verified"] == astra["n_false_verified"]
    # the counts reproduce the rates the table prints
    assert row["n_false_verified"] / n == pytest.approx(row["false_verification_rate"])
    assert astra["n_false_verified"] / n == pytest.approx(astra["false_verification_rate"])
    # the margin is a claim count, and the discordant cells are what the test sees
    margin = astra["n_false_verified"] - row["n_false_verified"]
    assert margin == row["fv_vs_astra_astra_only"] - row["fv_vs_astra_cascade_only"]
    # whatever the margin turns out to be, the paper may not quote this row against the
    # frontier arm without the test beside it
    assert 0.0 <= row["fv_vs_astra_mcnemar_p"] <= 1.0
    for name in ("CascadeAtNinetyNineFVRCount", "AstraFVRCount",
                 "CascadeAtNinetyNineFVRVsAstraMcNemarP",
                 "CascadeAtNinetyNineFVRVsAstraCascadeOnly",
                 "CascadeAtNinetyNineFVRVsAstraAstraOnly"):
        assert name in macros, f"the paper needs \\{name} to quote that row honestly"
    # and the high-effort frontier configuration is the other thing a reader will check
    assert out["configurations"]["astra_high"]["n_false_verified"] <= astra["n_false_verified"]


def test_the_same_prompt_repeat_covers_none_of_the_prompt_parity_flips(out, macros):
    """Section 12 may describe where the flips sit; it may not attribute them to the
    prompt. The repeat's coverage is the reason, and it is evidence rather than prose: all
    the flips are outside the compared rows, and the compared rows are the rows on which
    the configuration reports no internal reasoning."""
    sp = out["same_prompt_repeat"]
    assert sp["n_compared"] + sp["n_capped_no_answer"] == sp["n_paired"]
    # every flip is outside the set the repeat measured, so the repeat is silent on them
    assert sp["parity_flips_in_capped_no_answer"] == sp["parity_flips"]
    assert sp["prompt_edit_flips_on_compared"] == 0
    # and the measured set is the set where there is least for a fresh draw to move
    assert sp["compared_reasoning_zero_n_reported"] > 0.9 * sp["n_compared"]
    assert sp["excluded_reasoning_nonzero_n_reported"] > 0.5 * sp["excluded_reasoning_n"]
    assert sp["parity_flip_reasoning_mean_reported"] > sp["compared_reasoning_mean_reported"]
    # both passes carry a prompt no reported configuration used, and the paper says so
    assert sp["prompt_hash"] != sp["reported_prompt_hash"]
    for name in ("SamePromptRepeatComparedReasoningZero",
                 "SamePromptRepeatComparedReasoningMean",
                 "SamePromptRepeatExcludedReasoningNonzero",
                 "SamePromptRepeatPromptEditFlipsOnCompared",
                 "PromptParityFlipsReasoningMean"):
        assert name in macros, f"section 12 needs \\{name} to state its coverage"


def test_both_frontier_arms_are_measured_for_the_prompt_edit(out, macros):
    """One prompt edit was made and both frontier configurations were re-run under it, so
    there are two pre-correction runs. For several drafts only the low-effort pair was
    measured, and section 12 generalised its result: no flip there landed on a
    gold-unsupported claim, so it certified every separated comparison as undisturbed by
    the correction. On the high-effort arm a flip did land on one, and that arm's
    false-verification rate is a leg of one of the two Holm survivors, so the
    certification was an extrapolation across the arm where it was false.

    This holds both arms to being measured, and holds the survivor to being recomputed
    against the superseded run rather than asserted to be unaffected.
    """
    lo, hi = out["prompt_parity"], out["prompt_parity_high"]
    assert lo["arm"] == "astra" and hi["arm"] == "astra_high"
    assert lo["path"] != hi["path"], "two arms, two superseded runs, two measurements"
    assert lo["n"] == hi["n"] == out["n_paired"], "both cover the whole paired frame"
    # The asymmetry that makes the second measurement necessary, from the data.
    assert lo["n_flips_on_gold_unsupported"] == 0
    assert hi["n_flips_on_gold_unsupported"] > 0, (
        "if this ever becomes zero the prose below has to change with it, because the "
        "whole point of the paragraph is that the two arms behaved differently")
    assert hi["pre_false_verification_rate"] != hi["post_false_verification_rate"]

    # The survivor recomputed against the run the correction replaced, not asserted.
    xf = out["cascade"]["crossfit"]
    base = xf["vs_astra_high_inference"]["fvr_inference"]
    pre = xf["vs_astra_high_pre_parity_inference"]["fvr_inference"]
    assert pre["n_gold_unsupported"] == base["n_gold_unsupported"]
    assert pre["n_repeats_p_below_05"] == base["n_repeats_p_below_05"], (
        "the claim section 12 makes is that the survivor separates on either run")
    # and the reported figure is the conservative one, which is what the prose says
    assert base["fvr_delta_pp"] < pre["fvr_delta_pp"]

    for name in ("PromptParityHighFlips", "PromptParityHighFlipsOnUnsupported",
                 "PromptParityHighPreFVR", "PromptParityHighPostFVR",
                 "PromptParityHighPreBAcc", "PromptParityHighPostBAcc",
                 "CascadeAstraHighFVRPreParityDeltaPP",
                 "CascadeAstraHighFVRPreParityMcNemarPMedian",
                 "CascadeAstraHighFVRPreParityRepeatsPBelowFive"):
        assert name in macros, f"section 12 needs \\{name} to report the second arm"

    # The sentence that was wrong is gone, and cannot come back by that spelling.
    src = _paper_sources()
    flat = re.sub(r"\s+", " ", "\n".join(src.values()))
    assert "uncorrected are therefore undisturbed" not in flat, (
        "the separated results are not undisturbed by the correction: it moved the "
        "high-effort arm's false-verification rate, which is a leg of a survivor")
    assert "\\PromptParityHighFlipsOnUnsupported" in flat, (
        "section 12 must print the flip that landed on a gold-unsupported claim")

    # The pre-parity recomputation prints an exact McNemar p that is not in the Holm
    # family and is not one of the separated results. Both exclusions are decisions, so
    # the paper states them where it fixes the family rather than leaving a reader to
    # find an unadjusted fourteenth p-value and conclude the family is porous.
    assert "neither a member of the family nor one" in flat, (
        "the paper must say that the pre-parity recomputation sits outside the Holm "
        "family and outside the count of comparisons that clear both instruments")


def test_the_abstract_fits_the_length_arxiv_accepts():
    """arXiv refuses an abstract over 1920 characters
    (https://info.arxiv.org/help/prep.html), and this abstract is written against macros,
    so its source length says nothing: only the expanded text can be measured. The
    measurement lives in scripts/verify/check_paper_macros.py so that one implementation
    is both the gate and what this test reads."""
    gate = REPO / "scripts/verify/check_paper_macros.py"
    assert "ABSTRACT_LIMIT = 1920" in gate.read_text(), "the published limit must be the gate"
    r = subprocess.run([sys.executable, str(gate)], capture_output=True, text=True, cwd=REPO)
    m = re.search(r"abstract: (\d+) characters expanded, limit (\d+)", r.stdout)
    assert m, f"the macro gate no longer reports the abstract length:\n{r.stdout}"
    length, limit = int(m.group(1)), int(m.group(2))
    assert limit == 1920
    assert length <= limit, (
        f"the abstract expands to {length} characters; arXiv will refuse it")
    assert r.returncode == 0, r.stdout


def test_the_draft_is_the_size_the_approved_outline_says():
    """The paper grew to 2.5x the size Alden approved, and section 12 to 8.5x its budget,
    because nothing in this repository read arxiv/outline.md's word budgets. The
    measurement lives in scripts/verify/check_paper_length.py so that one implementation
    is both the gate and what this test reads, and the budgets live in the outline so
    that the approved document stays the single source of truth."""
    gate = REPO / "scripts/verify/check_paper_length.py"
    assert gate.is_file(), "the length gate is what stops the draft drifting off the outline"
    r = subprocess.run([sys.executable, str(gate)], capture_output=True, text=True, cwd=REPO)

    # the budgets have to come out of the outline, not out of the gate
    outline = (REPO / "arxiv/outline.md").read_text()
    assert "| 12 |" in outline and "| 400 |" in outline, (
        "the outline no longer carries a per-section word budget table for the gate to read")
    for n in ("1", "12", "13"):
        assert re.search(rf"^\s*{n}\s+\d+\s+\d+\s", r.stdout, re.M), (
            f"the gate does not report section {n} against a budget:\n{r.stdout}")

    total = re.search(r"^\s+(\d+)\s+(\d+)\s+SECTIONS 1\.\.N", r.stdout, re.M)
    assert total, f"the gate no longer reports a section total:\n{r.stdout}"
    words, budget = int(total.group(1)), int(total.group(2))
    assert budget == 7600, "the outline's section budgets sum to 7600 words"
    assert words <= int(budget * 1.10), (
        f"sections 1..N are {words} words against the outline's {budget}; cut prose or "
        f"move disclosure detail into the appendices")

    # The two sides of that comparison must be drawn from the same rows. The gate used to
    # add the abstract's words to a budget that is the sum of the SECTION rows, and the
    # outline gives the abstract no budget at all, so for several rounds sections 1..N were
    # held to roughly 260 words less than the approved outline allows and prose was cut to
    # meet a number nobody approved. The abstract is still hard-gated, on arXiv's published
    # 1920-character limit, which is a different unit and a different gate.
    abstract_row = re.search(r"^\s*-\s+(\d+)\s+-\s+abstract", r.stdout, re.M)
    assert abstract_row, f"the gate must still report the abstract's length:\n{r.stdout}"
    assert "not charged" in r.stdout, (
        "the abstract must be reported as not charged against the section budgets, so a "
        "reader of this output cannot mistake one budget for the other")
    assert int(abstract_row.group(1)) > 0
    assert "ABSTRACT_LIMIT = 1920" in (REPO / "scripts/verify/check_paper_macros.py").read_text(), (
        "the abstract's own limit has to stay enforced somewhere, or excluding it here "
        "would leave it unguarded")
    assert r.returncode == 0, r.stdout


# Section 12's list of limitations, in the order it states them, paired with the Appendix E
# paragraph or paragraphs that answer each one. Section 12 tells the reader the appendix
# carries the numbers "in this order" and the appendix repeats the claim, so the order is
# an assertion in the rendered PDF and it gets a test. For one round it was false: the
# context-length paragraph sat ninth in the appendix against eleventh in the list, ahead of
# five items that precede it, and the paragraph on the cascade's unstable sign answered no
# item in the list at all. The old test checked only that four topic words appeared
# somewhere in the appendix, so it saw none of that.
SECTION_TWELVE_ITEMS = [
    ("No equivalence is claimed or tested.",
     ["Indistinguishable is not equivalent.",
      # Renamed when the cascade gained its comparison against the high-effort arm: three
      # comparisons now clear both instruments uncorrected and two survive the correction,
      # so a heading that counted two differences and one failure counted neither side.
      "The \\NClearingUncorrected\\ differences that clear both instruments, and what the correction does to "
      "them.",
      "Raw and adjusted $p$ for the whole family."]),
    ("One threshold is fitted out of fold and nothing else is:",
     ["What was selected in sample, and what was not."]),
    ("The cross-fitted cascade's sign is unstable too, below Astra on accuracy",
     ["The cascade's sign is unstable in two different ways.",
      # Section 10 used to carry this paragraph in the body, where it read as method
      # rather than as a limitation and cost the body its word budget. It answers the
      # same item: both are about what one draw of the fold assignment decides.
      "One draw of the fold assignment."]),
    ("The evaluation sample is a released list of example IDs no code regenerates,",
     ["The evaluation sample is a released list, not a rerunnable draw."]),
    ("The cascade is an offline replay, not a deployed router;",
     ["The cascade is an offline replay."]),
    ("Costs are measured tokens at archived list prices, not invoices;",
     ["Every cost figure is two price lists on one day."]),
    ("Each arm ran once;", ["Each arm was run once."]),
    ("No latency figure appears, the bulk runs being concurrent.",
     ["Latency was not measured under control."]),
    ("Contamination cannot be ruled out, and the frontier side is one model family",
     ["Contamination cannot be ruled out.",
      "One frontier family, two reasoning efforts."]),
    ("A third arm was excluded:",
     ["An arm we ran and excluded.",
      "The published reference we quote is a partial capture."]),
    ("The calibration comparison has one participant,",
     ["The calibration comparison has one participant."]),
    ("Nothing was truncated and no example dropped for length,",
     ["No example was excluded for length, and nothing was truncated."]),
    ("The harness was written with AI coding assistance;",
     ["The harness was built with AI coding assistance."]),
    # The conflict-of-interest subsection, which follows the list in section 12 and sends
    # the reader to the appendix for the order the TypeSafe documents happened in. It is
    # here because the COI paragraph named "the decision that followed" a clause check and
    # shipped a decision dated two days BEFORE that check, with no account of the order
    # anywhere in the paper.
    ("For TypeSafe the released package carries the clause",
     ["The TypeSafe record, in order."]),
]


def test_section_twelve_opens_by_naming_the_size_of_the_study(out, macros):
    """Section 12 is the section the paper points at for every hedge it owes a reader, and
    it opened by naming the split and not the sample: 'Every number here comes from a
    public benchmark's development split'. A reader had to go back to Section 5 to learn
    that the whole paper rests on 495 claims, roughly 45 per dataset, with a thinnest
    minority class of four. The size belongs in the first sentence of the section that
    carries the limitations, not only in the section that describes the draw."""
    close = (REPO / "arxiv/source/sections/close.tex").read_text()
    body = close.split("\\appendix", 1)[0]
    section = body.split("\\section{Limitations and Conflict of Interest}", 1)[1]
    opening = re.sub(r"\s+", " ", section.split("\\subsection", 1)[0]).strip()
    first = opening.split(";")[0]
    assert "\\NPaired" in first, (
        "section 12's first sentence states the split and not the size of the study: "
        f"{first!r}")
    assert macros["NPaired"] == str(out["n_paired"])
    assert "development split" in first, "it must still say which split these numbers are"


def test_section_twelve_is_a_list_and_appendix_e_is_the_accounting():
    """Section 12 is budgeted 400 words and the disclosure detail runs to several
    thousand, so the detail lives in Appendix~\\ref{app:limitations}. This test fails if
    the detail creeps back into the body or the appendix stops existing, which is how the
    8.5x overrun happened the first time."""
    close = (REPO / "arxiv/source/sections/close.tex").read_text()
    body, appendix = close.split("\\appendix", 1)
    assert "\\label{app:limitations}" in appendix, (
        "Appendix E carries the limitations accounting and section 12 points at it")
    assert "\\ref{app:limitations}" in body, "section 12 must send the reader to the appendix"
    for topic in ("MiniCheck", "OpenTimestamps", "contamination", "Contamination"):
        assert topic in appendix or topic.lower() in appendix.lower(), (
            f"the {topic} disclosure belongs in the appendix, not dropped")


def test_appendix_e_follows_section_twelve_in_the_order_both_of_them_claim():
    """Both halves say the appendix follows the list's order. This checks it, item by item,
    and checks that no Appendix E paragraph answers nothing in the list.

    An order claim a reader can falsify by counting paragraphs is worse than no order
    claim, and this paper makes it twice in the rendered PDF.
    """
    close = (REPO / "arxiv/source/sections/close.tex").read_text()
    body, after_appendix = close.split("\\appendix", 1)
    # Appendix E only: the incidents appendix carries \paragraph commands of its own.
    app_e = after_appendix.split("\\label{app:limitations}", 1)[1]
    body_flat = re.sub(r"\s+", " ", body)
    app_flat = re.sub(r"\s+", " ", app_e)

    assert "in this order" in body_flat, (
        "section 12 must state the order this test enforces, or stop claiming one")
    assert "in the same order" in app_flat, (
        "Appendix E must state the order this test enforces, or stop claiming one")

    item_positions, paragraph_positions, expected = [], [], []
    for anchor, paragraphs in SECTION_TWELVE_ITEMS:
        flat_anchor = re.sub(r"\s+", " ", anchor)
        assert flat_anchor in body_flat, (
            f"section 12 no longer carries the list item {flat_anchor!r}; the appendix "
            "order claim is about this list, so update both together")
        item_positions.append((body_flat.index(flat_anchor), flat_anchor))
        for heading in paragraphs:
            cmd = "\\paragraph{" + heading + "}"
            assert cmd in app_flat, f"Appendix E no longer carries {cmd!r}"
            paragraph_positions.append((app_flat.index(cmd), cmd))
            expected.append(cmd)

    assert [a for _, a in sorted(item_positions)] == [a for _, a in item_positions], (
        "the list items in SECTION_TWELVE_ITEMS are not in the order section 12 prints "
        "them, so the mapping this test enforces is stale")
    in_file = [c for _, c in sorted(paragraph_positions)]
    assert in_file == expected, (
        "Appendix E's paragraphs do not run in the order section 12's list runs, and both "
        f"halves tell the reader they do.\nappendix order: {in_file}\nlist order: {expected}")

    # And nothing in Appendix E answers an item the list does not raise.
    found = re.findall(r"\\paragraph\{[^}]*\}", app_flat)
    assert found == expected, (
        "an Appendix E paragraph answers no item in section 12's list, or an item's "
        f"paragraph was dropped.\nin the appendix: {found}\nmapped to the list: {expected}")


def _separates(ci, p, same_estimand):
    """The paper's criterion, from Section 5, applied per estimand.

    Where the bootstrap and the test address the same quantity, which is every
    false-verification comparison because the rate is pooled over the gold-unsupported
    claims and McNemar runs on exactly those claims, both must clear: the interval
    excludes zero AND p < 0.05. Where they do not, which is every macro-balanced accuracy
    comparison because the interval is on the macro average while McNemar is on pooled
    per-item correctness, the interval governs alone; requiring the p-value there would be
    the incoherence Section 5 warns against. Either way, one instrument clearing on its own
    makes the result marginal and it is not counted.
    """
    lo, hi = sorted(ci)
    excludes_zero = lo > 0 or hi < 0
    return excludes_zero and (p < 0.05 if same_estimand else True)


def test_exactly_three_comparisons_separate_from_noise_and_the_paper_says_three():
    """Section 12 once said in one paragraph that two comparisons separate from noise and
    fourteen paragraphs later that one does, while Section 9.4 reported a third interval
    excluding zero. Nothing measured it. This counts them from numbers.json under the
    criterion Section 5 states, and checks the paper against the count.

    The count moved from two to three when the cascade was finally compared with the
    high-effort frontier arm, the row printed directly beneath it in Table 4. That
    comparison separates on the guard metric and goes against the architecture this paper
    proposes, which is why it was the one neighbour the paper had never differenced.
    """
    d = json.loads(NUMBERS_JSON.read_text())
    xf = d["cascade"]["crossfit"]
    gap = d["complementarity"]["fvr_gap_inference"]

    comparisons = {
        "jev vs astra, macro BAcc": (
            d["pairs"]["jev|astra"]["macro_bacc_delta_ci_pp"],
            d["pairs"]["jev|astra"]["mcnemar_p"]),
        "astra vs astra_high, macro BAcc": (
            d["pairs"]["astra|astra_high"]["macro_bacc_delta_ci_pp"],
            d["pairs"]["astra|astra_high"]["mcnemar_p"]),
        "jev vs astra_high, macro BAcc": (
            d["pairs"]["jev|astra_high"]["macro_bacc_delta_ci_pp"],
            d["pairs"]["jev|astra_high"]["mcnemar_p"]),
        "jev vs astra, FVR": (gap["gap_ci_pp"], gap["mcnemar_p"]),
        "cascade vs astra, macro BAcc": (
            xf["vs_astra_inference"]["macro_bacc_delta_ci_pp"],
            xf["vs_astra_inference"]["mcnemar_p_median"]),
        "cascade vs jev, macro BAcc": (
            xf["vs_jev_inference"]["macro_bacc_delta_ci_pp"],
            xf["vs_jev_inference"]["mcnemar_p_median"]),
        "cascade vs astra, FVR": (
            xf["vs_astra_inference"]["fvr_inference"]["fvr_delta_ci_pp"],
            xf["vs_astra_inference"]["fvr_inference"]["mcnemar_p_median"]),
        "cascade vs jev, FVR": (
            xf["vs_jev_inference"]["fvr_inference"]["fvr_delta_ci_pp"],
            xf["vs_jev_inference"]["fvr_inference"]["mcnemar_p_median"]),
        "cascade vs astra_high, macro BAcc": (
            xf["vs_astra_high_inference"]["macro_bacc_delta_ci_pp"],
            xf["vs_astra_high_inference"]["mcnemar_p_median"]),
        "cascade vs astra_high, FVR": (
            xf["vs_astra_high_inference"]["fvr_inference"]["fvr_delta_ci_pp"],
            xf["vs_astra_high_inference"]["fvr_inference"]["mcnemar_p_median"]),
    }
    separated = sorted(n for n, (ci, p) in comparisons.items()
                       if _separates(ci, p, same_estimand=n.endswith("FVR")))
    assert separated == ["cascade vs astra, FVR", "cascade vs astra_high, FVR",
                         "jev vs astra, FVR"], separated
    # The per-estimand rule must not let an accuracy comparison in through the back door:
    # none of them has an interval excluding zero, so the count is the same either way,
    # and this holds that fact rather than relying on it.
    for name, (ci, _) in comparisons.items():
        if name.endswith("BAcc"):
            lo, hi = sorted(ci)
            assert lo < 0 < hi, f"{name} interval no longer straddles zero; recount"

    # the one that clears the interval and fails the test is the one the paper calls
    # marginal, and it must not be counted as a third
    jev_fvr = xf["vs_jev_inference"]["fvr_inference"]
    lo, hi = sorted(jev_fvr["fvr_delta_ci_pp"])
    assert hi < 0, "the cascade against Jev guard interval excludes zero"
    assert jev_fvr["mcnemar_p_median"] >= 0.05, "and its test does not clear 0.05"

    # The printed count and this inventory must be the same number. They were not: a glob
    # in _n_clearing_uncorrected took in the pre-parity sensitivity block, so the macro
    # said four while this test said three and both passed, because nothing compared them.
    assert _macros()["NClearingUncorrected"] == str(len(separated)), (
        f"the paper prints {_macros()['NClearingUncorrected']} comparisons clearing both "
        f"instruments; this inventory finds {len(separated)}: {separated}")
    assert str(d["multiplicity"]["n_clearing_uncorrected"]) == str(len(separated))

    # And the comparisons counted must come from the family the very next clause divides
    # them into. The pre-parity block is a sensitivity check on a superseded run: it is
    # not in the Holm family, so it cannot be in the count either.
    raw = d["multiplicity"]["raw"]
    assert not any("preparity" in k.lower() or "pre_parity" in k.lower() for k in raw), (
        "the Holm family has gained a pre-parity member; the count above and the family "
        "must stay the same universe")
    assert all(not k.startswith("vs_") for k in raw), (
        "the Holm family is named by comparison, not by crossfit key; if that changes, "
        "the pre-parity check above has to be rewritten against the new names")


def test_every_crossfit_comparator_block_is_classified_as_reported_or_not():
    """Which cross-fitted comparators are results is a decision, not a glob.

    `_n_clearing_uncorrected` used to select them with `vs_*_inference`. When the
    pre-parity sensitivity block was added, the glob promoted a comparison against a run
    the paper itself calls superseded into the count of results that clear both of its
    instruments, one sentence before the paper says no other reported configuration
    clears. This holds the inventory: every block in numbers.json is either reported, and
    then it is one of the three this file enumerates, or it is declared a non-result in
    the analysis code with the reason.
    """
    sys.path.insert(0, str(REPO / "scripts/verify"))
    import paired_analysis as PA  # noqa: E402

    d = json.loads(NUMBERS_JSON.read_text())
    blocks = sorted(k for k in d["cascade"]["crossfit"]
                    if k.startswith("vs_") and k.endswith("_inference"))
    classified = set(PA.REPORTED_INFERENCE_BLOCKS) | set(PA.NONRESULT_INFERENCE_BLOCKS)
    assert set(blocks) <= classified, (
        f"unclassified comparator block(s): {sorted(set(blocks) - classified)}")
    assert sorted(PA.REPORTED_INFERENCE_BLOCKS) == [
        "vs_astra_high_inference", "vs_astra_inference", "vs_jev_inference"], (
        "the reported comparators are the three the enumeration test counts; adding one "
        "means updating that test, the Holm family and the prose that enumerates them")
    for key, reason in PA.NONRESULT_INFERENCE_BLOCKS.items():
        assert len(reason) > 80, f"{key} must carry the reason it is not a result"
    # The selection has to bite. While a non-result block is on disk, the old glob and the
    # named selection must give different answers, or this guard is decoration.
    xf = d["cascade"]["crossfit"]
    selected = PA._reported_inference_blocks(xf)
    assert set(selected) == set(blocks) & set(PA.REPORTED_INFERENCE_BLOCKS)
    if set(blocks) & set(PA.NONRESULT_INFERENCE_BLOCKS):
        assert len(selected) < len(blocks), (
            "a non-result block is present and the selection still takes everything")
    # An unclassified block must stop the run rather than be counted.
    with pytest.raises(AssertionError):
        PA._reported_inference_blocks(dict(xf, vs_something_new_inference={}))

    paper = "\n".join((REPO / "arxiv/source" / p).read_text() for p in
                      ("main.tex", "sections/setup.tex", "sections/same-score.tex",
                       "sections/different-errors.tex", "sections/cascade.tex",
                       "sections/close.tex"))
    flat = re.sub(r"\s+", " ", paper)
    assert "the one result here that separates from noise" not in flat
    assert "the only other comparison anywhere here" not in flat
    # The count is a macro now, not a word, because it drifted twice as a word. These
    # assert the macro is cited where the count is stated; the word form is banned by
    # test_the_separated_count_is_a_macro_everywhere_and_never_a_word.
    assert "\\NClearingUncorrected\\ comparisons meet the separation criterion" in flat, (
        "section 12 states the count, and states it from the macro")
    assert "\\NClearingUncorrected\\ comparisons meet it uncorrected" in flat, (
        "the appendix states the same count under the stated criterion, from the macro")
    # And the Holm result, because the count of three is the uncorrected count and the spec
    # requires the correction. The appendix must name how many survive it.
    assert "survive the Holm-Bonferroni adjustment" in flat, (
        "the appendix must say how many of the three survive the correction the spec "
        "requires")
    # The survivors are counted from the data rather than from the sentence, and the
    # sentence prints the count as a macro.
    mp = json.loads(NUMBERS_JSON.read_text())["multiplicity"]
    scoped = mp["survives_at_05_excluding_minicheck"]
    assert _macros()["MultiplicitySurvivingN"] == str(len(scoped)) == "2", scoped
    assert "CascadeAstraHighFVR" in scoped, (
        "the cascade's guard penalty against the high-effort arm no longer survives Holm, "
        "so Section 9 and Appendix E have to be rewritten rather than left claiming it")
    # Once, in one place. The appendix used to paraphrase it as an unconditional
    # conjunction of interval and McNemar p, which is the form Section 5 explicitly
    # declines to apply to macro-balanced accuracy, and cited Section 5 for it. A
    # substring test passed with both statements in the paper, so it counts them.
    assert flat.count("a comparison separates from noise when the") == 1, (
        "Section 5 must state the criterion once, in one place; a second statement of it "
        "elsewhere is how the paper came to hold two incompatible inference rules")
    assert "fixes the criterion per estimand" in flat, (
        "the appendix that counts the separated comparisons must carry the criterion in "
        "its per-estimand form, not as the unconditional conjunction")


def _paper_sources():
    return {p: (REPO / "arxiv/source" / p).read_text() for p in
            ("main.tex", "sections/setup.tex", "sections/same-score.tex",
             "sections/different-errors.tex", "sections/cascade.tex", "sections/close.tex")}


def test_the_random_escalation_control_is_never_called_out_of_sample():
    """The abstract said the operating point was 'evaluated out-of-sample by cross-fitting
    against a matched random-escalation control' while Table 9's own caption said none of
    that control is an out-of-sample operating point. No out-of-sample control exists: the
    control rows sit at fixed in-sample thresholds and none of them escalates the share the
    cross-fitted point escalates."""
    d = json.loads(NUMBERS_JSON.read_text())
    xf_escalated = d["cascade"]["crossfit"]["escalated_mean"] * 100
    shares = [row["escalated"] * 100 for row in d["cascade"]["random_control"]]
    assert all("threshold" in row for row in d["cascade"]["random_control"]), (
        "every control row is keyed by a fixed full-sample threshold, which is what makes\n"
        "it in-sample")
    assert all(abs(s - xf_escalated) > 0.5 for s in shares), (
        "a control row now reproduces the cross-fitted escalation share, so the paper could\n"
        "say something stronger than it does; update the prose deliberately")

    src = _paper_sources()
    flat = re.sub(r"\s+", " ", src["main.tex"])
    assert "cross-fitting against a matched random-escalation control" not in flat, (
        "the abstract must not fuse the out-of-sample cross-fitting with the in-sample control")
    assert "random-escalation control at in-sample thresholds" in flat, (
        "the abstract must say where the control sits")
    cascade = re.sub(r"\s+", " ", src["sections/cascade.tex"])
    assert "none of it is an out-of-sample operating point" in cascade, (
        "Table 9's caption carries the statement the abstract is measured against")


def test_no_claim_about_seeds_the_package_never_varies():
    """The draft explained away the cross-fitted interval's top by asserting that at other
    seeds the coincidence disappears. No code here varies the seed, and a reviewer who did
    found the opposite at most seeds tried. Only the single-seed count is claimed."""
    d = json.loads(NUMBERS_JSON.read_text())
    xf = d["cascade"]["crossfit"]
    assert "n_repeats_reproducing_in_sample" in xf, (
        "the count is the only thing the paper may claim about this coincidence")
    share = xf["n_repeats_reproducing_in_sample"] / xf["n_repeats"]
    assert share > 0.025, (
        "the paper says the share exceeds the 2.5% a 97.5th percentile needs")

    banned = re.compile(r"at other seeds it falls below|the coincidence disappears|"
                        r"at neighbouring seeds it falls below", re.I)
    for name, text in _paper_sources().items():
        assert not banned.search(re.sub(r"\s+", " ", text)), (
            f"{name} asserts behaviour at seeds nothing in this package measures")
    code = (REPO / "scripts/verify/paired_analysis.py").read_text()
    assert not banned.search(re.sub(r"\s+", " ", code)), (
        "the shipped analysis comment asserts the same unmeasured thing")


def test_point_estimates_are_signed_the_way_their_intervals_are():
    """The abstract printed '0.6 points, 95% CI [-5.6, +4.5]' and Section 6 called -0.3
    with CI [-1.3, 0.7] 'an improvement of 0.3 points'. Where a point estimate is printed
    beside its interval it must use the signed macro."""
    d = json.loads(NUMBERS_JSON.read_text())
    assert d["pairs"]["jev|astra"]["macro_bacc_delta_pp"] < 0, (
        "the stored delta is Jev minus Astra and is negative")
    src = _paper_sources()
    for name, text in src.items():
        flat = re.sub(r"\s+", " ", text)
        for unsigned, signed in (("JevAstraDeltaPP", "JevAstraDeltaSignedPP"),
                                 ("AstraAstraHighDeltaPP", "AstraAstraHighDeltaSignedPP"),
                                 ("JevAstraHighDeltaPP", "JevAstraHighDeltaSignedPP")):
            for m in re.finditer(r"\\" + unsigned + r"(?![A-Za-z])", flat):
                window = flat[m.end():m.end() + 160]
                assert "CI" not in window[:60] and "interval of" not in window[:60], (
                    f"{name} prints \\{unsigned} beside an interval; use \\{signed}")
    flat_main = re.sub(r"\s+", " ", src["main.tex"])
    # Either signed spelling will do, and the two decimal one is what the abstract prints
    # now that it also prints the two operands: 73.27 against 73.83 must subtract to the
    # difference beside them. What this holds is that whichever spelling is used is the
    # SIGNED one and that the direction is named in the same breath.
    assert re.search(r"\\JevAstraDeltaSigned(PP|TwoDP)\$ points, Jev minus Astra",
                     flat_main), (
        "the abstract must give the direction with the signed estimate")


def test_the_arithmetic_gate_sees_across_a_float_and_not_only_within_a_paragraph():
    """Table 5 printed its Macro BAcc column as 73.3, 74.10, 73.8, 74.11, 82.9 while the
    subsection under it reported the difference of two of those cells as 0.27, which is
    neither 74.10 - 73.8 nor 74.1 - 73.8. The gate that exists for exactly that defect did
    not see it: a table environment is a paragraph of its own, so its per-paragraph rule
    never had an operand and a difference in the same window at once, and its own comment
    said so. This test reverts one cell of that table to the spelling the defect had and
    requires the gate to fail on it, naming the window that caught it. The paragraph window
    cannot: the float holds no difference macro."""
    gate = REPO / "scripts/verify/check_paper_macros.py"
    table = REPO / "arxiv/source/sections/cascade.tex"
    original = table.read_text()
    row = "Astra, standard effort       & \\AstraBAccTwoDP           &"
    assert row in original, (
        "Table 5's standard-effort row is not where this test expects it, so the gate this "
        "test exercises is no longer being exercised")
    try:
        table.write_text(original.replace(
            row, "Astra, standard effort       & \\AstraBAcc                &"))
        r = subprocess.run([sys.executable, str(gate)],
                           capture_output=True, text=True, cwd=REPO)
    finally:
        table.write_text(original)
    assert r.returncode != 0, (
        "the gate passes a table whose cells do not give the difference the same section "
        f"prints, which is the defect it exists to catch:\n{r.stdout}")
    assert "one section, floats included" in r.stdout, (
        f"the gate failed, but not on the window that crosses the float:\n{r.stdout}")
    assert "CascadeAstraDeltaPP" in r.stdout, r.stdout

    # and the paper as it stands passes, so the test above proves a gate and not a bug
    clean = subprocess.run([sys.executable, str(gate)],
                           capture_output=True, text=True, cwd=REPO)
    assert clean.returncode == 0, clean.stdout


def _macros():
    """The generated macro table, for the tests below that take no fixture."""
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}",
                           NUMBERS_TEX.read_text()))


def test_the_holm_hedge_reaches_the_sections_that_report_the_result():
    """The Holm correction was applied to the abstract, Section 5, Table 1 and Appendix E
    and not to Sections 9 and 11, where the cascade's guard penalty is actually reported.
    For one round the paper said in its front matter that the penalty 'sits on the 0.05
    boundary under Holm' and in its results that it 'meets the separation criterion on
    both instruments', with the retraction and the claim on one printed page. The first
    repair round's test asserted only that the NEW sentence existed, so nothing caught the
    surviving old one. This asserts both directions: the hedge is present wherever the
    result is reported, and no unqualified form of the claim survives anywhere."""
    src = _paper_sources()
    flat = re.sub(r"\s+", " ", "\n".join(src.values()))
    macros = _macros()

    # The corrected p is a real number above 0.05, so the hedge is required rather than
    # stylistic. If the data ever moves it below, this test fails and the prose is rewritten
    # deliberately instead of drifting.
    p_holm = float(macros["CascadeAstraFVRMcNemarPHolm"])
    assert p_holm > 0.05, (
        "the cascade's guard penalty now clears 0.05 under Holm, so every hedge this test "
        f"pins is stale; adjusted p is {p_holm}")

    # No unqualified assertion of separation for that comparison, in any of the three
    # places that carried one.
    for banned in (
        "points that meets the separation criterion of Section~\\ref{sec:setup} on both "
        "instruments",
        "the comparison against Astra meets the criterion of Section~\\ref{sec:setup}",
        "\\CascadeAstraFVRDeltaCIHi], on both instruments.",
        # Appendix E was headed "The two differences we do read as real" four lines above
        # conceding that only one of the two survives the correction, and eight paragraphs
        # later called them "the two results this paper calls separated". Section 5 defines
        # separated as surviving the adjustment, so both phrasings assert the retracted
        # count. The wording that replaced them, "clear both instruments", was already in
        # use at close.tex and cascade.tex and says exactly what the data supports.
        "this paper calls separated",
        "we do read as real",
    ):
        assert re.sub(r"\s+", " ", banned) not in flat, (
            f"an unqualified separation claim survives: {banned!r}")

    # And the qualification is present in each of the two body sections that report it,
    # not only in the front matter and the appendix.
    for path in ("sections/cascade.tex", "sections/close.tex"):
        body = re.sub(r"\s+", " ", src[path])
        assert "Holm" in body, f"{path} reports the guard penalty and never names Holm"
    cascade = re.sub(r"\s+", " ", src["sections/cascade.tex"])
    assert "uncorrected but not the Holm correction the same section requires" in cascade, (
        "Section 9 must carry the correction alongside the gap it reports")
    assert "\\CascadeAstraFVRMcNemarPHolm$ Holm-adjusted over the" in cascade, (
        "Table 9's caption must print the adjusted p next to the criterion it cites")
    close = re.sub(r"\s+", " ", src["sections/close.tex"])
    assert "clearing both instruments uncorrected but not the Holm correction" in close, (
        "Section 11, the section a practitioner reads, must carry the same qualification")


def test_no_sentence_says_the_paper_applies_no_multiple_comparison_correction():
    """Three sentences said it. Two were removed and the third, in Appendix E's list of
    cautions, survived fourteen lines after that same appendix printed 'p = 0.005
    adjusted' and twenty-five after it tabulated Holm over all eleven tests. The one
    surviving 'no correction' sentence in the paper is about the gate-threshold sweep and
    the per-dataset cells, neither of which is in the Holm family, so it is pinned here as
    allowed rather than left for the next round to delete by mistake."""
    src = _paper_sources()
    flat = re.sub(r"\s+", " ", "\n".join(src.values()))
    assert "like every other comparison in this paper" not in flat, (
        "the paper tells the reader it corrects nothing, in a paper that corrects "
        "everything it reports")
    occurrences = re.findall(r"[^.]*carr(?:y|ies) no multiple-comparison[^.]*\.", flat)
    assert len(occurrences) == 1, (
        f"expected exactly the sweep-and-datasets caveat, found: {occurrences}")
    assert "gate thresholds in the sweep" in occurrences[0], (
        "the only surviving 'no correction' sentence must be the one about the families "
        f"that genuinely carry none: {occurrences[0]!r}")
    # The caution count in that paragraph has to match the cautions in it.
    para = next(p for p in re.split(r"\n\s*\n", src["sections/close.tex"])
                if "cautions attach to the gap" in p)
    assert "Four cautions" in para, "the paragraph states its own count"
    assert "Holm-Bonferroni adjustment\nprinted above" in para, (
        "caution two must state the correction the appendix already printed")


def test_section_twelve_scopes_its_holm_count_to_the_configurations_it_compares():
    """MultiplicitySurvivingN counts only the comparisons between reported
    configurations, and Section 12 said '1 survives ... and nothing else meets it either
    way' with no such scope, two sentences before an appendix reporting the two MiniCheck
    pairs as significant both raw and adjusted."""
    d = json.loads(NUMBERS_JSON.read_text())
    mp = d["multiplicity"]
    survivors = set(mp["survives_at_05"])
    scoped = set(mp["survives_at_05_excluding_minicheck"])
    assert survivors - scoped, (
        "no family member survives outside the scoped count, so the unscoped sentence "
        "would have been harmless and this test is pinning nothing")
    flat = re.sub(r"\s+", " ", "\n".join(_paper_sources().values()))
    assert "and nothing else meets it either way" not in flat, (
        "the unscoped claim is false: "
        f"{sorted(survivors - scoped)} also survive Holm at 0.05")
    # "these configurations" was not enough. It sat in the same paragraph that calls
    # MiniCheck "a third arm was excluded", so a reader who counts the adjusted p-values
    # below 0.05 in Appendix E finds three against a stated one, and the two the sentence
    # dropped are the two smallest p-values in the paper. Both use sites now name the set
    # the count is over AND say what the other survivors are, at the point of use, so the
    # carve-out travels with the number instead of living in the appendix.
    close = re.sub(r"\s+", " ", _paper_sources()["sections/close.tex"])
    setup = re.sub(r"\s+", " ", _paper_sources()["sections/setup.tex"])
    assert ("no other among the reported configurations meets it either way") in close, (
        "section 12 must name the set its Holm count is over")
    assert ("Both tests against the excluded MiniCheck arm clear the adjustment too, as "
            "evidence for that exclusion.") in close, (
        "section 12 must say, where it states the count, that the two MiniCheck tests also "
        f"clear the adjustment; they do: {sorted(survivors - scoped)}")
    assert ("comparisons among the reported configurations separate") in setup, (
        "section 5 defines the term and must scope the count it states")
    assert ("while the two tests against the excluded MiniCheck arm clear it as evidence "
            "for that exclusion.") in setup, (
        "section 5 must carry the same carve-out at its own point of use")


def test_appendix_e_prints_raw_and_adjusted_p_for_every_family_member():
    """The spec requires raw and corrected p for every test (threat C5). Appendix E
    reported the two MiniCheck pairs as 'both below 0.000 adjusted', a bound no
    probability satisfies, and printed neither raw value: 9 of 11 members, not 11."""
    d = json.loads(NUMBERS_JSON.read_text())
    mp = d["multiplicity"]
    macros = _macros()
    appendix = re.sub(r"\s+", " ", (REPO / "arxiv/source/sections/close.tex").read_text())
    para = next(p for p in appendix.split("\\paragraph{") if "whole family" in p[:80])

    for name in mp["raw"]:
        if f"{name}McNemarPHolm" in macros and "MiniCheck" not in name:
            assert f"\\{name}McNemarPHolm" in para or \
                   f"\\{name}McNemarPMedianHolm" in para, name
    # The two tiny members are printed in scientific form, mantissa and exponent, because
    # three decimals renders them as an impossible bound.
    for name in ("JevMiniCheck", "AstraMiniCheck"):
        for kind in ("", "Holm"):
            for part in ("Mantissa", "Exponent"):
                macro = f"{name}McNemarP{kind}{part}"
                assert macro in macros, f"{macro} is not emitted"
                assert f"\\{macro}" in para, f"{macro} is not printed in Appendix E"
        assert float(macros[f"{name}McNemarPHolm"]) == 0.0, (
            "the three-decimal macro no longer rounds to zero, so the scientific form may "
            "not be needed; revisit deliberately")
    assert "both below \\JevMiniCheckMcNemarPHolm" not in appendix


def test_appendix_c_holds_the_same_set_as_the_released_incident_log():
    """Appendix C ends "These N entries and the project's full incident log are the same
    set". Nothing checked it, and it was false in both directions at once: the appendix
    omitted an entry the log carried, the log omitted two the appendix carried, and neither
    of them mentioned that the cascade operating point the paper retracts is already
    published under the first author's byline on Medium.

    The count in the prose, the number of paragraphs in the appendix and the number of
    entries in the log are now one fact, checked here, so the sentence cannot outlive the
    set it describes.
    """
    close = (REPO / "arxiv/source/sections/close.tex").read_text()
    appendix = close.split("\\section{Incidents and corrections}", 1)[1] \
                    .split("\\section{Reproduction}", 1)[0]
    paragraphs = re.findall(r"\\paragraph\{([^}]*)\}", appendix)
    log = (REPO / "docs/reference/incident-log.md").read_text()
    entries = re.findall(r"^## (.+)$", log, re.M)
    assert len(paragraphs) == len(entries), (
        f"Appendix C carries {len(paragraphs)} entries and the released incident log "
        f"{len(entries)}, while the appendix says they are the same set:\n"
        f"  appendix: {paragraphs}\n  log: {entries}")
    words = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    stated = f"These {words[len(paragraphs)]} entries and the project's full incident log"
    assert re.sub(r"\s+", " ", stated) in re.sub(r"\s+", " ", appendix), (
        f"the appendix must state its own count: expected {stated!r}")
    # The entry this test was written for. The paper retracts an operating point that is
    # already in public circulation, so both documents have to name the article.
    assert any("published" in p.lower() for p in paragraphs), (
        "no Appendix C entry discloses the operating point published before the paper")
    assert "medium/article-v3.md" in appendix.replace("\\_", "_")
    assert "article-v3.md" in log, (
        "the incident log entry must name the published article the appendix names")


def test_the_multiplicity_family_says_which_members_are_medians():
    """Four of the eleven members are medians of one exact McNemar per cross-fitting
    repeat over overlapping resamples of the same claims. A median of p-values has no null
    distribution, so Holm over that family controls no error rate on those members, and the
    abstract leans on one of them. The paper has to say so."""
    d = json.loads(NUMBERS_JSON.read_text())
    mp = d["multiplicity"]
    assert mp["median_members"] == ["CascadeAstra", "CascadeAstraFVR",
                                    "CascadeAstraHigh", "CascadeAstraHighFVR",
                                    "CascadeJev", "CascadeJevFVR"], mp["median_members"]
    # One of them is now a survivor of the correction, which makes the descriptive-only
    # caveat load-bearing rather than housekeeping, so the appendix has to attach it to
    # that member by name.
    assert "CascadeAstraHighFVR" in mp["survives_at_05_excluding_minicheck"]
    macros = _macros()
    assert macros["MultiplicityMedianMembersN"] == str(len(mp["median_members"]))
    flat = re.sub(r"\s+", " ", "\n".join(_paper_sources().values()))
    assert "\\MultiplicityMedianMembersN" in flat, (
        "neither Section 5 nor Appendix E names how many family members are medians")
    assert "medians over cross-fitting repeats rather than single exact tests" in flat, (
        "Section 5 must say what those members are")
    assert "a median of $p$-values" in flat, (
        "Appendix E must say why the adjustment on them is descriptive")
    # And the one result the paper claims is not one of them, with a stated margin.
    assert "FVRGap" not in mp["median_members"]
    raw = mp["fvr_gap_raw_p"]
    assert mp["fvr_gap_holm_break_family_n"] == math.ceil(0.05 / raw)
    assert raw * (mp["fvr_gap_holm_break_family_n"] - 1) < 0.05 <= \
        raw * mp["fvr_gap_holm_break_family_n"], "the break point is off by one"


def test_section_eight_does_not_call_the_vendor_probability_well_calibrated(out):
    """Section 8 opened by calling Jev's probability "reasonably calibrated". Its ECE is
    9.9 points and its NLL is 1.133, and this is the section that motivates the routing
    signal the cascade rests on, written by the founder of the company that sells the
    verifier, so a favourable adjective here costs more than anywhere else in the paper.

    Every fact below comes from numbers.json, so the prose is checked against the
    evidence rather than against a reviewer's memory of it.
    """
    cal = out["configurations"]["jev"]["calibration"]
    clip = cal["logloss_clip"]
    macros = _macros()
    para = re.sub(r"\s+", " ", (REPO / "arxiv/source/sections/cascade.tex").read_text())

    assert "reasonably calibrated" not in para, (
        "the vendor probability is called well calibrated beside an ECE of "
        f"{cal['ece']:.3f} and an NLL of {cal['negative_log_loss']:.3f}")
    # Section 8 described the miscalibration as sitting "at the confident ends of the
    # scale" for three drafts, and this test pinned that phrase without ever looking at the
    # bins. They say the opposite. The location is now derived from the reliability array
    # and checked against the prose, so a future rewrite has to keep matching the data.
    rel = cal["reliability"]
    gaps = [(b["mean_p"], b["observed_supported_rate"] - b["mean_p"]) for b in rel]
    worst = max(gaps, key=lambda g: abs(g[1]))
    worst_p = worst[0]
    assert 0.0 < worst_p < 0.5, (
        f"the widest calibration gap is at predicted {worst_p:.3f}; if it moves to an "
        "extreme the prose in section 8 has to move with it")
    # The replacement claim was "the confident extremes are the best-calibrated bins", and
    # the guard that was supposed to hold it honest defined the confident set as
    # `g[0] > 0.97 or 0.0 < g[0] < 0.01`, which excludes by construction the two bins that
    # refute it: the strict `0.0 <` drops the p = 0.00 bin, which is the MOST confident bin
    # in the table, and the 0.01/0.97 cutoffs drop the widest gap of all at p = 0.05. So the
    # confidence is computed here, from the definition section 3 routes by, over every bin.
    def conf(p):
        return max(p, 1.0 - p)
    by_conf = sorted(gaps, key=lambda g: -conf(g[0]))
    best = min(gaps, key=lambda g: abs(g[1]))
    assert conf(worst_p) >= conf(best[0]), (
        "the widest gap is now at a less confident point than the flattest bin, which is "
        "the comfortable arrangement section 8 used to assert without checking; the prose "
        f"has to be rewritten: worst at c={conf(worst_p):.3f}, best at c={conf(best[0]):.3f}")
    assert conf(by_conf[0][0]) > conf(best[0]), "the best-calibrated bin is the most confident one"
    assert "confident extremes are the" not in para and "best-calibrated bins, not the worst" not in para, (
        "section 8 is claiming the confident bins are the best calibrated, and on "
        f"max(p, 1-p) the widest gap of all sits at a confidence of {conf(worst_p):.3f}")
    # And the claim it makes instead has to be the one the bins support, priced in the same
    # confidence, with the flattest bin named as the least confident rather than the most.
    for macro in ("JevCalWorstConfidence", "JevCalBestP", "JevCalFirstAtOrBelowP",
                  "JevCalBinsBelowDiagonalN", "JevCalInteriorHi"):
        assert f"\\{macro}" in para, (
            f"\\{macro} is not printed in section 8, so the reader is given a shape "
            "claim without the quantity that decides it")
    assert macros["JevCalWorstConfidence"] == f"{conf(worst_p):.2f}"
    assert macros["JevCalBestP"] == f"{best[0]:.2f}"
    assert macros["JevCalBinsBelowDiagonalN"] == str(sum(1 for g in gaps if g[1] < 0.0))
    assert macros["JevCalFirstAtOrBelowP"] == \
        f"{min(g[0] for g in gaps if g[1] <= 0.0):.2f}"
    # "It crosses below only between 0.92 and 0.98" was false: the curve is under the
    # diagonal at six of the ten bins, from 0.64 up. The word that made it false was
    # "only", so no sentence in the section may scope the crossing with it again.
    assert "crosses below only" not in para, (
        f"the curve is below the diagonal at {sum(1 for g in gaps if g[1] < 0.0)} bins, "
        "so the crossing cannot be scoped to the over-confident band")
    # The share of the error called "interior" is a fact about predicted probability, and
    # on this definition a predicted 0.05 is interior AND confident. Printing the share
    # without the bound is what let the two readings be conflated.
    assert "\\JevCalInteriorShareOfECEPct" not in para or "\\JevCalInteriorHi" in para, (
        "the interior share is printed without the bound that defines interior")
    # Derived rather than pinned: the lower half of the scale is under-confident in every
    # bin, which is what the opening sentence says.
    lower = [g for g in gaps if g[0] < 0.5]
    assert lower and all(g[1] > 0 for g in lower), (
        "at least one bin below a predicted 0.5 is now over-confident, so the opening "
        "sentence of section 8.1 has to stop saying the lower half is under-confident")
    assert "under-confident across the lower half of the scale" in para, (
        "section 8 must give the description its own reliability bins give")
    # The boundary itself, which the previous guard did not check. \JevCalUnderConfidentBelow
    # was the HIGHEST under-confident bin, so the region it bounds includes it, and the
    # prose said "below" and then offered that same bin as its second example nine words
    # later. The macro now carries the boundary in its name and the prose says "at and
    # below"; this derives both halves of that claim from the bins.
    # The stored boundary, not the printed one: the macro rounds 0.2904 to two decimals and
    # a comparison against the rounded value puts the boundary bin itself on the wrong side.
    boundary = cal["shape"]["under_confident_at_or_below"]
    assert macros["JevCalUnderConfidentAtOrBelow"] == f"{boundary:.2f}"
    at_or_below = [g for g in gaps if g[0] <= boundary + 1e-9]
    above = [g for g in gaps if g[0] > boundary + 1e-9]
    assert at_or_below and above
    assert all(g[1] > 0 for g in at_or_below), (
        "a bin at or below the stated boundary is not under-confident, so the scope in "
        f"section 8.1 is wrong: {[(round(p, 4), round(gap, 4)) for p, gap in at_or_below]}")
    first_above = min(above, key=lambda g: g[0])
    assert first_above[1] <= 0.02, (
        f"the bin above the boundary, at a predicted {first_above[0]:.4f}, clears the "
        "diagonal by more than the tolerance, so the boundary is in the wrong place")
    assert "above the diagonal at and below" in para, (
        "section 8.1 must include the boundary bin it bounds the region with, since that "
        "bin is one of the two examples the same sentence gives")
    assert "above the diagonal below" not in para, (
        "the preposition that excluded the boundary bin is back")
    assert "confident ends of the scale" not in para, (
        "the retracted description must not come back")

    # One quantity, one value. \JevCalTiedAtZeroN and \JevProbAtZeroN are both "the
    # predictions the vendor reports as exactly 0.00", and they were 50 and 86: the first
    # was derived from bin means, which can only ever find the lowest bin, so section 8
    # told a reader that 50 predictions are reported as zero while appendix E told the
    # same reader 86, and the sentence built on the first described binning cutting a set
    # of 50 into a bin of 50.
    res = cal["probability_resolution"]
    shape = cal["shape"]
    assert shape["n_tied_at_zero"] == res["n_at_zero"], (
        "the shape block and the resolution block disagree about how many predictions are "
        f"reported as exactly zero: {shape['n_tied_at_zero']} against {res['n_at_zero']}")
    assert macros["JevCalTiedAtZeroN"] == macros["JevProbAtZeroN"] == str(res["n_at_zero"])
    assert shape["n_tied_at_zero_in_lowest_bin"] == \
        sum(b["n"] for b in rel if b["lo"] == 0.0 and b["hi"] == 0.0)
    assert shape["n_tied_at_zero_in_lowest_bin"] < shape["n_tied_at_zero"], (
        "the lowest bin now holds every zero, so section 8 cannot describe the binning as "
        "cutting them")
    assert "\\JevProbAtZeroN" in para and "\\JevCalZeroInLowestBinN" in para, (
        "section 8 describes equal-count binning cutting the zeros and has to print both "
        "the number of zeros and the number of them the lowest bin holds")

    # And the clip's share travels with the number it decides.
    assert clip["share"] > 0.5, (
        "the clip no longer carries most of the NLL, so the sentence that says it does "
        f"must be rewritten rather than left: share is {clip['share']}")
    assert macros["JevNLLClipSharePct"] == f"{clip['share'] * 100:.0f}"
    assert macros["JevNLLClipN"] == str(clip["n"])
    for macro in ("JevNLLClipSharePct", "JevNLLClipN"):
        assert f"\\{macro}" in para, (
            f"\\{macro} is not printed beside the NLL, so a reader is told a figure that "
            "is mostly a clipping constant and not told that it is")
    # The prose says "called certain and got wrong", which is both directions of the
    # clip. It is not "supported claims given zero": one of these items is the mirror.
    assert clip["n"] == clip["n_supported_at_zero"] + clip["n_unsupported_at_one"]
    if clip["n_unsupported_at_one"]:
        assert "claims Jev called certain and got wrong" in para, (
            f"{clip['n_unsupported_at_one']} of the clipped items are unsupported claims "
            "given a probability of one, so the prose may not describe the set as "
            "supported claims given a probability of zero")


def test_the_log_loss_verdict_is_scoped_to_the_constant_that_decides_it(out):
    """Section 8 reported the NLL as "worse than a constant one-half forecast scores" in
    the same sentence that conceded 62% of it is ten items charged at a clipping constant
    of ours. Both halves were true and the comparison was not a fact about the model: the
    constant is marked PROVISIONAL in configs/benchmark.yaml, no gate ever fixed it, and
    moving it to anything above about 3e-6 reverses the verdict, while the vendor reports
    the probability to two decimal places. A reviewer can run that arithmetic from the
    released numbers.json in a minute, so the paper has to have run it first.

    This test is the old one turned around: it used to require the reference to be named,
    and now requires that naming it is never done without the constant and the crossing.
    """
    cal = out["configurations"]["jev"]["calibration"]
    sens = cal["logloss_eps_sensitivity"]
    res = cal["probability_resolution"]
    macros = _macros()
    src = _paper_sources()

    # 1. The arithmetic that makes the scope necessary rather than decorative.
    assert sens["reference_nll"] == pytest.approx(math.log(2)), (
        "the reference is a constant one-half forecast on these labels")
    assert sens["reported_exceeds_reference"], (
        "at the configured constant the reported NLL no longer exceeds the reference, so "
        "every sentence this test pins is stale and must be rewritten deliberately")
    assert sens["crossover_eps"] is not None, (
        "no constant on the grid reverses the comparison, which is the premise of the "
        "scope the paper now carries")
    assert sens["crossover_eps"] < res["half_step"], (
        "the comparison reverses only below the vendor's own reporting resolution "
        f"({sens['crossover_eps']} against {res['half_step']}), which is the fact the "
        "paper states; if this ever flips, the prose is wrong")
    by_eps = {row["eps"]: row["nll"] for row in sens["grid"]}
    assert by_eps[res["half_step"]] < sens["reference_nll"], (
        "at a constant the size of the reporting resolution the model beats the reference")

    # 2. No unqualified comparison against that reference survives anywhere.
    flat = re.sub(r"\s+", " ", "\n".join(src.values()))
    assert "worse than a constant one-half forecast" not in flat, (
        "the comparison is stated as a property of the model; at a constant of "
        f"{res['half_step']} the same predictions score {by_eps[res['half_step']]:.3f} "
        f"against the reference {sens['reference_nll']:.3f}")

    # 3. Section 8 states the resolution, says the constant is ours, and re-scores.
    cascade = re.sub(r"\s+", " ", src["sections/cascade.tex"])
    assert "\\JevProbDecimals" in cascade, (
        "section 8 calls items certain without saying that certain is a reported "
        f"{res['decimals']}-decimal value")
    assert "provisional" in cascade, (
        "the constant carrying most of the reported NLL is marked provisional in the "
        "config the paper cites, and section 8 must say so where it cites it")
    assert "\\JevNLLAtHalfStepClip" in cascade and "\\JevProbHalfStep" in cascade, (
        "section 8 must print the figure at a constant the size of the reporting "
        "resolution, which is the sensitivity a reader would otherwise compute")

    # 4. Where the reference IS named, in Appendix E, the crossing is named with it.
    close = re.sub(r"\s+", " ", src["sections/close.tex"])
    reference_sentences = [t for t in re.findall(r"[^.]*constant\s+one-half forecast[^.]*\.",
                                                 close)]
    assert reference_sentences, (
        "Appendix E is where the reference is priced, and it no longer names it")
    for sentence in reference_sentences:
        assert "\\JevNLLCrossoverEps" in sentence or "\\JevNLLConstantHalf" in sentence, (
            f"the reference is named without what decides the comparison: {sentence!r}")
    assert "\\JevNLLCrossoverEpsMantissa" in close, (
        "Appendix E must print the constant at which the comparison reverses")
    for name in ("JevNLLCrossoverEpsMantissa", "JevNLLCrossoverEpsExponent",
                 "JevNLLAtHalfStepClip", "JevNLLConstantHalf", "JevProbDecimals",
                 "JevProbHalfStep"):
        assert name in macros, f"the paper needs \\{name} to state this honestly"


def test_the_headline_gap_is_printed_beside_the_movement_of_one_verifier(out):
    """The same verifier, run twice over the identical 495 claims, moves macro-BAcc by
    more than the headline difference the paper is built on. That figure appeared once,
    in Appendix E, on page 30. A reader who takes the headline as a comparison of two
    systems has to be told, where the headline is stated, that one of the two moves
    further than the gap between them."""
    rerun = abs(out["jev_rerun"]["macro_bacc_delta_pp"])
    headline = abs(out["pairs"]["jev|astra"]["macro_bacc_delta_pp"])
    assert rerun > headline, (
        "the rerun no longer moves further than the headline gap, so the clause this "
        f"test pins overstates it: {rerun} against {headline}")
    src = _paper_sources()
    for name in ("sections/same-score.tex", "sections/close.tex"):
        assert "\\JevRerunDeltaPP" in src[name], (
            f"{name} states the headline difference and not the movement of one of the "
            "two systems between two runs of itself")


def test_the_rerun_movement_is_stated_in_the_direction_it_actually_moved(out):
    """Appendix E printed a RISE of 0.83 points as "-0.8 points, which reverses this
    paper's headline ordering against the frontier arm's 73.83", and both of Jev's passes
    are below 73.83, so nothing reversed. Section 6 of the same paper called the same
    comparison "a deficit, not a tie". The macro was defined earlier minus reported while
    all three sentences that print it read reported minus earlier, and the one test that
    touched the value wrapped it in abs() and never looked at the sentence.

    The macro is unsigned now and the direction is a word, so this checks both halves:
    that no sign can contradict a sentence, and that no sentence claims an ordering that
    the two levels beside it refute.
    """
    rerun = out["jev_rerun"]["macro_bacc"] * 100
    headline = out["configurations"]["jev"]["macro_bacc"] * 100
    astra = out["configurations"]["astra"]["macro_bacc"] * 100
    macros = _macros()

    assert not macros["JevRerunDeltaPP"].startswith("-"), (
        "the movement is printed with a sign again, and every sentence that carries it "
        "states the direction in words, so the sign can only disagree with one of them")
    assert macros["JevRerunDeltaPP"] == f"{abs(headline - rerun):.1f}"
    assert macros["JevRerunBAccTwoDP"] == f"{rerun:.2f}"
    assert macros["JevBAccTwoDP"] == f"{headline:.2f}"

    src = _paper_sources()
    flat = re.sub(r"\s+", " ", "\n".join(src.values()))
    assert "reverses this paper's headline ordering" not in flat, (
        "the retracted claim is back; on these data the frontier arm leads on both of "
        f"Jev's passes ({rerun:.2f} and {headline:.2f} against {astra:.2f})")

    if rerun < astra and headline < astra:
        # Nothing about the run-to-run movement can be described as changing which system
        # is ahead, because neither pass is ahead. What it does do is exceed the gap.
        assert "widens the deficit rather than reversing anything" in \
            re.sub(r"\s+", " ", src["sections/close.tex"]), (
            "Appendix E has to say what the movement does to an ordering in which the "
            "frontier arm leads on both passes")
        assert abs(headline - rerun) > abs(out["pairs"]["jev|astra"]["macro_bacc_delta_pp"])
    else:
        raise AssertionError(
            "one of Jev's two passes now leads the frontier arm, so Appendix E and "
            f"Section 6 both have to be rewritten: {rerun:.2f}, {headline:.2f}, {astra:.2f}")

    # Section 6 and Appendix E describe the same two numbers and used to disagree about
    # which way they run. Both say the later pass is the higher one now.
    assert "rose from" in re.sub(r"\s+", " ", src["sections/close.tex"])
    assert "points above it and still short of" in \
        re.sub(r"\s+", " ", src["sections/same-score.tex"])
    assert "a deficit, not a tie" in re.sub(r"\s+", " ", src["sections/same-score.tex"])


def test_section_seven_does_not_present_a_one_claim_specificity_as_a_property(out):
    """Section 7.4 said both systems' RAGTruth specificities were "pinned at the top of
    the scale", which the reported scores do force. The paper also releases an earlier
    pass of the same verifier over the same four gold-unsupported claims that
    false-verifies one of them and takes the dataset down by 13.7 points, so what the
    scores force is a fact about this run and not about either system."""
    rerun = out["jev_rerun"]
    fv = rerun["per_dataset_false_verified"]["RAGTruth"]
    assert fv > 0, (
        "the earlier pass no longer false-verifies a RAGTruth claim, so the disclosure "
        "this test pins is stale and section 7.4 should be rewritten deliberately")
    assert rerun["per_dataset"]["RAGTruth"] < out["configurations"]["jev"]["per_dataset"]["RAGTruth"], (
        "the earlier pass no longer scores below the reported run on this dataset")
    macros = _macros()
    assert macros["JevRerunFalseVerifiedRAGTruth"] == str(fv)
    text = re.sub(r"\s+", " ",
                  (REPO / "arxiv/source/sections/different-errors.tex").read_text())
    assert "both specificities are pinned at the top of the scale" not in text, (
        "the paper presents as forced a specificity its own released rerun breaks")
    for name in ("JevRerunFalseVerifiedRAGTruth", "JevRerunBAccRAGTruth"):
        assert f"\\{name}" in text, (
            f"\\{name} is not printed where the dataset's specificity is read, so the "
            "counterexample stays in the appendix while the claim stays in the body")


def test_appendix_e_carries_both_clauses_of_the_ablation_item_it_expands():
    """Section 4 concedes that the reasoning-effort ablation cannot answer the objection
    that the frontier arm was told not to think, because both efforts ran under one prompt
    that forbids reasoning in the response, byte for byte. Appendix E reported the same
    ablation as showing that "the ablation moved nothing on this task", named neither the
    instruction nor the provider-default run the protocol asks for (errata E21, threat
    B1), and is by SECTION_TWELVE_ITEMS the accounting for the section 12 item that
    carries both clauses. The retraction sat in the section a reader skims and the
    overclaim in the section a reviewer reads, and the overclaim survived a repair round
    in which the executor reports having swept for it. So it is pinned.

    The claim of identity is evidence, not rhetoric: both frontier runs are read here and
    must carry the same prompt hash.
    """
    close = re.sub(r"\s+", " ", (REPO / "arxiv/source/sections/close.tex").read_text())
    para = next(p for p in close.split("\\paragraph{")
                if p.startswith("One frontier family"))
    assert "moved nothing on this task" not in para, (
        "the paragraph draws exactly the inference section 4 says these data cannot "
        "support")
    assert "forbidding reasoning in the response" in para, (
        "the paragraph must say what both efforts ran under, or the contrast reads as a "
        "test of reasoning rather than of the effort knob")
    assert "provider-default configuration" in para and "never run" in para, (
        "the paragraph must say the configuration the protocol asks for was not run; "
        "section 12's item does, and this paragraph is its accounting")
    flat = re.sub(r"\s+", " ", "\n".join(_paper_sources().values()))
    assert "the ablation moved nothing" not in flat, (
        "no section may state the conclusion section 4 retracts")

    # The prompts really are identical, which is what makes the retraction necessary.
    hashes = set()
    for run in ("runs/paired-astra-lowthink-20260921-p2b/predictions_gpt6_astra.jsonl",
                "runs/paired-astra-highthink-20260921-p2b/predictions_gpt6_astra.jsonl"):
        path = REPO / run
        if not path.is_file():
            pytest.skip(f"{run} is not on disk")
        hashes |= {json.loads(line)["prompt_hash"] for line in path.read_text().splitlines()
                   if line.strip()}
    assert len(hashes) == 1, (
        "the two frontier efforts no longer share one prompt, so the paragraph's "
        f"'byte for byte' is stale: {sorted(hashes)}")


def test_the_printed_p_values_reproduce_the_printed_holm_adjustments():
    """Holm arithmetic a reader can do with a pencil, on the digits the paper prints.

    The paper's second result is that the cascade's guard penalty does NOT survive the
    correction. Appendix E gave the raw median as 0.006, said in the same paragraph that
    Holm multiplies the member by the tests at or above it in the step-down, and printed
    0.051 adjusted. 0.006 x 8 = 0.048, which clears the 0.05 the paper says it misses, so
    the one arithmetic in this paper that can flip a claim could not be done from the
    paper. The value was right and the printed precision was not.

    Every family member whose raw p the paper prints as a plain decimal therefore has to
    reproduce its printed adjusted p, at that adjusted value's own precision, from the
    printed digits and the multiplier Holm gives it. The macro gate exists because a gate
    that checks arithmetic the paper does not print is worth nothing; this is the same
    rule applied to the correction.
    """
    mp = json.loads(NUMBERS_JSON.read_text())["multiplicity"]
    macros = _macros()
    raw = mp["raw"]
    checked = []
    for name, value in sorted(raw.items(), key=lambda kv: kv[1]):
        if "MiniCheck" in name:
            continue  # printed as mantissa and exponent; covered by the test above
        printed_raw = macros.get(f"{name}McNemarPMedian") or macros.get(f"{name}McNemarP")
        printed_adj = macros.get(f"{name}McNemarPHolm")
        if printed_raw is None or printed_adj is None:
            continue
        # Holm multiplies a member by the number of tests whose raw p is at least its
        # own, and caps the result at 1. That is the arithmetic the appendix describes
        # in prose, so it is the arithmetic a reader will try.
        multiplier = len([v for v in raw.values() if v >= value])
        dp = len(printed_adj.partition(".")[2])
        got = min(1.0, float(printed_raw) * multiplier)
        assert f"{got:.{dp}f}" == printed_adj, (
            f"\\{name}McNemarPHolm prints {printed_adj}, but a reader multiplying the "
            f"printed raw {printed_raw} by its Holm multiplier of {multiplier} gets "
            f"{got:.{dp}f}. Print the raw p at the precision its own adjustment needs "
            "rather than at a precision that inverts it.")
        checked.append(name)
    assert "CascadeAstraFVR" in checked, (
        "the guard penalty is the member whose adjusted p decides the paper's second "
        "result, so it is the one this test exists for")
    assert len(checked) >= 5, f"only {checked} were checked; the guard has gone stale"


def test_the_family_boundary_holds_for_the_digits_the_paper_prints():
    """Appendix E tells a reader the false-verification gap's adjusted p 'stays below
    0.05 for any family smaller than \\FVRGapHolmBreakFamilyN tests'. Holm's largest
    possible multiplier is the family size, so that sentence is an arithmetic claim about
    the printed raw p and not only about the stored one. At five decimals the printed
    0.00053 x 94 gave 0.04982, which falsifies the sentence, while the value behind it
    (0.0005335) does not."""
    macros = _macros()
    p = float(macros["FVRGapMcNemarP"])
    n = int(macros["FVRGapHolmBreakFamilyN"])
    assert p * (n - 1) < 0.05, (
        f"the printed {p} carried over a family of {n - 1} gives {p * (n - 1)}, so the "
        "sentence claims a family the printed digits do not support")
    assert p * n >= 0.05, (
        f"the printed {p} carried over a family of {n} gives {p * n}, which is still "
        "below 0.05, so the stated boundary is not where the printed digits put it")


def test_the_family_sensitivity_the_paper_reports_is_the_one_that_can_change_the_answer():
    """The paper told the reader the family size "decides an answer", that it fixed the
    family itself in code, and then ran one robustness check on it: dropping the two
    MiniCheck members, which are smaller than every other member and therefore move only
    the step-down positions. Arithmetic guarantees that check cannot change anything.

    The check that can is the other direction. Holm multiplies this member by the number of
    tests whose raw p is at least its own, so dropping any one of those takes the cascade's
    guard penalty from just above 0.05 to just below it. Two paragraphs later the paper
    supplies exactly this bound for the result that survived, which is what made the
    omission conspicuous. Both bounds are now computed in holm_bonferroni() and printed.
    """
    d = json.loads(NUMBERS_JSON.read_text())
    mp = d["multiplicity"]
    macros = _macros()
    close = re.sub(r"\s+", " ", (REPO / "arxiv/source/sections/close.tex").read_text())

    # The sensitivity the paper used to report on its own is genuinely inert.
    assert mp["cascade_guard_holm_p_excluding_minicheck"] > 0.05, (
        "dropping the MiniCheck members now changes the answer, so the sentence that says "
        "it cannot is stale")
    # The one it omitted is not. How many members have to go changed when the family grew
    # by the two comparisons against the high-effort arm: at eleven members dropping any
    # one of the larger tests cleared it, at thirteen it takes two. The paragraph prints
    # the count, so the count is checked against the data rather than against the word
    # that used to be in the sentence.
    dropped = mp["cascade_guard_holm_break_dropped"]
    assert macros["CascadeGuardHolmBreakDroppedN"] == str(len(dropped)) == "2", dropped
    assert mp["cascade_guard_holm_clears_dropping_any_one_above"] is False, (
        "dropping a single member now clears the guard penalty, so the appendix sentence "
        "saying it takes two is stale")
    assert mp["cascade_guard_holm_p_at_that_family"] < 0.05 <= \
        float(macros["CascadeAstraFVRMcNemarPHolm"]), (
        "the guard penalty no longer sits on the boundary, so this whole paragraph needs "
        "rewriting rather than patching")
    assert mp["cascade_guard_holm_clears_at_family_n"] == mp["family_size"] - len(dropped)
    assert mp["cascade_guard_holm_members_above_n"] == len(
        [k for k, v in mp["raw"].items()
         if k != "CascadeAstraFVR" and v >= mp["raw"]["CascadeAstraFVR"]])

    # And the paper states it, in the paragraph that carries the inert one.
    para = next(p for p in close.split("\\paragraph{") if "clear both instruments" in p[:90])
    for macro in ("CascadeGuardHolmPExcludingMiniCheck", "CascadeGuardHolmMembersAboveN",
                  "CascadeGuardHolmClearsAtFamilyN", "CascadeGuardHolmPAtThatFamily",
                  "CascadeGuardHolmBreakDroppedN"):
        assert macro in macros, f"{macro} is not emitted by the analysis"
        assert f"\\{macro}" in para, (
            f"Appendix E states a family sensitivity without \\{macro}, so the reader gets "
            "the perturbation that cannot move the answer and not the one that can")
    assert "turns on where the family boundary falls" in para, (
        "the paragraph must say what the bound means, not only print it")


def test_no_lead_is_stated_as_a_negative_number():
    """The paragraph added to disclose the prompt-parity sign reversal read 'the frontier
    model leads by -0.56 points'. The convention is Jev minus Astra, so a frontier lead is
    a negative number under it and cannot be phrased as a lead."""
    flat = re.sub(r"\s+", " ", "\n".join(_paper_sources().values()))
    signed = {name for name, val in _macros().items() if val.strip().startswith("-")}
    for m in re.finditer(r"leads by \$?\{?\\([A-Za-z]+)\}?\$?", flat):
        assert m.group(1) not in signed, (
            f"\\{m.group(1)} is negative and the prose calls it a lead: {m.group(0)!r}")
    assert "the ordering reverses, to ${\\JevAstraDeltaSignedTwoDP}$ points on Jev minus" \
        in flat, "the disclosure must state the reversal as a reversal"


def test_section_seven_divides_the_cancelled_residue_by_the_dataset_count():
    """'the 0.6-point headline is what survives after 75.5 points of per-dataset
    disagreement cancel against each other' set a mean against a sum: what survives the
    cancellation is the signed residue, 6.1, and the headline is that over eleven
    datasets. A reader with a pencil got a different answer from the one the sentence
    asserted, and Appendix A prints both halves of the residue, so the falsification was
    inside the document."""
    d = json.loads(NUMBERS_JSON.read_text())
    pc = d["complementarity"]["per_dataset_concentration"]
    macros = _macros()
    signed = float(macros["SignedSumDeltasPP"])
    assert signed == round(pc["sum_of_deltas_pp"], 2)
    assert abs(signed / pc["n_datasets"] - pc["macro_delta_pp"]) < 5e-3, (
        "the residue over the dataset count must be the macro difference")
    assert abs(signed) < pc["sum_abs_deltas_pp"] / 2, (
        "the signed residue is no longer much smaller than the absolute sum, so the "
        "sentence about cancellation needs rewriting")
    body = re.sub(r"\s+", " ",
                  (REPO / "arxiv/source/sections/different-errors.tex").read_text())
    assert "cancel against each other" not in body, (
        "the sentence that omitted the division survives")
    assert "cancel down to a signed residue of ${\\SignedSumDeltasPP}$ points" in body
    assert "residue spread across the \\NDatasets\\ datasets" in body


def test_the_reasoning_partition_reconciles_with_the_section_4_totals():
    """Appendix E partitions the reported low-effort frontier run into the rows the
    same-prompt repeat covers and the rows it excludes. Those two parts must add up to the
    totals Section 4 prints for the same configuration.

    They did not: two of the three counts in that sentence were computed on the superseded
    pre-parity run and the third on the reported run, so 397 + 4 came to 401 where Section 4
    said 407. Both partitions are now emitted with the file in the key, and this holds the
    arithmetic that makes them the same configuration.
    """
    d = json.loads(NUMBERS_JSON.read_text())
    sp = d["same_prompt_repeat"]
    astra = d["configurations"]["astra"]["reasoning_tokens"]

    covered, excluded = sp["n_compared"], sp["n_capped_no_answer"]
    assert covered + excluded == sp["n_paired"] == astra["n"]

    zero_covered = sp["compared_reasoning_zero_n_reported"]
    nonzero_excluded = sp["excluded_reasoning_nonzero_n_reported"]
    zero_excluded = excluded - nonzero_excluded
    assert zero_covered + zero_excluded == astra["n_zero"], (
        f"{zero_covered} zero-reasoning rows among the {covered} compared plus "
        f"{zero_excluded} among the {excluded} excluded must equal Section 4's "
        f"{astra['n_zero']}; if this fails the appendix is quoting a different run")

    nonzero_covered = covered - zero_covered
    assert nonzero_covered + nonzero_excluded == astra["n"] - astra["n_zero"]

    # And the repeat's own partition is kept, separately, so nobody splices them again.
    for key in ("compared_reasoning_zero_n_repeat", "compared_reasoning_mean_repeat",
                "excluded_reasoning_nonzero_n_repeat"):
        assert key in sp, f"{key} must stay available and separately named"
    assert sp["compared_reasoning_zero_n_repeat"] != zero_covered, (
        "the two runs partition differently, which is why the keys carry the file name")


def test_the_separated_count_is_a_macro_everywhere_and_never_a_word():
    """The paper states how many comparisons clear both instruments in five places. Twice
    it has printed two different values in the same appendix, because the count was an
    English word: a repair round updated three of the five and left two behind, and
    check_paper_macros.py exempts spelled numbers below four so nothing caught it.

    The count is now \\NClearingUncorrected, emitted from the same per-estimand rule
    test_exactly_three_comparisons... applies. This bans the word form beside the phrases
    that carry it, so the next edit cannot reintroduce the drift.
    """
    src = _paper_sources()
    flat = re.sub(r"\s+", " ", "\n".join(src.values()))
    macros = _macros()
    d = json.loads(NUMBERS_JSON.read_text())

    assert macros["NClearingUncorrected"] == str(d["multiplicity"]["n_clearing_uncorrected"])

    # "one of the N results" is legitimate: it picks a member, it does not state the count.
    words = ("two|three|four|five|six|seven|eight|nine|ten")
    phrases = (r"results? (?:here )?that clear both instruments",
               r"comparisons? meet (?:it|the separation criterion)",
               r"differences? that clear both instruments")
    for phrase in phrases:
        for m in re.finditer(rf"\b(?:{words})\b[^.]{{0,40}}?{phrase}", flat, re.I):
            raise AssertionError(
                f"the separated count is spelled as a word: {m.group(0)!r}. Use "
                "\\NClearingUncorrected so the five places cannot disagree")
        # and every occurrence of the phrase must carry the macro. This block used to sit
        # after a `raise` in the loop below, so it never ran: the guard written for this
        # exact drift was dead. It is inside the phrase loop now, where `phrase` is the one
        # being scanned rather than whichever value the earlier loop happened to leave.
        for m in re.finditer(rf"[^.]{{0,60}}{phrase}", flat):
            seg = m.group(0)
            assert "NClearingUncorrected" in seg or "both instruments uncorrected but" in seg, (
                f"a count phrase carries no macro: {seg.strip()!r}")
    # The same count, restated as the denominator of the Holm survivors. This escaped the
    # loop above because the word follows the phrase instead of preceding it: the appendix
    # read "2 of the three survive", which goes stale the moment the count moves.
    for m in re.finditer(rf"of the (?:{words})\b[^.]{{0,40}}?surviv", flat, re.I):
        raise AssertionError(
            f"the separated count is spelled as a word: {m.group(0)!r}. Say \"of them\" "
            "or print \\NClearingUncorrected")


def test_the_cascade_is_priced_against_the_cheap_tier_wherever_it_is_recommended():
    """The cascade was framed against the expensive frontier arm in the abstract, in
    Section 11.2 and in the conclusion, and against the cheap tier it is built from only
    in Section 9.4, which opens by conceding that the frontier comparison is "the
    comparison that flatters it" and then reports that the cascade costs
    \\CostRatioCascadeOverJev-fold what Jev alone costs for an accuracy difference whose
    interval contains zero. A reader of the abstract alone, which on arXiv is most
    readers, came away recommending an architecture Section 9.4 declines to recommend,
    and Section 11.2 was headed "Route rather than replace" with nothing in it to say
    which arm routing beats.

    So the three most-read places must each carry the non-flattering comparison: the cost
    ratio against Jev alone and the interval on that accuracy difference. Both are
    already macros, so this costs the paper nothing but the sentence."""
    src = _paper_sources()
    main = src["main.tex"]
    close = src["sections/close.tex"]

    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S)
    assert abstract, "no abstract to check"
    discussion = re.search(r"\\section\{Discussion\}(.*?)\\section\{", close, re.S)
    assert discussion, "Section 11 is no longer where this test looks for it"
    conclusion = re.search(r"\\section\{Conclusion\}(.*?)\\appendix", close, re.S)
    assert conclusion, "Section 13 is no longer where this test looks for it"

    for where, text in (("the abstract", abstract.group(1)),
                        ("Section 11 (Discussion)", discussion.group(1)),
                        ("Section 13 (Conclusion)", conclusion.group(1))):
        flat = re.sub(r"\s+", " ", text)
        for macro in ("CostRatioCascadeOverJev", "CascadeJevDeltaCILo",
                      "CascadeJevDeltaCIHi"):
            assert re.search(r"\\" + macro + r"(?![A-Za-z])", flat), (
                f"{where} recommends the cascade without printing \\{macro}, so it makes "
                "only the comparison Section 9.4 calls the one that flatters it")

    # And the two places that give advice say which arm routing beats, rather than
    # leaving a heading that reads as advice to prefer it over anything.
    close_flat = re.sub(r"\s+", " ", close)
    assert "\\subsection{Route rather than replace}" not in close_flat, (
        "the Section 11.2 heading recommends routing without naming the arm it beats")
    assert "over Jev alone it is not justified on this sample" in close_flat, (
        "Section 11.2 must state that routing is not justified over replacing here")
    assert "it replaces the frontier model, not the cheap tier" in close_flat, (
        "Section 13 must say which of the two arms the cascade replaces")

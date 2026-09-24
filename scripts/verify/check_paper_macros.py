#!/usr/bin/env python3
"""Structural check on the arXiv sources, for environments with no TeX installed.

    docker compose run --rm dev uv run python scripts/verify/check_paper_macros.py

Two failures this catches without a LaTeX run, both of which a reader would meet as a
broken PDF rather than as a wrong number:

  1. A generated macro used in the prose that numbers.tex does not define. Every reported
     value in the paper is a macro (CLAUDE.md rule 4), so a renamed or dropped macro is
     an "Undefined control sequence" at build time and a silently missing number if the
     build is run with \\nonstopmode.
  2. A tabular row whose cell count does not match its column specification, which is
     what an edited table looks like when a column was added to the header and not to a
     row.

  3. A reported quantity spelled out as an English word next to its unit, such as
     "eleven datasets" or "roughly eight points wide". CLAUDE.md rule 4 says every
     reported value is a macro, and a checker that looks only for digits is trivially
     evaded by writing the digit as a word. Three rounds of review found six of these,
     twice on the line after the macro for the same quantity, so the word form is now
     checked rather than trusted.

It also refuses the em dash in any of its three spellings, which is a hard gate for this
paper, and reports macros defined but never used, as information rather than a failure.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NUMBERS = REPO / "arxiv/generated/numbers.tex"
SOURCES = sorted((REPO / "arxiv/source").rglob("*.tex"))

# LaTeX and package commands that are letters-only and would otherwise look like macros.
KNOWN = {
    "begin", "end", "documentclass", "usepackage", "graphicspath", "hypersetup", "title",
    "author", "date", "maketitle", "section", "subsection", "subsubsection", "paragraph",
    "label", "ref", "cite", "citep", "citet", "input", "includegraphics", "caption",
    "centering", "toprule", "midrule", "bottomrule", "appendix", "bibliographystyle",
    "bibliography", "newcommand", "IfFileExists", "paperinput", "texttt", "textbf",
    "textit", "emph", "small", "linewidth", "url", "qquad", "times", "max", "mathrm",
    "text", "tfrac", "frac", "dagger", "ddagger", "item", "enumerate", "itemize",
    "verbatim", "tabular", "table", "figure", "abstract", "document", "percent",
    "textwidth", "hline", "quad", "footnotesize", "normalsize", "newline", "par",
    # math names that are letters-only and capitalised, so they would otherwise read
    # as generated macros
    "Delta", "Sigma", "Pr", "Big", "Bigl", "Bigr", "Large", "Huge",
}
DEFINED = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", NUMBERS.read_text()))

# Two signatures of a reported value typed as a word. Both are deliberately narrow: a
# check that fires on ordinary English gets switched off, and this paper's prose is full of
# legitimate small counts ("three configurations", "two runs", "per thousand claims") that
# are design constants rather than results and have no macro to cite.
#
#   A. A hedge followed by a spelled number. A hedged number is a measurement someone
#      rounded by hand: "roughly eight points wide", "a half-width of about five points",
#      "by up to two tenths of a point", "by a factor of roughly five on the median". No
#      quantity noun is required, because the hedge alone establishes that a measured value
#      is being approximated.
#   B. A spelled number of four or more immediately before a noun this study counts
#      results in: "eleven datasets", "eleven groups", "nine flips". The threshold of four
#      is what keeps the design constants out, since all of them are one, two or three.
NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    "sixty|seventy|eighty|ninety|tenth|tenths|quarter|quarters|half"
)
NUMBER_WORDS_FOUR_UP = (
    "four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    "seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
)
HEDGES = (r"roughly|about|approximately|around|almost|nearly|up\s+to|just\s+over|"
          r"just\s+under|a\s+little\s+over|a\s+little\s+under")
RESULT_NOUNS = "datasets?|flips?|groups?|verdicts?|calls?|repeats?|examples?|folds?|claims?"

TYPED_QUANTITY = (
    re.compile(rf"\b({HEDGES})\s+\b({NUMBER_WORDS})\b", re.I),
    re.compile(rf"\b({NUMBER_WORDS_FOUR_UP})\b(?:\s+\w+){{0,2}}?\s+\b({RESULT_NOUNS})\b",
               re.I),
)

fails: list[str] = []


def strip_verbatim(tex: str) -> str:
    """Blanks verbatim blocks, keeping offsets so line numbers stay right."""
    return re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}",
                  lambda m: " " * len(m.group(0)), tex, flags=re.S)


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*$", "", tex, flags=re.M)


def column_count(spec: str) -> int:
    """Cells implied by a column spec such as {lrrrr} or {l|rr}."""
    return len(re.findall(r"[lcrp]", re.sub(r"p\{[^}]*\}", "p", spec)))


used: set[str] = set()
for path in SOURCES:
    rel = path.relative_to(REPO)
    raw = path.read_text()
    if "---" in raw or "\u2014" in raw or "\u2013" in raw:
        fails.append(f"{rel}: contains a dash run that renders as an em or en dash")
    body = strip_comments(raw)

    prose = strip_verbatim(body)
    for pattern in TYPED_QUANTITY:
        for m in pattern.finditer(prose):
            line = prose.count("\n", 0, m.start()) + 1
            phrase = " ".join(m.group(0).split())
            fails.append(
                f"{rel}:{line}: reported quantity typed as a word, not a macro: {phrase!r}")

    for name in re.findall(r"\\([A-Za-z]+)", body):
        used.add(name)
        if name in KNOWN or name in DEFINED:
            continue
        # Generated macros are CamelCase and start with a capital; anything else is a
        # LaTeX command this checker simply does not know about.
        if name[0].isupper():
            fails.append(f"{rel}: \\{name} is used but numbers.tex does not define it")

    for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", body, re.S):
        ncols, rows = column_count(m.group(1)), m.group(2)
        for line in rows.split(r"\\"):
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            core = re.sub(r"\\(toprule|midrule|bottomrule)", "", line).strip()
            if not core:
                continue
            cells = len(re.findall(r"(?<!\\)&", core)) + 1
            if cells != ncols:
                fails.append(
                    f"{rel}: tabular declares {ncols} columns, row has {cells}: "
                    f"{core[:70]}")

# --- negative numbers need a real minus sign -------------------------------------
# numbers.tex states this rule in its own header and the paper broke it in ten places,
# starting in the abstract, where one interval renders with U+2212 and the next with an
# ASCII hyphen six lines later. A macro whose body starts with "-" must be used inside
# math mode, which is the only way LaTeX produces a minus rather than a hyphen. Checked
# against the macro VALUES rather than a name convention, so a macro that turns negative
# when the data changes is caught the first time it does.
NEGATIVE_MACROS = {name for name, val in
                   re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}",
                              NUMBERS.read_text())
                   if val.strip().startswith("-")}
for path in SOURCES:
    body = strip_verbatim(strip_comments(path.read_text()))
    # Blank the escaped dollar FIRST. Every cost figure in this paper starts with "\$",
    # and treating those as math delimiters mispairs every span after the first one. That
    # exact bug was found in lint_paper_prose.py one commit earlier, and reproduced here
    # within the hour, so it is spelled out rather than left to be rediscovered.
    body = body.replace(r"\$", "  ")
    # Blank out every math span, so what survives is exactly the text-mode prose.
    text_mode = re.sub(r"\$[^$]*\$", lambda m: " " * len(m.group(0)), body)
    for name in NEGATIVE_MACROS:
        for m in re.finditer(rf"\\{name}(?![A-Za-z])", text_mode):
            line = text_mode.count("\n", 0, m.start()) + 1
            fails.append(
                f"{path.relative_to(REPO)}:{line}: \\{name} is negative and is used in text "
                "mode, so it renders with an ASCII hyphen instead of a minus sign; wrap it "
                f"as ${{\\{name}}}$")

# --- arXiv abstract length ------------------------------------------------------
# arXiv's submission form rejects an abstract over ABSTRACT_LIMIT characters
# ("abstracts longer than 1920 characters will not be accepted; abridge your abstract if
# necessary", https://info.arxiv.org/help/prep.html). The abstract in main.tex is written
# against macros, so its source length says nothing about the length arXiv sees; only the
# expanded text does, and three rounds of repair grew it past the limit without anyone
# noticing. The paper is unsubmittable in that state, so this is a gate and not a warning.
ABSTRACT_LIMIT = 1920
MACRO_VALUES = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}",
                               NUMBERS.read_text()))


def expand(tex: str) -> str:
    """The abstract as a reader of the arXiv metadata field would see it."""
    t = strip_comments(tex)
    t = re.sub(r"\\([A-Za-z]+)", lambda m: MACRO_VALUES.get(m.group(1), m.group(0)), t)
    t = re.sub(r"\\(texttt|emph|textbf|textit)\s*\{([^{}]*)\}", r"\2", t)
    t = t.replace("\\%", "%").replace("\\$", "$").replace("\\&", "&")
    t = t.replace("~", " ").replace("$", "")
    t = re.sub(r"\\[A-Za-z]+\*?", " ", t)      # any command the macro file does not define
    t = re.sub(r"\\[^A-Za-z]", " ", t)         # escaped punctuation and the \ space
    t = t.replace("{", "").replace("}", "")
    t = re.sub(r"[ \t]*\n[ \t]*\n[\s]*", "\n\n", t)   # paragraph breaks survive
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


# --- arithmetic a reader can check with a pencil ---------------------------------------
# paired_analysis.py emits a two decimal form of every headline score for the sentences
# that print a DIFFERENCE beside its operands, and says so in a comment that ships in the
# public export. The first version of this check declared seven triples by hand, two of
# them naming difference macros the paper never prints, and it looked only at those seven
# names. So the abstract could print 73.3 against 73.8, call the difference 0.6 and pass,
# while a reader with a pencil got 0.5. A gate that checks arithmetic the paper does not
# print is worth nothing, and that is exactly how the last one passed. This is the check
# that covers what the paper actually prints.
#
# A RELATION is one arithmetic fact, declared at the level of the QUANTITY rather than the
# macro: each of the three slots lists the spellings the paper may use for that quantity,
# lowest precision first. Three rules run over the table.
#
#   1. COVERAGE. Every relation's result must be printed somewhere in the paper, and every
#      difference-shaped macro the paper prints must appear in some relation or be listed
#      in NOT_TWO_OPERAND with a reason. This is the rule the old gate lacked.
#   2. REPRODUCTION, over two windows, because what a reader subtracts is what is near
#      what. In either window, when all three quantities are printed the result must equal
#      the printed operands combined, as decimals, at the result's own precision. Rounding
#      is not an excuse: the fix is to print the operands at the precision the result needs,
#      which is why the two decimal forms exist. A window that prints a difference and only
#      one of its operands hands the reader nothing to subtract, so it is not checked.
#
#      2a. THE PARAGRAPH, the tightest window, with no exemption. This is the rule that
#          holds a relation which cannot reproduce at any precision to printing two of its
#          three numbers and never all three.
#      2b. THE SECTION, which is the window that crosses a float. A table environment is a
#          paragraph of its own, so window 2a never saw an operand in a table beside the
#          difference in the prose that discusses it. That is how Table 5 came to print a
#          Macro BAcc column at mixed precision, 74.10 against 73.8, while the subsection
#          below it reported the difference of those two cells as 0.27, which is neither
#          74.10 - 73.8 = 0.30 nor 74.1 - 73.8 = 0.3. Window 2b puts every macro printed
#          anywhere in a numbered section, floats and captions included, in scope for every
#          relation whose result that section prints. Relations in NO_EXACT_PRECISION are
#          exempt here and only here, because each of those is a per-dataset difference
#          taken from unrounded accuracies whose paragraph says so in prose; window 2a
#          still holds them to printing two of the three.
#   3. FEASIBILITY. Every relation must reproduce for at least one combination of the
#      spellings that exist, or be listed in NO_EXACT_PRECISION with a reason. Where no
#      combination works the paper prints two of the three numbers and never all three,
#      and rule 2 is what holds it to that. A relation listed there that does reproduce is
#      also a failure, so the list cannot outlive the problem it records.
JEV_BACC = ("JevBAcc", "JevBAccTwoDP")
ASTRA_BACC = ("AstraBAcc", "AstraBAccTwoDP")
ASTRA_HIGH_BACC = ("AstraHighBAcc", "AstraHighBAccTwoDP")
CASCADE_BACC = ("CascadeCrossfitBAcc", "CascadeCrossfitBAccTwoDP")
JEV_FVR = ("JevFVR", "JevFVRTwoDP")
ASTRA_FVR = ("AstraFVR", "AstraFVRTwoDP")
ASTRA_HIGH_FVR = ("AstraHighFVR", "AstraHighFVRTwoDP")
CASCADE_FVR = ("CascadeCrossfitFVR", "CascadeCrossfitFVRTwoDP")
MINICHECK_BACC = ("MiniCheckBAcc", "MiniCheckBAccTwoDP")
MINICHECK_BEST_CUT = ("MiniCheckBestCutBAcc", "MiniCheckBestCutBAccTwoDP")

# (minuend spellings, subtrahend spellings, result spellings, kind)
RELATIONS = [
    (JEV_BACC, ASTRA_BACC,
     ("JevAstraDeltaSignedPP", "JevAstraDeltaSignedTwoDP"), "diff"),
    (JEV_BACC, ASTRA_BACC, ("JevAstraDeltaPP",), "absdiff"),
    (ASTRA_BACC, ASTRA_HIGH_BACC, ("AstraAstraHighDeltaSignedPP",), "diff"),
    (ASTRA_BACC, ASTRA_HIGH_BACC, ("AstraAstraHighDeltaPP",), "absdiff"),
    (JEV_BACC, ASTRA_HIGH_BACC, ("JevAstraHighDeltaSignedPP",), "diff"),
    (JEV_FVR, ASTRA_FVR, ("FVRGapPP", "FVRGapTwoDP"), "diff"),
    (CASCADE_BACC, ASTRA_BACC, ("CascadeAstraDeltaPP",), "diff"),
    (CASCADE_BACC, JEV_BACC, ("CascadeJevDeltaPP", "CascadeCrossfitMinusJevPP"), "diff"),
    (CASCADE_FVR, ASTRA_FVR, ("CascadeAstraFVRDeltaPP",), "diff"),
    (CASCADE_FVR, JEV_FVR, ("CascadeJevFVRDeltaPP",), "diff"),
    # The neighbour Table 4 prints directly beneath the cascade and the paper never
    # differenced: on accuracy the two round to the same value at one decimal, which is why
    # the column is printed at two, and on the guard metric the gap is six points.
    (CASCADE_BACC, ASTRA_HIGH_BACC, ("CascadeAstraHighDeltaPP",), "diff"),
    (CASCADE_FVR, ASTRA_HIGH_FVR, ("CascadeAstraHighFVRDeltaPP",), "diff"),
    (("CascadeInSampleBAcc", "CascadeInSampleBAccTwoDP"), CASCADE_BACC,
     ("SelectionOptimismPP",), "diff"),
    (MINICHECK_BEST_CUT, MINICHECK_BACC, ("MiniCheckRetuningHeadroomPP",), "diff"),
    (JEV_BACC, MINICHECK_BEST_CUT, ("MiniCheckGapToJevAtBestCutPP",), "diff"),
    # Unsigned. The three sentences that print it state the direction in words, because
    # the signed spelling (earlier minus reported) reads as a fall in every one of them
    # and was once carried into a claim that the ordering reversed.
    (("JevRerunBAcc", "JevRerunBAccTwoDP"), JEV_BACC, ("JevRerunDeltaPP",), "absdiff"),
    (("SamePromptRepeatBAcc",), ("SamePromptRepeatBAccOther",),
     ("SamePromptRepeatDeltaPP",), "diff"),
    (("PromptParityPreFVR",), ("PromptParityPostFVR",),
     ("PromptParityFVRGapShiftPP",), "diff"),
    (("JevBAccAggreFactCNN",), ("AstraBAccAggreFactCNN",), ("DeltaAggreFactCNNPP",), "diff"),
    (("JevBAccExpertQA",), ("AstraBAccExpertQA",), ("DeltaExpertQAPP",), "diff"),
    (("AstraBAccTofuEvalMediaS",), ("JevBAccTofuEvalMediaS",),
     ("MaxAbsRemainingPP",), "absdiff"),
    (("DeltaAggreFactCNNPP",), ("DeltaExpertQAPP",), ("WidestTwoSumPP",), "sum"),
    (("DeltaAggreFactCNNPP",), ("DeltaExpertQAPP",), ("WidestTwoSumAbsPP",), "sumabs"),
    (("WidestTwoSumAbsPP",), ("RemainingSumAbsPP",), ("SumAbsDeltasPP",), "sum"),
    # The step section 7.3 used to omit: the signed residue of the per-dataset
    # cancellation, divided by the number of datasets, is the headline.
    (("SignedSumDeltasPP",), ("NDatasets",),
     ("JevAstraDeltaSignedPP", "JevAstraDeltaSignedTwoDP"), "quotient"),
    (("RemainingSumAbsPP",), ("NRemainingDatasets",), ("MeanAbsRemainingPP",), "quotient"),
    (("OracleBAcc", "OracleBAccTwoDP"), JEV_BACC, ("OracleHeadroomPP",), "diff"),
    (("CascadeAtNinetyNineBAcc", "CascadeAtNinetyNineBAccTwoDP"), ASTRA_BACC,
     ("CascadeAtNinetyNineMinusAstraPP",), "diff"),
    # The random-escalation control against the frontier arm. Section 9.5 prints the
    # control's level, the frontier arm's level and the difference in one paragraph,
    # which is exactly the arithmetic a reader checks when a paper says two things do
    # not separate.
    (("RandomAtNinetyBAcc", "RandomAtNinetyBAccTwoDP"), ASTRA_BACC,
     ("RandomVsAstraDeltaPP",), "diff"),
    # The same guard penalty, recomputed against the run the prompt correction replaced.
    # Section 12 prints it beside the reported one, so the two must be differences of the
    # same minuend against the two spellings of the comparator arm.
    (CASCADE_FVR, ("PromptParityHighPreFVR",),
     ("CascadeAstraHighFVRPreParityDeltaPP",), "diff"),
    # Section 7.5 prints all three of these in one sentence, so the division is one a
    # reader does. It is not difference-shaped, so rule 1 would never have demanded it.
    (("NeitherCorrect",), ("NeitherExpectedIndependent",),
     ("JointFailureExcess",), "quotient"),
]

# A relation whose result cannot be reproduced from any spelling of its operands that
# exists, with the reason. Rule 2 then keeps the paper from printing all three numbers in
# one paragraph, which is the only place a reader would try the arithmetic.
NO_EXACT_PRECISION = {
    "DeltaAggreFactCNNPP": (
        "the per-dataset cells exist only at one decimal and the difference only at two, "
        "so no printed pair reproduces it: Section 7.3 prints the two cells and Appendix A "
        "the difference, never both"),
    "DeltaExpertQAPP": (
        "same as AggreFact-CNN: one decimal cells against a two decimal difference"),
    "MaxAbsRemainingPP": (
        "the widest of the nine remaining gaps is computed from unrounded accuracies and "
        "is 9.6, while the difference of the rounded TofuEval-MediaS cells in Table 8 is "
        "9.7; Appendix A says so in prose and prints the two apart"),
    "SumAbsDeltasPP": (
        "each of the three is rounded to one decimal from values that sum exactly, so the "
        "rounded parts add to 75.6 against a rounded total of 75.5; Appendix A prints the "
        "two parts, Section 7.3 the total, and the sentence in Appendix A that links them "
        "says the total is the sum taken before rounding and that adding the printed parts "
        "need not land on it"),
}

# A difference-shaped macro the paper prints that is not the result of a two-operand
# relation over other printed macros, with the reason it cannot be one.
NOT_TWO_OPERAND = {
    "RemainingNineSumPP": (
        "the signed sum of the nine per-dataset differences, which Table 8 carries as "
        "cells rather than as difference macros, so there is no pair to subtract"),
    "ConfMinusRandomMinPP": (
        "the smallest of the six per-threshold confidence-minus-random gaps, so it selects "
        "one of a set rather than subtracting two printed numbers"),
    "ConfMinusRandomMaxPP": (
        "the largest of the same six gaps, on the same reading"),
    "FVRGapHolmBreakFamilyN": (
        "a family size, not a difference: the count of tests at which Holm's largest "
        "possible multiplier would carry the FVR gap's raw p to 0.05. It carries the "
        "FVRGap stem because it is about that comparison, not because it subtracts two "
        "reported quantities"),
}

# Names that say the macro is a difference between two reported quantities. Interval
# bounds, test statistics and counts carry these stems too, so they are excluded by name.
DIFFERENCE_SHAPED = re.compile(
    r"Delta|Gap|Optimism|Headroom|Shift|AbsRemaining|SumAbs|NineSum|Minus")
NOT_A_DIFFERENCE = re.compile(
    r"CILo|CIHi|CIWidth|McNemar|Discordant|Count|Pct|Stat|Median|Mean(?!Abs)|Ratio")


def _dec(name: str):
    raw = MACRO_VALUES.get(name)
    if raw is None:
        return None, None
    try:
        return Decimal(raw.replace(",", "").replace("+", "")), len(raw.partition(".")[2])
    except InvalidOperation:
        return None, None


def _combine(kind: str, a: Decimal, b: Decimal) -> Decimal:
    if kind == "diff":
        return a - b
    if kind == "absdiff":
        return abs(a - b)
    if kind == "sum":
        return a + b
    if kind == "sumabs":
        return abs(a) + abs(b)
    if kind == "quotient":
        return a / b
    raise AssertionError(f"unknown relation kind {kind!r}")


def _check(kind: str, a_name: str, b_name: str, d_name: str):
    """(ok, what the printed operands give) at the printed result's own precision."""
    a, _ = _dec(a_name)
    b, _ = _dec(b_name)
    d, dp = _dec(d_name)
    if a is None or b is None or d is None or (kind == "quotient" and b == 0):
        return None, None
    got = _combine(kind, a, b).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    return got == d, got


declared_names: set[str] = set()
for a_sp, b_sp, d_sp, kind in RELATIONS:
    declared_names.update(a_sp + b_sp + d_sp)
    for name in a_sp + b_sp + d_sp:
        if name not in MACRO_VALUES:
            fails.append(f"relation \\{d_sp[0]}: \\{name} is not a plain decimal in "
                         "numbers.tex, so the relation cannot be checked")
    # rule 1, per relation
    if not any(name in used for name in d_sp):
        spellings = ", ".join("\\" + n for n in d_sp)
        fails.append(
            f"relation \\{d_sp[0]}: the paper prints no spelling of this result "
            f"({spellings}), so declaring it checks nothing")
    # rule 3
    outcomes = [_check(kind, a, b, d) for a in a_sp for b in b_sp for d in d_sp]
    feasible = any(ok for ok, _ in outcomes if ok is not None)
    listed = d_sp[0] in NO_EXACT_PRECISION
    if not feasible and not listed:
        shown = "; ".join(
            f"\\{a} {kind} \\{b} = {got} against \\{d} = {_dec(d)[0]}"
            for (a, b, d), (ok, got) in zip(
                [(a, b, d) for a in a_sp for b in b_sp for d in d_sp], outcomes)
            if ok is not None)
        fails.append(
            f"relation \\{d_sp[0]}: no spelling of its operands reproduces it ({shown}). "
            "Print the operands at the precision the result needs, or record why that is "
            "impossible in NO_EXACT_PRECISION")
    if feasible and listed:
        fails.append(
            f"relation \\{d_sp[0]}: NO_EXACT_PRECISION says it cannot reproduce, and it "
            "does. Delete the entry rather than leave a stale excuse in the gate")

for name in sorted(NO_EXACT_PRECISION):
    if name not in declared_names:
        fails.append(f"NO_EXACT_PRECISION names \\{name}, which no relation declares")
for name in sorted(NOT_TWO_OPERAND):
    if name not in used:
        fails.append(f"NOT_TWO_OPERAND names \\{name}, which the paper no longer prints")

# rule 1, over every difference-shaped macro the paper prints
for name in sorted(used & DEFINED):
    if not DIFFERENCE_SHAPED.search(name) or NOT_A_DIFFERENCE.search(name):
        continue
    if name in declared_names or name in NOT_TWO_OPERAND:
        continue
    fails.append(
        f"\\{name} is printed in the paper and reads as a difference between two reported "
        "quantities, but no relation declares it. Add it to RELATIONS, or to "
        "NOT_TWO_OPERAND with the reason it has no two operands")

def windows(body: str):
    """(name, text, exemptions honoured) for every window a reader subtracts within.

    Window 2a is the paragraph. Window 2b is everything from one \\section to the next,
    which is the only window that holds a table and the prose that differences its cells
    at the same time, since a float is a paragraph of its own.
    """
    for para in re.split(r"\n\s*\n", body):
        yield "one paragraph", para, False
    for chunk in re.split(r"\\section\*?\{", body)[1:]:
        yield "one section, floats included", chunk, True


# rule 2, over the paragraph and then over the section
for path in SOURCES:
    body = strip_comments(path.read_text())
    for window, text, exempt in windows(body):
        printed = {n for n in re.findall(r"\\([A-Za-z]+)", text)}
        for a_sp, b_sp, d_sp, kind in RELATIONS:
            if exempt and d_sp[0] in NO_EXACT_PRECISION:
                continue
            for a in (n for n in a_sp if n in printed):
                for b in (n for n in b_sp if n in printed):
                    for d in (n for n in d_sp if n in printed):
                        ok, got = _check(kind, a, b, d)
                        if ok is False:
                            better = [
                                (x, y, z) for x in a_sp for y in b_sp for z in d_sp
                                if _check(kind, x, y, z)[0]]
                            hint = (f" Print \\{better[0][0]}, \\{better[0][1]} and "
                                    f"\\{better[0][2]} instead."
                                    if better else
                                    " No spelling of these three reproduces, so print two "
                                    f"of them and not all three in {window}.")
                            fails.append(
                                f"{path.relative_to(REPO)}: {window} prints "
                                f"\\{a} = {_dec(a)[0]}, \\{b} = {_dec(b)[0]} and "
                                f"\\{d} = {_dec(d)[0]}, but the first two give {got} at "
                                f"that precision.{hint}")

abstract_src = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                         (REPO / "arxiv/source/main.tex").read_text(), re.S)
if abstract_src is None:
    fails.append("arxiv/source/main.tex: no abstract found, so its length cannot be checked")
    abstract_len = 0
else:
    abstract_txt = expand(abstract_src.group(1))
    abstract_len = len(abstract_txt)
    leftover = sorted(set(re.findall(r"\\[A-Za-z]+", abstract_txt)))
    if leftover:
        fails.append("arxiv/source/main.tex: the abstract still holds LaTeX after "
                     f"expansion, so its length is not measurable: {leftover[:5]}")
    if abstract_len > ABSTRACT_LIMIT:
        fails.append(
            f"arxiv/source/main.tex: the abstract expands to {abstract_len} characters and "
            f"arXiv refuses more than {ABSTRACT_LIMIT}; cut "
            f"{abstract_len - ABSTRACT_LIMIT} characters")

unused = sorted(DEFINED - used)
print(f"abstract: {abstract_len} characters expanded, limit {ABSTRACT_LIMIT} "
      f"({len(abstract_txt.split()) if abstract_src else 0} words)")
print(f"{len(SOURCES)} source files | {len(DEFINED)} macros defined | "
      f"{len(DEFINED) - len(unused)} used | {len(unused)} unused")
print(f"\npaper macro gate: {'FAIL' if fails else 'PASS'}")
for f in sorted(set(fails)):
    print("  x", f)
if unused:
    print("\n  (defined and not used, which is allowed:)")
    print("   ", ", ".join(unused))
sys.exit(1 if fails else 0)

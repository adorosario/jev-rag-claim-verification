#!/usr/bin/env python3
"""Check the published article against medium/generated/numbers.json.

Publishing a number the raw data does not support is the failure this repo exists
to prevent (CLAUDE.md rule 4). There are two kinds of number in the article and
each gets its own guarantee:

* **Quoted in prose** -> the exact formatted value must appear in the text.
* **Shown in a figure** -> figures are rendered from this same numbers.json by
  scripts/make_article_figures.js, so they cannot drift. What is checked is that
  every figure exists, is referenced, and was rebuilt AFTER the data changed.

    docker compose run --rm dev uv run python scripts/verify/check_article_numbers.py

The default target is the article AS PUBLISHED. It used to be medium/article.md, the
unpublished v1 draft, which the public export withdraws because it makes an accuracy
claim the paper declines to make: so the one command the released README tells a reader
to run died on a missing file in the published repository, which nothing caught, because
the release check verifies that files are present and not that the commands work.

The figure checks that need the PNGs run only where the PNGs are. The export ships the
article and its numbers but not the rendered figures, which live in the published Medium
post, so there the file checks are skipped and said to be skipped. The checks that read
the article text alone always run.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
d = json.loads((REPO / "medium/generated/numbers.json").read_text())
ARTICLE = sys.argv[1] if len(sys.argv) > 1 else "medium/article-v3.md"
art = (REPO / ARTICLE).read_text()
bad: list[str] = []
checked = 0

j, a, h = d["jev"], d["astra"], d["astra_high_effort"]


# Values the PUBLISHED article carries that the current data no longer produces, each
# with the reason and the log entry that records it. The article is frozen: it went out
# under the first author's byline before the preprint, and it cannot be quietly edited to
# agree with data that moved afterwards. So a divergence here is reported and named, not
# silenced, and a label whose value has come back into agreement fails, because a list of
# known divergences that outlives the divergence is how a stale exception becomes a lie.
SUPERSEDED = {
    "cascade 0.90 cost": (
        "2.95",
        "the escalation rule in the code was not the rule the paper states, and fixing it "
        "moved the cascade costs by about a cent; the published figure predates the fix. "
        "See docs/reference/incident-log.md, 'the cascade escalation rule was not the one "
        "the paper stated', and Appendix C of the preprint. The article's whole cascade "
        "operating point is superseded by the preprint's cross-fitted one, which is the "
        "subject of its own entry in the same log."),
}
superseded_seen: list[str] = []


def chk(label, expected, fmt="{:.1f}"):
    """The formatted value must appear verbatim in the article text."""
    global checked
    checked += 1
    s = fmt.format(expected)
    if s not in art:
        if label in SUPERSEDED:
            published, why = SUPERSEDED[label]
            if published in art:
                superseded_seen.append(f"{label}: the article says {published}, the data "
                                       f"now gives {s}. {why}")
                return
            bad.append(f"{label}: '{s}' is not in the article, and neither is the "
                       f"superseded value '{published}' this check expected to find")
            return
        bad.append(f"{label}: '{s}' is not in the article")


# --- numbers quoted in the prose -------------------------------------------------
chk("jev bacc", j["mean_bacc"] * 100)
chk("astra bacc", a["mean_bacc"] * 100)
chk("astra high bacc", h["mean_bacc"] * 100)
chk("delta", abs(d["delta"]["jev_minus_astra_pp"]))
chk("delta ci lo", abs(d["delta"]["ci"][0] * 100))
chk("delta ci hi", d["delta"]["ci"][1] * 100)
chk("mcnemar p", d["paired"]["mcnemar_p"], "{:.2f}")
chk("agreement", d["paired"]["agreement"] * 100)
chk("jev only correct", d["paired"]["jev_only_correct"], "{:d}")
chk("astra only correct", d["paired"]["astra_only_correct"], "{:d}")
chk("jev cost", j["cost_per_1k"], "{:.3f}")
chk("astra cost", a["cost_per_1k"], "{:.2f}")
chk("astra high cost", h["cost_per_1k"], "{:.2f}")
chk("cost ratio", d["ratios"]["cost"], "{:.0f}")
chk("jev p50", j["p50_ms"], "{:.0f}")
chk("astra p50", a["p50_ms"], "{:,.0f}")
chk("astra high p50", h["p50_ms"], "{:,.0f}")
chk("jev fvr", j["fvr"] * 100)
chk("astra fvr", a["fvr"] * 100)
chk("n paired", d["n_paired"], "{:d}")
chk("raw accuracy jev", d["raw_accuracy"]["jev"] * 100)
chk("raw accuracy astra", d["raw_accuracy"]["astra"] * 100)
chk("jev mean input tokens", j["mean_input_tokens"], "{:,.0f}")
chk("astra mean input tokens", a["mean_input_tokens"], "{:,.0f}")
chk("jev AggreFact-CNN", j["per_dataset"]["AggreFact-CNN"] * 100)
chk("gating 0.99 precision", [g for g in d["gating"] if g["threshold"] == 0.99][0]["verified_precision"] * 100)
cas90 = [c for c in d["cascade"] if c["threshold"] == 0.9][0]
chk("cascade 0.90 escalated", cas90["escalated"] * 100, "{:.0f}")
chk("cascade 0.90 bacc", cas90["bacc"] * 100)
chk("cascade 0.90 cost", cas90["cost_per_1k"], "{:.2f}")

if "latency_p50" in d["ratios"]:
    bad.append("numbers.json exposes a latency ratio; those runs were not a controlled measurement")

# --- numbers shown in figures ----------------------------------------------------
FIGURES = ["headline", "cost", "per-dataset", "gating", "cascade"]
nums_mtime = (REPO / "medium/generated/numbers.json").stat().st_mtime
have_figures = (REPO / "medium/figures").is_dir()
if not have_figures:
    print("medium/figures is absent, so the figure files are not checked here: the "
          "rendered figures live in the published Medium post, not in this package. "
          "The checks below that read the article text still run.")
for name in FIGURES:
    checked += 1
    p = REPO / f"medium/figures/{name}.png"
    if have_figures:
        if not p.is_file():
            bad.append(f"figure {name}.png is missing")
        elif p.stat().st_mtime < nums_mtime:
            bad.append(f"figure {name}.png is older than numbers.json: re-run scripts/make_article_figures.js")
    if f"figures/{name}.png" not in art:
        bad.append(f"figure {name}.png is not referenced by the article")
if art.count("![") != len(FIGURES):
    bad.append(f"the article embeds {art.count('![')} images but {len(FIGURES)} figures are expected")
if "|---" in art:
    bad.append("a markdown table survives in the article; Medium does not render them, so it must be a figure")

for label in SUPERSEDED:
    if label not in [t.split(":")[0] for t in superseded_seen]:
        bad.append(f"{label} is listed as superseded but did not diverge. Remove it from "
                   "SUPERSEDED: the list must not outlive the divergence it records")

print(f"checked {checked} values and figures against numbers.json")
if superseded_seen:
    print("\nSUPERSEDED, recorded rather than silenced:")
    for t in superseded_seen:
        print(" -", t)
if bad:
    print("\nMISMATCHES:")
    for b in bad:
        print(" -", b)
    sys.exit(1)
print("article matches the data")

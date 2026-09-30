# Limitations in full

Long-form accounting behind Section 12 of *Jev-as-a-Judge for RAG Claim Verification:
Confidence-Gated Cascading Between a Non-Generative Verifier and a Frontier LLM*.

Section 12 of the paper names every limitation with the number that matters for it, and
Appendix E gives each one a short entry. This is the version with the derivations: the
family-boundary sensitivity of the Holm correction, the MiniCheck anchor's full diagnosis,
the same-prompt repeat's coverage and its reasoning-token accounting. It lives here rather
than in the paper so the paper stays readable. Nothing was dropped in the move.

Every value is expanded from `arxiv/generated/numbers.tex`, which
`scripts/verify/paired_analysis.py` regenerates from the locked predictions, so this
document cannot disagree with the paper. It is generated, not written:

    docker compose run --rm dev uv run python scripts/make_limitations_doc.py

Section 12 lists the limitations of this study. This appendix gives each
one its numbers, in the same order. The long-form accounting behind them, including the
family-boundary sensitivity of the correction below, the MiniCheck anchor's full diagnosis
and the same-prompt repeat's per-item reasoning-token coverage, is
`docs/reference/limitations-in-full.md` in the released package, generated from this
appendix and the same macro file so it cannot disagree with what is printed here.

### Indistinguishable is not equivalent

The paired difference is -0.6 points on Jev minus Astra, interval
[-5.6, 4.5] over 10000 resamples, wide
enough to hold a several-point advantage either way, and the exact McNemar at
p = 0.49 is a failure to reject and nothing more. We preregistered no
equivalence margin and ran no equivalence test, so no claim of statistical equivalence
appears anywhere here. The thinnest minority class in any component dataset is
4 claims, which makes the per-dataset cells of
Appendix A descriptive rather than inferential.

### The 3 differences that clear both instruments, and what the correction does to them

Section 5 fixes the criterion per estimand: on false verification, where the
interval and the test rest on the same gold-unsupported claims, both instruments are
required; on macro-balanced accuracy, where the interval is on the macro average and the
test on pooled per-item correctness, the interval governs and the p-value corroborates.
3 comparisons meet it uncorrected, and 2 of them
survive the Holm-Bonferroni adjustment the criterion also requires.

The false-verification gap between the single systems is 9.1 points, interval
[4.55, 13.64] over the 198 gold-unsupported claims, exact
McNemar p = 0.000534 on 22 and
4 discordant claims, p = 0.005 adjusted. That one
is real on every reading.

Four cautions attach to the gap between the two single systems. It is a difference in
permissiveness rather than in skill, and any system can lower it by calling fewer claims
supported, which is why Section 11 asks for it to be priced against the
false-alarm rate instead of read alone. It carries the Holm-Bonferroni adjustment
printed above, at 0.005 adjusted against 0.000534 raw, which is a
correction over the 13 tests this paper reports and not over the wider
space of comparisons these data would support. It is not the same estimand as the accuracy
comparison it is set against: the accuracy tie is a macro average over 11 datasets
of roughly 45 claims each, while the gap is a rate pooled over all
198 gold-unsupported claims with no per-dataset weighting, so pooling gives
it more effective sample and the two are not two readings of one measurement on the same
footing. And it comes from one sample and one pair of model versions, so what the contrast
shows is that these 495 claims are enough to separate the systems on one quantity and
not on the other, rather than that the two quantities were tested head to head.

The cascade's guard penalty against the high-effort arm is the other survivor,
6.09 points, interval [2.53,
10.10], median exact McNemar
p = 0.00049, p = 0.005 adjusted.
It is a median rather than a single exact test, and its direction is against the
architecture this paper proposes. The cascade's guard penalty against Astra is
5.08 points, interval [1.52,
9.09], median p = 0.00635 and
p = 0.057 adjusted, so we report it as clearing both instruments
uncorrected and sitting on the 0.05 boundary once corrected, not as a demonstrated
separation. Dropping the two MiniCheck tests leaves it at
p = 0.057 and dropping any single member also leaves it
above 0.05. Holm multiplies it by the number of tests whose raw p is at least its own,
so the 8 members above it in the step-down decide it: it
clears only when the 2 largest of those go, for a family of
11, where it adjusts to
p = 0.044. Unlike the gap between the single systems, which
holds for any family smaller than 94 tests, this result turns on
where the family boundary falls, and we report it against the boundary fixed in code before
the adjustment was computed rather than against one that would let it through.

### Raw and adjusted p for the whole family

Table 11 gives every exact McNemar test this paper reports, raw and adjusted.
6 of the 13 entries are medians over
cross-fitting repeats rather than single exact tests, computed on overlapping resamples of
the same 495 claims, and a median of p-values has no null distribution of its own,
so Holm over a family containing them is descriptive on those members and we claim no error
rate for them. That includes one of the 2 survivors, which is why
Section 9.3 reports it with the interval beside the test rather than
on the p alone. One further comparison clears one instrument and not the other and is
therefore not counted: the cascade's guard improvement over Jev alone is
-4.01\ points with an interval of [-8.08,
-0.51] that excludes zero, while its median p of
0.05737 falls below 0.05 in only
76 of 200 repeats.
table[htbp]

Holm-Bonferroni over the 13 exact McNemar tests this paper
reports, the family fixed in code before the adjustment was computed. Rows marked
 are medians over cross-fitting repeats rather than single exact tests. The two
MiniCheck rows are evidence for excluding that arm rather than results about it.
tab:holm
tabularlrr

Comparison & raw p & Holm p \\

Jev vs.\ Astra, accuracy & 0.49 & 1.000 \\
Jev vs.\ Astra high effort, accuracy & 0.48 & 1.000 \\
Astra vs.\ Astra high effort, accuracy & 1.00 & 1.000 \\
Jev vs.\ Astra, false verification & 0.000534 & 0.005 \\
Cascade vs.\ Astra, accuracy^ & 0.20 & 1.000 \\
Cascade vs.\ Astra, false verification^ & 0.00635 & 0.057 \\
Cascade vs.\ Jev, accuracy^ & 0.86 & 1.000 \\
Cascade vs.\ Jev, false verification^ & 0.05737 & 0.459 \\
Cascade vs.\ Astra high, accuracy^ & 0.23 & 1.000 \\
Cascade vs.\ Astra high, false verification^ & 0.00049 & 0.005 \\
Fixed-threshold guard (Table 9) & 1.00 & 1.000 \\

Jev vs.\ MiniCheck & 1.6 10^-16 & 2.1 10^-15 \\
Astra vs.\ MiniCheck & 4.0 10^-14 & 4.8 10^-13 \\

tabular
table

### What was selected in sample, and what was not

The gating sweep of Section 8, the tuned cascade of
Section 10 and every row of the confidence column of Table 9
are scored on the claims their thresholds were read off. The reportable cascade is the
exception: its threshold is chosen out of fold by 200 repeats of
5-fold cross-fitting, so the 1.72 points between the
in-sample best (75.82) and the cross-fitted estimate is a penalty that
procedure avoids rather than one the headline pays. What cross-fitting does not undo is the
layer outside it: the threshold grid, the decision to route on confidence at all and
macro-balanced accuracy as the objective were fixed with the development outcomes visible.
The selection is unstable at this size, picking one threshold 669 times
and a very different one 253 times across
1000 fold-fits. The gate thresholds in the sweep and the 11\
per-dataset cells sit outside the Holm family of Section 5, which covers the
exact McNemar tests between the configurations this paper reports, and carry no
multiple-comparison correction. So does the recomputation of the guard penalty against the
superseded pre-parity run: that
recomputation is a sensitivity check on a retracted comparator rather than a result, and is
neither a member of the family nor one of the 3 comparisons that clear
both instruments.

### The cascade's sign is unstable in two different ways

On the discordant items the exact McNemar works from, the cascade is ahead in
198 of 200 repeats and Astra in
0; the test clears 0.05 in
16 of them, all 16\
favouring the cascade. On the macro-balanced accuracy the paper reports it is not stable:
the cascade lands below Astra in 68 of the repeats and
43.1\
zero.

### One draw of the fold assignment

Every cross-fitted mean averages 200 draws of the fold assignment at one
recorded seed, so each carries a Monte Carlo error separate from the sampling uncertainty
over claims: 0.062 points on macro-balanced accuracy,
0.048 on cost per thousand claims, 0.467 points on the
escalation share and 0.068 points on the false-verification rate. Read those
quantities to that precision rather than to the digits we print. A different seed moves
them together, since they descend from how often the fold-fits choose
0.80, so this is one draw and not several.

### The evaluation sample is a released list, not a rerunnable draw

The 495 example IDs are a fixed list, hashed and released with the predictions, and no
code here regenerates the draw, so a reader can check exactly what was evaluated and cannot
re-derive how it was chosen. The list came out of an exploratory run of the decision model in
which every retained row succeeded, so this sample cannot speak to that system's failure
rate, and the absence of model-caused failures in Section 5 is an independent
observation only for the frontier configurations and MiniCheck. No claim here rests on
either system's failure rate.

### The cascade is an offline replay

Both systems answered all 495 claims independently and the cascade is composed from
those stored predictions. No claim was re-sent and no deployed router was built. A live
implementation adds a second network hop on the escalated fraction, with queueing and
failure behaviour this study does not measure.

### Every cost figure is two price lists on one day

Costs are measured token counts at list prices from two vendors' public pages on a single
date, archived and hashed as `configs/` `pricing_2026-09-20.yaml` in the
released package. Nothing was invoiced, so these are modelled costs. Three consequences:
prices move, and a ratio from a snapshot ages the moment either vendor reprices, which is
why the dated file ships rather than the ratio being asserted; list price is not what a
buyer at volume pays, and negotiated rates are public for neither vendor and would not move
the arms by the same factor; and the decision model's free output tokens are its vendor's
commercial policy, not a property of the architecture, so reading
187 as durable would be reading a price sheet as an engineering
result. What survives repricing is that a binary decision need not be billed at generation
rates.

### Each arm was run once

Every number comes from a single pass of each configuration at the provider's default
sampling settings, and no repeatability study was run. One same-prompt repeat of the
frontier arm exists in the released artifacts: the discarded first pass of
Appendix C and the pass that replaced it carry the same prompt hash,
reasoning effort and model, differing only in our completion-token cap, raised from
16 to 256. That shared prompt is the
superseded pre-parity wording rather than the corrected prompt behind every frontier number
here, so the bound is measured on a prompt no reported configuration used. On the
411 claims the capped pass answered, the two agree on every verdict:
0 flips and macro-balanced accuracy 77.55 against
77.55, a difference of 0.00. It is a narrow bound. Those rows are the ones whose answers fitted
the smaller budget, so they are the short-answer end of the sample and the end where the model
reports zero reasoning tokens 395 times; the remaining
84 claims are untested for repeatability, and they are where every
prompt-parity flip landed. No comparable repeat exists for the high-effort configuration.

For the decision model an earlier pass over the identical example IDs is on disk and is the
file the example-ID list is drawn from. Between it and the run behind this paper,
7 of 495 verdicts changed and macro-balanced accuracy rose from
72.44 to 73.27, 0.8 points. Both passes sit below the
frontier arm's 73.83, so the movement widens the deficit rather than reversing
anything; what it does is exceed the headline gap of 0.6 points that
Section 6 reports.

On the frontier arm we varied the prompt once, in the parity correction of
Appendix C, and every run on both sides of that edit is released. Two runs
of a sampled model differ by the edit and by a fresh draw at once, and this study cannot
separate them: the repeat above covers only rows where the model does no sampled reasoning,
and all 9 of the 9 flips the edit moved
fall outside those rows. The direction is worth printing because it is not the direction the
edit pushes: the added trigger makes NOT_SUPPORTED easier to reach, and
2 of the flips went that way against
7 the other. On the low-effort arm the edit moved
9 of 495 verdicts and macro-balanced accuracy from
73.06 to 73.83, all 9 of them on
gold-supported claims and 0 on gold-unsupported ones, so it
moved the false-verification gap by 0.00 points and that arm's rate
stands at 14.1\

That is the low-effort arm and it does not speak for the other. The same edit moved
5 verdicts on the high-effort arm,
1 of them on a gold-unsupported claim, taking that arm's
false-verification rate from 12.6\
macro-balanced accuracy from 73.93 to 74.11. That
rate is one leg of the guard penalty that is one of the 2 survivors,
so rather than certify the survivor as unaffected we recomputed it against the run the
correction replaced: 6.60 points there against
6.09 reported, median exact McNemar
p = 0.00024, and all
200 repeats separating on either run. The
correction moved that result towards the smaller number, which is the one we report. The
decision model's prompt was never varied at all.

What the movement did disturb is the sign of the headline comparison. Jev scores
73.27. Against the pre-correction frontier run at 73.06 the decision
model led; against the corrected run at 73.83 the ordering reverses, to
-0.56 points on Jev minus Astra. So a correction to our own prompt,
made for a reason unrelated to the outcome, moved the comparator further than the entire
difference this paper reports and reversed its direction. That is the strongest available
illustration of this paper's own thesis, and both runs are released so a reader can check
the ordering.

### Latency was not measured under control

The bulk runs were issued concurrently, so their per-request timings describe our harness as
much as either system. No latency figure or ratio appears in this paper, and the controlled
single-request protocol the project specifies was not run.

### Contamination cannot be ruled out

LLM-AggreFact and its components have been public long enough that any system here may have
seen them in training, and we cannot inspect either vendor's training data. A
contamination-resistant holdout of newly written claims exists for this project and is not
part of this study.

### One frontier family, two reasoning efforts

Both efforts ran under one prompt, byte for byte, which ends by forbidding reasoning in the
response (Appendix B), though not under one completion-token cap,
256 against 4096, because the cap covers the reasoning too and the
low one would have truncated the high arm. On the convention of
Table 2 the two efforts differ by
-0.3 points at exact McNemar p = 1.00, while
the reasoning the provider reports moves from a median of 0 tokens to
21. So the contrast bounds what the effort parameter buys, with the
headroom it needs, under that output-format instruction, and it is not a test of a prompt
that invites reasoning: the provider-default configuration was never run. A single family at
a single point in time supports no statement about frontier models in general.

### An arm we ran and excluded

We ported MiniCheck (tang2024minicheck) from the authors' public release
(tang2024aggrefactrepo) as a third system and it scored 61.6\
[56.3, 66.7] macro-balanced accuracy on the same 495\
claims. That is not a threshold artifact: sweeping 454 cutpoints buys
0.9 points at the best reachable cut, still
10.8 points below Jev. It was also the external anchor for harness
validity, and the published LLM-AggreFact figure is 75.0 on the test
split against our 61.6 on the dev claims, a shortfall of
13.4 points that our own interval's half-width of
5.2 points cannot absorb. The conclusion runs both ways: we exclude
the arm, and we do not claim the rest of the harness is sound. What the failure implicates is
narrower than the whole harness. Not the metric layer, a second implementation cross-checked
against established libraries, which produces the Jev and Astra rows; not the gold labels or
the join, since all four arms are scored against one gold vector built once from the pinned
parquet, so an error there would move every arm together. What it does implicate is
everything specific to our adapter: chunking, evidence truncation and the aggregation of
per-chunk entailment scores, none of which the API arms use.

### The published reference we quote is a partial capture

The file we take that reference from holds 4 rows, and quoting only
the one that excuses our port would be the selective reading Section 10 objects
to, so here are all of them on the test split: Bespoke-MiniCheck-7B
77.4, Claude-3.5-Sonnet
77.2, `gpt-4o-2024-05-13`
75.9, MiniCheck-Flan-T5-L 75.0. It is
a capture made to pin the harness anchor, not a survey: the leaderboard lists systems we did
not keep, and the rows we kept were not chosen by any stated rule, so these figures place
nobody in a field and bound nothing. Nor are they a comparison with our results, which are
macros over 495 dev claims under one prompt written to be semantically identical across
arms. They do not establish that our short-answer prompt cost the frontier arm points, and
they do not establish that it did not.

### The calibration comparison has one participant

Jev returns a native probability. The provider of GPT-6 Astra rejects the logprobs parameter
for this model, recorded verbatim in the released `runs/model_live_check.json`, and
no self-reported confidence was elicited, so the frontier configurations have no probability
of any provenance. Every calibration statement here is about Jev, and the asymmetry is
architectural rather than a gap in our measurement. Section 8 reports
Jev's negative log likelihood as 1.133 and says most of it is the clip:
10 claims carry a reported probability at the boundary on the wrong side of the
gold label, each charged 34.5 nats at the constant
`configs/benchmark.yaml` marks provisional. tab:logloss-eps re-scores the
identical predictions at other constants. A constant one-half forecast scores
0.693, so the reported figure exceeds that reference only for constants below
2.9 10^-6, which is finer than
anything this vendor reports, p_supported arriving at
2 decimals with 86 literal zeros and 26 literal
ones, which is why Section 8 reads the verdict off the bounded scores
instead.

[t]

's negative log likelihood over clipping constants, same predictions and same
labels each time. The first row is the configured value and the figure
Section 8 reports; the last is half the step the vendor reports the
probability on, which is the coarsest constant that still charges a reported zero more
than a reported 0.01. The reference a constant one-half forecast scores is
0.693.
:logloss-eps

Clipping constant & Jev NLL \\

1.0 10^-15 (configured) & 1.133 \\
1.0 10^-6 & 0.715 \\
1.0 10^-3$ & 0.575 \\
0.005 (half the reported step) & 0.544 \\

tabular
table

### No example was excluded for length, and nothing was truncated

The longest input is 12,977 tokens for the decision model and
12,359 for the frontier arm, well inside both documented context limits, and
all 495 rows of all four prediction files carry a successful status. Nothing was dropped
for exceeding a context window and no evidence text was shortened to fit one. That also bounds
what the sample can say: a set of claims that never approaches either limit cannot tell a
reader how either system behaves on the long documents that would.

### The harness was built with AI coding assistance

Claude Code wrote most of the analysis and adapter code under human direction. What stands
behind the numbers is this. The metric layer is a second implementation:
`src/metrics.py` imports no scikit-learn and computes every metric directly, and
`tests/test_metrics_crosscheck.py` recomputes each with scikit-learn on randomised
fixtures and requires agreement to `1e-12`, negative log loss excepted at
`1e-9` because the two clip differently, so a bug would have to exist identically in
two implementations to survive. Expected calibration error has no library equivalent and is
covered by unit tests on constructed cases. The array shortcuts inside the bootstrap and
cross-fitting loops are asserted equal to that metric code before any result is reported.
Each prediction file was committed when its run finished and has not changed since: the
analysis re-derives every input's sha256 on every run and aborts rather than rewrite the
stamped evidence lock, so an input edited after the lock fails the run instead of quietly
moving a number. What we do not claim is a second implementation of the pipeline above the
metrics: one script turns the locked predictions into every number here.

### The TypeSafe record, in order

Our protocol cited a clause restricting publication of benchmark results. It is not in the
live agreement. The citation traces to a deep-research summary in this project's research
directory, which the project's own README marks as not a source of truth. The founder decision
dated 2026-09-18, in the released package, accepted a breach risk on that premise and now
carries a correction header, because the premise did not hold. On 2026-09-20 we fetched the
live agreement, captured it, hashed the capture and searched it: the words
`benchmark` and `publish` do not occur, and the subsection our protocol cited,
2.3(f), is about interfering with the operation of the service. The decision that followed is
Alden's of 2026-09-21, recorded in the package's permission README: publish, and request no
permission. The request drafted under the old premise was never sent and ships headed with
the reason it must not be sent as written. The hash covers our own capture of a page that had
already changed version line between the protocol's reading and ours, so it fixes the bytes
our excerpt came from and nothing else; a reader who wants to check the finding fetches the
clause rather than matching a digest. Neither document is a statement by the vendor.

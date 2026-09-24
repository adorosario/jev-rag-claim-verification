
## 2026-09-21 — gpt-6-astra paired dev run: 84/495 failures caused by our token cap

- **Run:** `paired-astra-vs-jev-20260921` (exploratory, dev split, not a gated run).
- **Symptom:** 80 `format_error` ("empty response") and 4 `provider_error` (HTTP 400,
  "Could not finish the message because max_tokens or model output limit was reached").
- **Root cause:** ours, not the model's. `max_completion_tokens=16` (configs/models.yaml)
  is consumed by the model's internal reasoning tokens, leaving no budget for the
  answer. Every one of the 80 format errors reported exactly `output_tokens: 16`;
  successful calls used 4-5.
- **Fix:** re-run with `max_completion_tokens=256` as run
  `paired-astra-vs-jev-20260921-v2`. The v1 rows are kept for the record.
- **Reporting rule:** the v1 failure rate must NOT be published as an Astra defect.
  It is a harness configuration error (spec §17, §43).

## 2026-09-23: the stamped evidence file was rewritten after it was locked

**What happened.** A D3 repair round edited `scripts/verify/paired_analysis.py` to hash two
extra inputs into `arxiv/evidence/raw_hashes.json`. That file is covered by an
OpenTimestamps proof and by the pushed tag `results-lock-arxiv-v1`, so rewriting it made
`ots verify` report `File does not match original!` for anyone checking our provenance
claim. The paper meanwhile told readers the tag proved the predictions were fixed.

**Impact on results: none.** The `prediction_files` block was byte-identical across both
versions (`git diff results-lock-arxiv-v1 -- arxiv/evidence/raw_hashes.json` touches only
the `configs` block; the sorted prediction block hashes to
`cd3c9e90a4a6d9e32504d9e1f3b74db3a4eaa0fc603dccd705f721eb336face8` in each). No prediction
changed, so no number was regenerated.

**Fix.** The stamped file was restored from the tag and `ots verify` passes again. The two
extra hashes moved to `arxiv/evidence/input_hashes.json`, which is deliberately not
stamped. `write_evidence()` now refuses to rewrite the stamped file at all and says why,
so the next agent that tries gets an error instead of a broken proof. The refusal was
tested by planting a tampered file.

**Caught by.** The D3 `repro` verifier, by running `ots verify` rather than trusting the
sentence in the draft that said the lock held.

## 2026-09-23: the cascade's escalation rule was not the one the paper stated

**What happened.** The paper states the gate as "when the confidence $\max(p, 1-p)$ clears
the threshold, Jev's verdict is the pipeline's verdict". `cascade_conf()` implemented the
algebraically identical `(p >= t) | (p <= 1 - t)`. The two are not identical in binary
floating point: `1 - 0.8` is `0.19999999999999996`, so the single prediction in this sample
with `p_supported` exactly `0.2` failed `p <= 1 - t`, was escalated by the code, and would
have been kept by the stated rule. A reader reimplementing the paper's sentence therefore
could not reproduce the paper's escalation rate or its cascade cost. Flagged by a D1
verifier, restated by a D3 verifier, closed here.

**Impact on results: small and confined to the escalation shares.** The paper's rule is the
one a reimplementer would write, so the code was changed to follow the paper rather than the
paper changed to follow the code, and every number was regenerated from the unchanged locked
predictions. The headline did not move: cross-fitted cascade macro BAcc 74.10, cost
$2.92/1k, selection optimism 1.72pp, cascade minus Astra 0.27pp [-3.85, +4.15] are all
unchanged. What moved: in-sample escalation 17.4% to 17.2% and its cost $1.63 to $1.62, the
t=0.90 sweep row 31.1% to 30.9% and $2.95 to $2.94, mean cross-fitted escalation 30.0% to
29.9%, the matched random-escalation control's budgets and interval by up to 0.1pp, and two
gating rows (coverage at t=0.80 82.6% to 82.8%, at t=0.90 68.9% to 69.1%, with BAcc at
t=0.90 76.2 to 75.8). No prediction file was touched and `arxiv/evidence/raw_hashes.json`
is unchanged, so the OpenTimestamps proof and the `results-lock-arxiv-v1` tag still hold.

**Fix.** `cascade_conf()` now computes `np.maximum(p, 1 - p) >= t`, with the reason in its
docstring so nobody "simplifies" it back.

**Caught by.** The D1 `repro` verifier, by reimplementing the rule as written instead of
reading the code.

## 2026-09-24: the log-loss clipping constant was described as preregistered, and it is not

**What happened.** `src/metrics.py` carried, in the comment over `DEFAULT_LOGLOSS_EPS`,
"The VALUE is a protocol decision preregistered at G2A". `configs/benchmark.yaml` carries
the same constant annotated `# PROVISIONAL: preregister the value at G2A`, and G2A never
ran for this study. The comment ships in the public export, so a reader checking how the
log loss is computed was told the constant had been fixed in advance when it had not.
The constant is load-bearing: 62% of Jev's reported negative log likelihood of 1.133 is
ten claims charged `-log(eps)` at `eps = 1e-15`, and Section 8 compared that figure with
the 0.693 a constant one-half forecast scores. That comparison reverses at any clip above
2.9e-6, while the vendor reports the probability to two decimal places, so the sentence
was a statement about our constant as much as about the model.

**Impact on results: none, and one sentence retracted.** No prediction and no metric
changed. The comparison against the constant one-half forecast is gone from Section 8,
which now prints the resolution the probability arrives at, says the constant is
provisional, and gives the figure at a clip the size of that resolution (0.544).
Appendix E re-scores the same predictions over the whole grid and prints the clip at
which the comparison reverses.

**Fix.** The comment now says what is true and points at the sensitivity.
`negative_log_loss_eps_sensitivity()` and `probability_resolution()` compute both
quantities into `numbers.json`, with unit tests, so the paper cites measurements rather
than adjectives. `test_the_log_loss_verdict_is_scoped_to_the_constant_that_decides_it`
fails if any unqualified form of the comparison returns, and fails on the arithmetic if
the crossing ever moves above the vendor's reporting resolution.

**Caught by.** A D3 verifier, by recomputing the log loss at other clipping constants
from the `logloss_clip` decomposition already in `numbers.json`.

## 2026-09-21: the frontier prompt was missing a trigger the decision model's prompt had

**What happened.** A pre-publication audit of our own prompts found that the decision
model's criteria named wrong entities as grounds for NOT_SUPPORTED while the frontier
template listed only numeric and temporal errors. Entity-swap hallucinations are a large
share of the AggreFact negatives, so the asymmetry favoured the decision model.

**Impact.** We added the missing trigger to the frontier template and re-ran both frontier
configurations. Every frontier number in the paper comes from the corrected runs. On the
low-effort arm the correction moved 9 of 495 verdicts and macro balanced accuracy from
73.06 to 73.83, which reverses the sign of the headline comparison against the decision
model's 73.27. The paper states that in Appendix E rather than leaving it to be noticed.

On the high-effort arm it moved 5 verdicts and macro balanced accuracy from 73.93 to
74.11. One of those five landed on a gold-unsupported claim, taking that arm's
false-verification rate from 12.63% to 13.13%. That rate is one leg of the cascade's guard
penalty against the high-effort arm, which is one of the two comparisons in the paper that
survive the Holm adjustment, so the paper recomputes that comparison against the
superseded run rather than asserting the correction left it alone: the penalty is 6.60
points against the superseded run and 6.09 reported, and all 200 cross-fitting repeats
separate on either. For several drafts Appendix E reported only the low-effort arm and
generalised its result, which was false on the arm it did not measure.

**Fix.** Both prompts ship verbatim in the paper's Appendix B, and all four runs, both
pre-correction and both corrected, are released so the change is auditable. The two
superseded runs are hashed in `arxiv/evidence/input_hashes.json`.

**Caught by.** Our own prompt audit, before publication.

## 2026-09-22: the MiniCheck arm was run, measured and excluded

**What happened.** MiniCheck-Flan-T5-L was ported as a third arm and as the external
harness-validity anchor: a public model with a published score, pushed through our loader,
gold-label join and metric code, so that a large shortfall would implicate those components
rather than the model. It scored 61.58 macro balanced accuracy against the published
test-split 75.0.

**Impact.** The arm is excluded from every comparison. Sweeping every available cut point
does not rescue it, so the exclusion is not a threshold artefact. The shortfall implicates
our adapter, most likely its chunking and evidence handling, and we therefore do not claim
the harness is sound; we claim only that the metric layer and the gold join are shared with
the reported arms and would have moved them too.

**Fix.** None available from this data. The arm is reported as run, measured and dropped,
with its predictions released so a reader can diagnose the port.

**Caught by.** The anchor failing its own target.

## 2026-09-24: the cascade operating point we published on Medium is superseded by the paper

**What happened.** The companion article (`medium/article-v3.md`, link in
`medium/story-link.md`) was published under the first author's byline before the preprint.
Its cascade figure is the in-sample threshold sweep, and the row it leans on is the best row
of that sweep: 75.8 macro balanced accuracy at 17% escalation for $1.63 per 1,000 claims
(now $1.62 after the escalation-rule fix logged above). The preprint reports the
cross-fitted operating point instead, 74.10 at $2.92 per 1,000, and Section 10 of the
preprint criticises exactly the practice of sweeping a threshold and quoting the best row,
without, in its first draft, saying that the authors had done it in public first.

**Impact on results: none, and a disclosure added.** No number in the paper comes from the
article. What was missing was the disclosure: Appendix C claimed to hold the same set of
entries as this log while neither document mentioned that the retracted operating point is
already in circulation under the authors' names. Appendix C now carries the entry, this is
its counterpart, and the two are the same set (eight, with the release defect logged
below).

**Also true of the published article, and not corrected there.** It predates the
escalation-rule fix, so its t=0.90 sweep row reads 31.1% and $2.95 where this log records
30.9% and $2.94, and its in-sample row reads $1.63 where the corrected value is $1.62. The
article is released as published rather than edited after the fact; the paper is the
corrected document and says so.

**Caught by.** A D3 verifier, by reading the published article against the paper's own
Section 10.

## 2026-09-24: the published package carried the superseded registry, and the release check answered PASS

**What happened.** `configs/models.yaml` was corrected at 17:01:55 UTC on 2026-09-24 to
carry both completion-token caps per reasoning effort, 256 and 4096, taken from the
reported runs' manifests, plus a note naming the token-cap incident at the top of this log
and a line recording that the high-effort configuration of the older model never became an
arm. The public reproduction package at `github.com/adorosario/jev-rag-claim-verification`
had last been built and pushed at 15:44 UTC, 77 minutes before that correction. From then
until the next push, a reader who followed Appendix D of the preprint to the pinned model
registry read `max_completion_tokens: 16` with no note: the cap of the discarded first pass
that the first entry of this log records as the cause of 84 failures in 495 answers.

Three further artefacts in that same package were the pre-correction copies.
`scripts/verify/check_public_release.sh`, which Appendix D offers as the reader-side check
of the package's inventory, was published in its earlier form, which fetched each path and
compared only the HTTP status; a stale export answers 200 on every path, so it printed
`PASS: all 45 artefacts the paper names are public` against the superseded registry it was
supposed to catch. `tests/test_paper_evidence.py` was the superseded copy, and
`tests/test_model_registry.py`, the guard added with the registry correction, was absent
from the package, which made it one of the paths the gate's own list names and a reader
cannot fetch.

**Impact on results: none. Impact on the release: three sentences of the paper were false
at the reader's end.** No number moves: the caps are documentation of runs that were
locked and hashed before the correction, and the correction changed no prediction. What was
wrong was the package the paper sends a reader to, which is the artefact the paper's
reproducibility claim rests on.

**Fix.** Two, and only the first is code. The reader-side check now compares each published
file by sha256 against the copy in the checkout it is run from, so a stale export fails
instead of reporting a status, and the gate carries this failure mode in its own header
comment. Second, the package is rebuilt with `scripts/build_public_export.sh` and
republished, and that check has to exit 0 before the preprint is submitted; no agent
performs the push (CLAUDE.md rule 12). The check is the release gate rather than a
formality: it is what turned a PASS into a FAIL here.

**Caught by.** A D3 verifier fetching the published files themselves rather than reading
our export log, twice, over two review rounds.

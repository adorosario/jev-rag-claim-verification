# Jev-as-a-Judge for RAG Claim Verification

*We gave Jev and GPT-6 Astra the same 495 labelled claims from a public
grounded-factuality benchmark. The accuracy difference was half a point. The cost
difference was 187×.*

---

## Key takeaways

- **On identical inputs, Jev and GPT-6 Astra were statistically indistinguishable**
  on grounded claim verification: 73.3 vs 73.8 balanced accuracy, a difference of
  0.6 points with a 95% confidence interval of [−5.6, +4.5] and McNemar p = 0.49.
  Astra is nominally ahead; the sample cannot resolve a gap this small.
- **Jev cost $0.048 per 1,000 claim checks. Astra cost $8.93.** That is 187×, from
  measured token counts at published standard rates, not list-price arithmetic.
- **Letting Astra think harder did not change the picture.** At maximum reasoning
  effort it reached 74.1 — 0.8 points better than its cheap configuration — while
  costing $10.70 per 1,000. The frontier model was not handicapped.
- **Jev is clearly worse at the thing verification exists to catch.** It wrongly
  marked 23.2% of unsupported claims as supported, against Astra's 14.1%. Used alone
  and unfiltered, it is a permissive verifier.
- **Its confidence score is what makes that fixable.** Above 0.99 it verified claims
  at 97.1% precision on 35% of the traffic. Escalating the 31% it was least sure
  about scored 74.0 at $2.95 per 1,000 — Astra's accuracy at a third of Astra's cost.

---

## Why we ran this

Last week [LangChain published "Jev-as-a-Judge for Agent Evals"](https://www.langchain.com/blog/jev-agent-evals-langsmith),
asking whether TypeSafe's "System One" model could replace an LLM judge for scoring
agent traces. Their headline result: Jev matched their human oracle on all 500
repeated decisions, with quality-score variance 92–913× lower than GPT-5.6 Luna,
Terra and Claude Sonnet 4.6, at $0.00035 per call.

That study answered a precise question: **is Jev a consistent judge?** It replayed
five captured weather-agent runs 100 times each and measured how far the verdicts
drifted.

It deliberately left the neighbouring question open. In their words:

> Lower variance does not automatically mean higher accuracy: a judge can still be
> consistently wrong.

So we ran the other half: **is Jev a correct judge, on a task where the right answers
were published before anyone thought to ask?**

Not five items replayed a hundred times. 495 different claims, each with a
human-assigned gold label, from a peer-reviewed benchmark.

---

## The task

Every RAG system runs one decision over and over:

```
claim + evidence  ->  does the evidence support the claim?
```

It is the judge inside hallucination detection, citation checking and response
verification. In most production systems it is an LLM call, it runs once per claim,
and a single answer contains many claims. It is the most expensive cheap decision in
the stack.

It also has a real public benchmark, which is the point — nobody has to trust our
labels.

## Setup

**Benchmark.** [LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact),
introduced with [MiniCheck](https://arxiv.org/abs/2404.10774) (Tang, Laban and
Durrett, EMNLP 2024): 11 grounded-factuality datasets — news summarisation, meeting
notes, Wikipedia, long-form QA, RAG outputs — unified into one binary task with human
labels.

**Sample.** 495 dev-split examples, 45 from each of the 11 component datasets. Both
models answered **exactly the same 495 items**, which is the only reason a
half-point difference is worth discussing at this sample size.

**Models.** `jev-1.13.0`, pinned, through TypeSafe's API. `gpt-6-astra` through
OpenAI's API, run twice: a task-optimised configuration (`reasoning_effort=low`) and
maximum reasoning (`reasoning_effort=high`). Standard service tiers, no tools, no
retrieval, no chain-of-thought requested, no self-consistency.

**Prompts.** Jev got one `Choice` question with two criteria. Astra got the same
instruction as text with an enum answer. We caught one asymmetry before running the
final comparison — Jev's criteria named wrong entities as grounds for "not supported"
and Astra's did not — and fixed it, because entity swaps dominate two of the eleven
datasets and the imbalance favoured Jev. Neither prompt was tuned against the scores.

**What we did not do.** No test-split evaluation, no threshold fitting on held-out
data, no prompt iteration against these results.

---

## Results

### Accuracy

Balanced accuracy per dataset, averaged across the 11 — the LLM-AggreFact
convention, chosen because the benchmark uses it, not after seeing our numbers.
Intervals are 10,000-sample stratified paired bootstraps.

| Model | Balanced accuracy | 95% CI | False-verification rate | Precision of "supported" | $ / 1,000 claims |
|---|---|---|---|---|---|
| GPT-6 Astra (high reasoning) | 74.1 | — | **13.1%** | **89.3%** | $10.70 |
| GPT-6 Astra (task-optimised) | 73.8 | [68.8, 78.8] | 14.1% | 88.6% | $8.93 |
| **Jev 1.13.0** | **73.3** | [68.5, 77.9] | 23.2% | 84.1% | **$0.048** |

The paired difference between Jev and task-optimised Astra is **−0.6 points**, 95% CI
**[−5.6, +4.5]**. An exact McNemar test on the paired per-item errors gives
**p = 0.49**; note that McNemar tests raw per-item accuracy rather than the
macro-averaged balanced accuracy in the headline, so the two answer slightly
different questions, and here they point in opposite directions. We say
indistinguishable rather than
equivalent: the interval is wide enough that a real gap of several points either way
would look like this.

**One honest complication.** The ranking flips with the metric. On raw accuracy Jev
leads, 79.8 to 78.4. Balanced accuracy weights the rare negative class equally, and
Jev over-predicts "supported" on a sample that is 60% positive, so raw accuracy
rewards it. We report balanced accuracy as the headline because that is what the
benchmark's leaderboard uses — but a reader who prefers raw accuracy gets the
opposite winner, and should know that.

The two models also disagree more than the aggregate suggests: they agreed on 84.8%
of individual claims, with Jev right where Astra was wrong 41 times and Astra right
where Jev was wrong 34 times. Equal average accuracy does not mean interchangeable.

### Cost

Measured from actual token usage over these 495 calls, priced from provider pages
captured and hashed on the day of the run.

| | $ / 1,000 claims | mean input tokens | mean output tokens |
|---|---|---|---|
| **Jev 1.13.0** | **$0.048** | 1,139 | 0 (billed) |
| GPT-6 Astra (task-optimised) | $8.93 | 843 | 11 |
| GPT-6 Astra (high reasoning) | $10.70 | 843 | 46 |

**187× cheaper** for accuracy we cannot distinguish from the frontier model's. Jev
bills input only; output is free, which is the whole economic story of a model that
returns a typed answer instead of writing one.

Note that Jev's own tokenizer counted *more* input tokens for the same text — 1,139
against Astra's 843. The 187× is computed with that handicap included.

### Latency

Both runs used concurrency 12, so their timings are comparable to each other, but
neither is a controlled single-request measurement: each number includes
client-side queueing. Median per-call latency was **214 ms for Jev** and
**1,478 ms for Astra** (2,133 ms at high reasoning). Treat this as the shape of the
difference, not a benchmark figure.

### Where Jev is worse, and why it matters

Balanced accuracy hides an asymmetry that verification cares about intensely. A
verifier that waves a false claim through does more damage than one that flags a true
one, and Jev waves through **23.2%** of unsupported claims against Astra's **14.1%**.

The per-dataset view shows where that comes from:

| Dataset | Jev | Astra (task-optimised) |
|---|---|---|
| RAGTruth | 92.7 | 87.8 |
| Wice | 87.9 | 80.7 |
| ClaimVerify | 84.8 | 85.6 |
| Lfqa | 84.2 | 81.8 |
| TofuEval-MeetB | 80.5 | 79.1 |
| Reveal | 72.9 | 80.5 |
| AggreFact-XSum | 67.6 | 63.0 |
| FactCheck-GPT | 64.8 | 67.5 |
| TofuEval-MediaS | 61.5 | 71.2 |
| ExpertQA | 60.5 | 46.1 |
| **AggreFact-CNN** | **48.7** | 68.8 |

Jev takes 6 of 11 and the two trade the rest, which is what a half-point aggregate
gap should look like. The one worth singling out is RAGTruth, built from
retrieval-augmented generation outputs and therefore the closest of the eleven to a
production RAG workload: Jev leads there by 5 points, its best dataset of the set.
That is a single 45-example slice, so it is a hint rather than a finding, but it is
the hint most relevant to anyone deploying this.

Then Jev collapses on AggreFact-CNN, at 48.7: a coin flip. It answered "supported"
on nearly every example there. AggreFact-CNN packs a three-sentence summary into a
single claim, and TypeSafe's documentation warns that Jev degrades when several
judgments hide inside one question. The obvious fix — split multi-sentence claims
into atomic ones before asking — is untested here, so that is a diagnosis with a
plausible cause and no cure yet. With only 7 unsupported examples in that dataset,
the 50.0 is itself a noisy estimate.

### The confidence score is the useful part

Jev returns a probability with every decision, natively, rather than being asked to
rate its own confidence in prose. Ours was usable as a routing signal:

| Keep decisions above | Coverage | Balanced accuracy | Precision of "supported" |
|---|---|---|---|
| 0.50 (everything) | 100% | 73.3 | 84.1% |
| 0.90 | 69% | 76.2 | 88.1% |
| 0.95 | 56% | 80.4 | 91.3% |
| 0.99 | 35% | 87.9 | **97.1%** |

On a third of the traffic, at five cents per thousand checks, this model verifies
claims at 97% precision.

Which makes the interesting architecture a cascade, not a replacement. Let Jev decide
when it is confident; send the rest to the frontier model:

| Escalate when Jev is below | Escalated | Balanced accuracy | $ / 1,000 |
|---|---|---|---|
| — (Jev alone) | 0% | 73.3 | $0.05 |
| 0.90 | 31% | 74.0 | $2.95 |
| 0.95 | 44% | 74.5 | $4.56 |
| 0.99 | 65% | 75.3 | $6.43 |
| — (Astra alone) | 100% | 73.8 | $8.93 |

Escalating under a third of claims matches Astra's accuracy at under a third of its
cost, and escalating two thirds beats it outright. That falls out of the confidence
score, not out of clever engineering. It is the configuration we would put behind a
[response-verification pipeline](https://customgpt.ai/platform/verify-responses/).

One caution: these thresholds were read off the same 495 examples they are scored on.
The shape is real; the exact numbers are optimistic until someone picks the threshold
on one split and tests it on another.

---

## What this does not show

- **One frontier model, one benchmark, 495 examples.** Not the full LLM-AggreFact
  test set, not Claude, not Gemini, not your data.
- **Dev split only.** We did not touch the held-out test split, so these numbers stay
  comparable with future work instead of burning the holdout on a blog post.
- **45 examples per dataset**, with as few as 4 negatives in one. Per-dataset rows
  move several points on one flipped answer; the aggregate is the trustworthy part.
- **A stratified sample, not a random draw** from dev.
- **The benchmark is public and predates both models**, so contamination is possible
  for either.
- **Latency is not a controlled measurement** — see above.
- **Thresholds were selected in-sample.**
- **We are not neutral.** We build RAG software and this decision is our production
  workload. That is exactly why the raw predictions are published.

And one about process. Our first Astra run failed on 84 of 495 examples, and it would
have been easy to write "the frontier model is unreliable". It was our bug: a
16-token answer budget that the model's own reasoning tokens consumed before it could
answer. The API said so in plain English. We raised the budget, re-ran, got 495 of
495 clean, and logged the whole thing. If you benchmark someone else's model, assume
the first anomaly is yours.

---

## What we think it means

Grounded claim verification is a decision, not a composition. The output is one bit
and a confidence. Producing it by generating text and parsing the text back into a
bit is, in hindsight, an odd way to spend $8.93 per thousand.

LangChain found a System One model was a dramatically more *consistent* judge than an
LLM. On a public benchmark with published answers, we find it is also — within the
resolution of 495 examples — an equally *accurate* one for this task, while being
distinctly worse at the failure mode verification exists to catch, and carrying a
probability that makes that weakness routable.

The honest summary is not "Jev beats frontier models". It is narrower and, we think,
more useful: for one high-volume decision at the centre of every RAG system, a model
costing 187× less was not measurably less accurate, and the confidence it returns
lets you spend the frontier model's budget only where it earns its price.

---

## Reproducibility

Everything is public: the raw predictions for all three runs with their manifests,
the prompts, the model registry, the hashed pricing snapshots, the analysis script
and the numbers file.

Every figure above is generated from `medium/generated/numbers.json`, produced by
`scripts/verify/paired_analysis.py` from the raw predictions, and a checker script
verifies that each number in this article matches that file. None was typed by hand.

- **Models:** `jev-1.13.0` (pinned), `gpt-6-astra`, standard tiers, 21 September 2026
- **Data:** `lytang/LLM-AggreFact`, dev split, 495 stratified examples, IDs published
- **Statistics:** stratified paired bootstrap, 10,000 resamples, seed 20260918;
  exact McNemar on paired errors

*Thanks to the LangChain team, whose experiment prompted this one.*

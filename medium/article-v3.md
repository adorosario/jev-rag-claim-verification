# Jev-as-a-Judge for RAG Claim Verification

### I ran 495 labelled claims through a five-cents-per-thousand decision model and a frontier LLM. Accuracy came out statistically tied, cost came out 187x apart, and the cheap one waved through 23% of the unsupported claims it was hired to catch (the expensive one waved through 14%).

---

A few days ago, before Jev was released, I wrote this on LinkedIn:

> Everytime I need a good LLM-As-A-Judge, I cringe that I have to build it using a
> large LLM, instead of just buying a trustworthy one from a trusted source.

That cringe has a very specific source. Somewhere inside every RAG pipeline there is ONE
question that runs millions of times a month:

**does this evidence actually support this claim?**

Yes or no. That is the whole job. And the way we answer it today is by calling a giant
language model that writes its answer out one token at a time so that our code can read
the text back and turn it into a boolean, which is a bit like hiring a novelist to
operate a light switch.

Then [LangChain published a Jev evaluation](https://www.langchain.com/blog/jev-agent-evals-langsmith)
last week, and TypeSafe's new "System One" model started looking like the thing I had
been cringing for. So I spent a day pointing it at the decision that costs us the most
money: grounded claim verification, on a public benchmark whose answers were published
long before either model existed.

What came back is messier than "cheap model wins" and messier than "you get what you
pay for". Which is annoying, because the messy answer is the one you can actually build
on.

## What a "System One" model actually is

Thirty seconds on this, because the category name does more harm than good.

A normal LLM answers a classification question by writing. You ask "supported or not
supported?", it emits tokens that happen to spell a word, and your code maps that word to
a label. The decision falls out of the writing, as a side effect.

Jev does not write. You hand it a state (here: a claim plus its evidence) and a typed
question (here: a two-way choice), and you get back a value with a probability on it. No
parsing. No "respond ONLY in valid JSON". No retry loop for when the model decides to
apologize first, right? That is the part that made me pay attention.

Which has an economic consequence: Jev bills input tokens only, because there is
basically nothing to bill on the output side.

> The open question is whether skipping the writing costs you the thinking.

## What LangChain measured, and what they did not

Their headline: Jev matched their human oracle on all 500 repeated decisions, with
quality-score variance 92-913x lower than GPT-5.6 Luna, Terra and Claude Sonnet 4.6,
at $0.00035 per call.

Good experiment. It answered a precise question, which is whether Jev is a CONSISTENT
judge. They replayed five captured weather-agent runs 100 times each and watched how far
the verdicts drifted.

And to their credit, they said out loud what it did not cover:

> Lower variance does not automatically mean higher accuracy: a judge can still be
> consistently wrong.

So I went and ran the other half. Not five items replayed a hundred times, but 495
different claims, each one carrying a human-assigned gold label from a peer-reviewed
benchmark. Consistency is one virtue. Being RIGHT is a completely different one .. and
we keep letting those two get mixed up.

## The setup: 495 claims, 11 datasets, 2 judges, same items

**The data.** [LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact),
introduced with [MiniCheck](https://arxiv.org/abs/2404.10774) (Tang, Laban & Durrett,
EMNLP 2024). It folds 11 grounded-factuality datasets (news summaries, meeting notes,
Wikipedia, long-form QA, RAG outputs) into one binary task with human labels. Public
benchmark, so nobody has to trust my labels, and I did not get to pick the easy cases.

**The sample.** 495 dev-split examples, 45 from each of the 11 datasets. Both models
answered exactly the same 495 items, which is the only reason a half-point difference is
even discussable .. we are comparing two judges on identical work, instead of comparing
two separate benchmark runs and hoping.

**The models.** `jev-1.13.0`, pinned, through TypeSafe's API. `gpt-6-astra` through
OpenAI's API, run twice: once task-optimised (`reasoning_effort=low`, which is how any
sane engineer deploys a binary classifier) and once at maximum reasoning, so that nobody
gets to say I crippled the expensive model to make a point.

**The metric.** Balanced accuracy per dataset, then averaged. That is the LLM-AggreFact
convention, and it weights the rare "not supported" class equally with the common one, so
a judge cannot score well by cheerfully agreeing with everything.

## Accuracy: a statistical tie

![Comparison table: GPT-6 Astra high reasoning 74.1 balanced accuracy, 13.1% false-verification rate, $10.70 per 1,000 claims; GPT-6 Astra task-optimised 73.8, 14.1%, $8.93; Jev 1.13.0 73.3, 23.2%, $0.048.](figures/headline.png)

*Both models answered exactly the same 495 claims. Jev trails by 0.6 points and costs 187x less.*

The paired difference is 0.6 points in Astra's favour, 95% CI [−5.6, +4.5]. Exact
McNemar on the paired per-item errors gives p = 0.49. (Careful with that one: McNemar is
testing raw per-item accuracy, not the macro-averaged balanced accuracy in the headline,
so the two are answering slightly different questions .. and on this run they point in
opposite directions.)

Indistinguishable does not mean equivalent. That interval is wide enough that a real gap
of several points in either direction would look exactly like what I measured.

**And the winner flips with the metric.** On raw accuracy Jev leads, 79.8 to 78.4,
because it over-predicts "supported" on a sample that is 60% positive and raw accuracy
rewards exactly that behaviour. I lead with balanced accuracy because the benchmark does.
But if raw accuracy is your thing, you get the opposite winner, and you should hear that
from me rather than from some guy in the comments.

The two judges also disagree a LOT more than the aggregate suggests. They agreed on
84.8% of individual claims. Jev was right where Astra was wrong 41 times, Astra was right
where Jev was wrong 34 times. Same average, very different judges. (Hold that thought, it
pays off later.)

## Cost: this one is not close

![Cost chart: Jev $0.048 per 1,000 claim checks against GPT-6 Astra at $8.93 task-optimised and $10.70 at high reasoning, a 187x difference.](figures/cost.png)

*Costs come from the token counts these 495 calls actually used.*

187x cheaper, at an accuracy I cannot distinguish from the frontier model's.

Two things make that number more credible rather than less, and the first is that it
comes from the token counts these 495 calls actually burned, priced against provider
pages that were captured and hashed on the day of the run, instead of me multiplying a
pricing page by a guess. The second is that Jev's own tokenizer counted MORE input tokens
for the same text (1,139 against Astra's 843), so the 187x is already carrying that
handicap on its back.

At a million claim checks a month, which is not a big number for a production RAG system
doing per-claim verification:

- Jev: about $48
- GPT-6 Astra: about $8,900

Median latency was 214 ms for Jev and 1,478 ms for Astra (2,133 ms at high reasoning).
Both runs used concurrency 12, so those timings are comparable to each other but they are
not controlled single-request measurements, and I would not put them in a spreadsheet.

## Now the part I am not happy about

Balanced accuracy hides an asymmetry that verification cares about enormously. A verifier
that waves a false claim through does far more damage than one that flags a true claim
for a human to look at.

Jev waves through 23.2% of unsupported claims. Astra waves through 14.1%.

So if you drop Jev in alone and unfiltered as your hallucination guard, congratulations,
you now have a MORE permissive guard than the LLM you just replaced.

![Dot plot of balanced accuracy per dataset for both models. Jev leads on RAGTruth, Wice, Lfqa, TofuEval-MeetB, AggreFact-XSum and ExpertQA; Astra leads on ClaimVerify, Reveal, FactCheck-GPT, TofuEval-MediaS and AggreFact-CNN, where Jev scores 48.7.](figures/per-dataset.png)

*Jev leads on six of eleven datasets, including RAGTruth. And then falls over on AggreFact-CNN.*

Jev takes 6 of 11 and they trade the rest, which is roughly what a half-point aggregate
gap should look like.

The row I keep staring at is RAGTruth, which is built out of retrieval-augmented
generation outputs and is therefore the closest thing in this benchmark to a real
production RAG workload. Jev leads it by 5 points, its best dataset of the set.

And then I went and looked at what that lead is actually made of, which is where it gets
less flattering. RAGTruth contributed 4 unsupported claims to my sample, and both models
got all 4 right. So the entire gap is Jev catching *supported* claims that Astra missed,
which means this particular row says nothing at all about the permissiveness problem
above. Treat it as a hint about the workload, and not as evidence about verification
quality.

Then there is AggreFact-CNN, where Jev scores 48.7. That is a coin flip. It said
"supported" to nearly everything in that dataset.

AggreFact-CNN packs a three-sentence summary into a single claim, and TypeSafe's own docs
warn that Jev degrades when several judgments are hiding inside one question. The obvious
fix is to decompose multi-sentence claims into atomic ones before you ask, and I have not
tested that yet, so file it as a diagnosis with a plausible cause and no cure. With only
7 unsupported examples in that dataset, the 48.7 is noisy anyway.

## The confidence score is the actual product

Here is where a decision model earns its keep. Jev returns a probability with every
answer, natively. That is a very different animal from asking a language model how
confident it feels about the thing it just finished writing.

![Confidence gating table: keeping decisions above 0.90 covers 69% of claims at 88.1% precision; above 0.99 covers 35% at 97.1% precision.](figures/gating.png)

*Jev returns a probability with every answer, and it is usable as a routing signal.*

On a third of the traffic, at five cents per thousand checks, this thing verifies claims
at 97.1% precision.

One caveat belongs right next to that number instead of in a footnote. The 0.99 gate
keeps 174 of the 495 claims, and at that threshold two of the eleven datasets have no
examples left to score, so the figure is computed across 9 datasets rather than 11. It is
not measured on quite the same benchmark as the 73.3 headline. The 0.90 gate keeps all
eleven and needs no such asterisk.

The permissiveness problem does not disappear here. It becomes routable, which is a far
more useful property than either "good" or "bad".

Which is why the interesting architecture is a cascade instead of a replacement. Let Jev
decide when it is confident, and send the rest to the frontier model:

![Cascade table showing all seven thresholds swept: Jev alone escalates 0% of claims and scores 73.3 at $0.048; escalating 11% scores 74.8 at $1.00, 17% scores 75.8 at $1.63, 31% scores 74.0 at $2.95, 44% scores 74.5 at $4.56, 51% scores 74.8 at $5.21, 65% scores 75.3 at $6.43; Astra alone is 73.8 at $8.93.](figures/cascade.png)

*Escalating a tenth to a third of claims lands in the frontier model's accuracy band, at a fraction of its cost.*

I want to be careful about how hard I lean on any single row of that table, because the
accuracy column does not climb as you escalate more: it runs 73.3, 74.8, 75.8, 74.0,
74.5, 74.8, 75.3. Given the five-point confidence interval on the headline, those rows
are not distinguishable from each other, and picking the flattering one is exactly the
sort of thing I would rip apart in somebody else's benchmark.

So the honest version is a band and not a point. Escalating somewhere between a tenth and
a third of claims puts you in the same accuracy territory as the frontier model for
between $1 and $3 per thousand, against $8.93 for Astra alone. Where exactly you sit in
that band is a tuning decision for your own traffic and your own tolerance, not a finding
from 495 examples.

And remember those two judges disagree on 15% of claims .. the cascade works BECAUSE they
fail on different things, not because one is a cheap clone of the other.

That falls straight out of the confidence score. No clever engineering required.

## What I would actually do on Monday

Three things I would act on, and one I would not.

**Stop paying generation prices for decisions.** Anywhere your pipeline asks a yes/no or
a pick-one question at volume (grounding checks, routing, relevance filtering, safety
classification) is a place where a decision model belongs in the architecture
conversation. On this task, the accuracy tax was not measurable.

**Route on confidence, do not replace on faith.** Nobody should read this as "the cheap
model is just as good". Read it as: a cheap model that knows when it is unsure lets you
spend the expensive model's budget precisely where that budget earns its price.

**Measure false verification, not accuracy.** If I had published balanced accuracy alone,
Jev would look like a drop-in replacement, which it very much is not: 23.2% against
14.1% makes it nearly twice as permissive on unsupported claims, and for a verifier that
single number should outrank the headline.

**And what I would not do is generalise any of this.** One task, one benchmark, 495
examples, one frontier model, one day of work. A decision model that ties an LLM on
grounded claim verification tells you exactly nothing about open-ended reasoning,
multi-hop synthesis, or anything else that needs the model to actually write.

So, not an LLM killer. But for the highest-volume, lowest-glamour decision in the whole
stack, this is a genuinely useful new primitive .. and I did not expect to be writing
that sentence a week ago.

## Methodology and reproducibility

Everything is public at
[github.com/adorosario/jev-rag-claim-verification](https://github.com/adorosario/jev-rag-claim-verification):
raw predictions for all three runs with their manifests, both prompts, the model
registry, the hashed pricing snapshots, the analysis script, and the numbers file.

A fresh clone reproduces every figure here with two Docker commands, and a checker
script fails the build if any number in the prose drifts from `numbers.json`.

## Technical details

- **Dataset**: `lytang/LLM-AggreFact`, dev split, pinned revision, 495 stratified
  examples (45 per dataset), example IDs published
- **Models**: `jev-1.13.0` (pinned) and `gpt-6-astra`, standard service tiers,
  21 September 2026
- **Configs**: no tools, no retrieval, no chain-of-thought, no self-consistency;
  `reasoning_effort` low and high for Astra, 256 and 4,096 completion-token budgets
- **Prompts**: Jev gets one typed `Choice` with two criteria; Astra gets the same
  instruction as text with an enum answer
- **Statistics**: stratified PAIRED bootstrap, 10,000 resamples, seed 20260918; exact
  McNemar on paired per-item errors
- **Costs**: measured token counts x provider list prices captured and hashed on the
  run date, uncached, standard tier

One methodology note that deserves your attention. Jev's criteria originally named wrong
entities as grounds for "not supported" and the frontier prompt did not, which quietly
favoured Jev (entity swaps are most of what two of these datasets contain). I caught it
before the final runs, fixed it, and re-ran everything.

And now the embarrassing one. My first Astra run failed on 84 of 495 examples, and it
would have been very easy to write "look at this, the frontier model is unreliable". It
was my bug. A 16-token answer budget that the model's own reasoning tokens ate before it
could answer, and the API said so in plain English, and I had to go actually read it.
Raised the budget, re-ran, 495 of 495 clean, logged the whole thing in the repo. If you
are benchmarking somebody else's model, assume the first anomaly is yours.

## Limitations

- **Sample size**: 495 examples, 45 per dataset, with as few as 4 negatives in one of
  them. Per-dataset rows move several points on a single flipped answer. The aggregate
  is the trustworthy part, and even that carries a 5-point CI
- **One frontier model**: not Claude, not Gemini, not your data
- **Dev split only**: I did not touch the held-out test split, so these numbers stay
  comparable with future work instead of getting burned on a blog post
- **Stratified sample**, not a random draw from dev
- **Contamination is possible** for either model, since the benchmark is public and
  predates both
- **Latency is not a controlled measurement**: both runs used concurrency 12 and the
  numbers include client-side queueing
- **Cascade thresholds were selected in-sample**, so the shape is real and the exact
  numbers are optimistic. Somebody (probably me) should pick a threshold on one split
  and test it on another

## Disclosure

I am the Founder and CEO of [CustomGPT.ai](https://customgpt.ai), a RAG platform for
enterprises that care about governed AI. Grounded claim verification is the decision
sitting behind [Verify Responses](https://customgpt.ai/platform/verify-responses/), the
feature that checks each factual claim in an answer against the sources it came from and
shows customers the evidence.

So yes, I have a dog in this fight. Verifying claims at volume is one of the more
expensive things we run, which is exactly why a decision model at 1/187th the cost was
worth a day of measurement. We are evaluating Jev to augment that pipeline, with
confident decisions handled cheaply and instantly and the uncertain ones escalated to a
frontier model. The numbers above are why. And that 23.2% false-verification rate, the
number I am least happy about in this whole article, is why it will be a cascade rather
than a swap.

And on the TypeSafe side, since that is the relationship people will ask about: I am a
paying API customer, nothing more. No credits, no early access, no engineering help, and
no input from them into the design, the data or the results. They are reading this at the
same time you are.

*Alden Do Rosario is Founder & CEO of* [CustomGPT.ai](https://customgpt.ai). *The full
benchmark code, raw predictions and analysis are at*
[github.com/adorosario/jev-rag-claim-verification](https://github.com/adorosario/jev-rag-claim-verification). *Thanks to the LangChain team, whose experiment prompted this one.*

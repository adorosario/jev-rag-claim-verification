# Jev vs. GPT-6 Astra on grounded claim verification

A paired comparison of [Jev](https://typesafe.ai) (`jev-1.13.0`) and `gpt-6-astra`
on **495 labelled claims** from [LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact),
the grounded-factuality benchmark introduced with [MiniCheck](https://arxiv.org/abs/2404.10774)
(Tang, Laban & Durrett, EMNLP 2024).

Both models answered **exactly the same 495 items**, which is what makes a difference
this small worth discussing at all.

## Headline

| Model | Balanced accuracy | 95% CI | False-verification rate | $ / 1,000 claims |
|---|---|---|---|---|
| GPT-6 Astra (`reasoning_effort=high`) | 74.1 | — | 13.1% | $10.70 |
| GPT-6 Astra (task-optimised) | 73.8 | [68.8, 78.8] | 14.1% | $8.93 |
| Jev 1.13.0 | 73.3 | [68.5, 77.9] | 23.2% | $0.048 |

Difference between Jev and task-optimised Astra: **−0.6 points**, 95% CI
[−5.6, +4.5]; exact McNemar on paired per-item errors p = 0.49. Statistically
indistinguishable accuracy at **187× lower cost** — and a clearly higher
false-verification rate, which matters for a verifier.

Full write-up, including the confidence-gating and cascade results and the
limitations: [`medium/article.md`](medium/article.md).

## Reproduce

Everything below runs in Docker; no Python on the host.

```bash
cp .env.example .env          # HF_TOKEN is enough to re-analyse
docker compose build dev

# regenerate every published number from the raw predictions
docker compose run --rm dev python scripts/verify/paired_analysis.py

# check that every number in the article matches that file
docker compose run --rm dev python scripts/verify/check_article_numbers.py
```

`HF_TOKEN` must belong to an account that has accepted the LLM-AggreFact terms on
its Hugging Face page (the gold labels are joined at analysis time, never sent to
any model).

To re-run the models themselves, add `TYPESAFE_API_KEY` and `OPENAI_API_KEY` and:

```bash
docker compose run --rm dev python scripts/run_jev_paired.py
docker compose run --rm dev python scripts/run_astra_paired.py       # reasoning_effort=low
docker compose run --rm dev python scripts/run_astra_highthink.py    # reasoning_effort=high
```

## What's here

```
runs/paired-jev-20260921/                    Jev predictions + run manifest
runs/paired-astra-lowthink-20260921-p2b/     Astra, task-optimised
runs/paired-astra-highthink-20260921-p2b/    Astra, maximum reasoning
configs/prompts/                             both prompts, verbatim
configs/models.yaml                          model IDs and settings, as run
configs/pricing_2026-09-20.yaml              prices used, from the snapshots below
data/pricing_snapshots/                      hashed provider pricing pages
data/manifests/aggrefact_revision.json       pinned dataset revision + row hashes
src/                                          loader, adapters, metrics
scripts/verify/paired_analysis.py            produces medium/generated/numbers.json
scripts/verify/check_article_numbers.py      article <-> data consistency check
medium/article.md                            the write-up
medium/generated/numbers.json                every published number
```

Each prediction row carries the requested and reported model ID, the prompt hash,
token counts, latency, attempts and status. Each run directory carries a manifest
with the git commit, the dataset revision and hashes, the environment and the seed.

## Method notes

- **Metric:** balanced accuracy per dataset, averaged across the 11 — the
  LLM-AggreFact convention. Raw accuracy is also reported in `numbers.json`, and on
  this sample it favours Jev (79.8 vs 78.4), because Jev over-predicts "supported"
  on a 60%-positive sample. Both are published for exactly that reason.
- **Statistics:** 10,000-sample stratified *paired* bootstrap (the same resampled
  rows score both models), seed 20260918; exact McNemar on the paired errors.
- **Prompt parity:** an early asymmetry — Jev's criteria named wrong entities as
  grounds for "not supported" and the frontier prompt did not — was found and fixed
  before the final runs, because it favoured Jev. See `configs/prompts/`.
- **A discarded run:** an earlier Astra run failed on 84 of 495 examples because our
  own 16-token completion budget was consumed by the model's reasoning tokens. That
  was our bug, not a model defect, and the run was discarded rather than published.
- **Latency is not a controlled measurement.** Both runs used concurrency 12, so the
  timings include client-side queueing.
- **Scope:** dev split only, 45 examples per dataset, one frontier model. Thresholds
  in the gating and cascade tables were selected in-sample.

## Licence

Code and analysis: MIT. The LLM-AggreFact dataset is not redistributed here — the
loader fetches it from Hugging Face under its own terms, and only prediction rows
keyed by content-derived IDs are published.

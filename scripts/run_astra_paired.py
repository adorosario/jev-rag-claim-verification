#!/usr/bin/env python3
"""Run gpt-6-astra on the EXACT examples Jev already answered (paired head-to-head).

The pairing is what makes this comparable: same items, same prompt semantics,
same day. Writes spec-§28 prediction rows and a §29 run manifest.

    docker compose run --rm dev uv run python scripts/run_astra_paired.py
"""
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.load_aggrefact import load_dev_frame
from src.models.openai_adapter import OpenAIAdapter
from src.run_eval import Example, run_model, write_run_manifest

JEV = REPO / "runs/explore-jev-dev/jev_dev_sample.jsonl"
RUN_ID = "paired-astra-lowthink-20260921-p2b"
RUN_DIR = REPO / "runs" / RUN_ID

def main() -> int:
    ids = [json.loads(l)["example_id"] for l in JEV.read_text().splitlines() if l.strip()]
    dev = load_dev_frame()
    inp = dev.inputs.set_index("example_id")
    missing = [i for i in ids if i not in inp.index]
    if missing:
        print(f"FATAL: {len(missing)} ids not in dev split", file=sys.stderr); return 2
    examples = [Example(example_id=i, dataset=inp.loc[i, "dataset"],
                        claim=inp.loc[i, "claim"], evidence=inp.loc[i, "doc"]) for i in ids]
    print(f"{len(examples)} paired examples", flush=True)

    adapter = OpenAIAdapter("gpt-6-astra", model_key="gpt6_astra", max_completion_tokens=256)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    counts = run_model(adapter, examples, run_id=RUN_ID, run_dir=RUN_DIR,
                       model_key="gpt6_astra", prompt_hash=adapter.prompt_hash,
                       config_hash=adapter.prompt_hash, concurrency=12, resume=True)
    print(f"done in {time.time()-t0:.0f}s: {counts}", flush=True)

    write_run_manifest(
        RUN_DIR, run_id=RUN_ID,
        dataset={"name": "lytang/LLM-AggreFact", "split": "dev",
                 "count": len(examples), "parquet_sha256": dev.parquet_sha256,
                 "rows_sha256": dev.rows_sha256,
                 "selection": "the exact example_ids Jev answered in runs/explore-jev-dev"},
        models={"gpt6_astra": {"requested": "gpt-6-astra",
                               "mode": adapter.mode, "reasoning_effort": adapter.reasoning_effort,
                               "max_completion_tokens": adapter.max_completion_tokens,
                               "native_probability": adapter.native_probability}},
        prompt_sha256=adapter.prompt_hash, started_at_utc=started,
        note="Paired exploratory comparison against the Jev dev sample; dev split only.")
    (RUN_DIR / "paired_example_ids.json").write_text(json.dumps(ids, indent=1) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

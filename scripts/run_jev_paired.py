#!/usr/bin/env python3
"""Re-run Jev over the same 495 examples through the standard runner.

The earlier exploratory Jev file had no §29 manifest and a reduced schema, so the
model version and prompt hash behind those answers could not be proven. This run
fixes that and matches the frontier runs' concurrency so the two are comparable.
"""
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.load_aggrefact import load_dev_frame
from src.models.jev import JevAdapter
from src.run_eval import Example, run_model, write_run_manifest

IDS = REPO / "runs/explore-jev-dev/jev_dev_sample.jsonl"
RUN_ID = "paired-jev-20260921"
RUN_DIR = REPO / "runs" / RUN_ID
CONCURRENCY = 12          # matched to the frontier runs

def main() -> int:
    ids = [json.loads(l)["example_id"] for l in IDS.read_text().splitlines() if l.strip()]
    dev = load_dev_frame()
    inp = dev.inputs.drop_duplicates("example_id").set_index("example_id")
    examples = [Example(example_id=i, dataset=inp.loc[i, "dataset"],
                        claim=inp.loc[i, "claim"], evidence=inp.loc[i, "doc"]) for i in ids]
    adapter = JevAdapter()
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    counts = run_model(adapter, examples, run_id=RUN_ID, run_dir=RUN_DIR,
                       model_key="jev", prompt_hash=adapter.prompt_hash,
                       config_hash=adapter.prompt_hash, concurrency=CONCURRENCY, resume=True)
    print(f"done in {time.time()-t0:.0f}s: {counts}", flush=True)
    write_run_manifest(
        RUN_DIR, run_id=RUN_ID,
        dataset={"name": "lytang/LLM-AggreFact", "split": "dev", "count": len(examples),
                 "parquet_sha256": dev.parquet_sha256, "rows_sha256": dev.rows_sha256,
                 "selection": "the same 495 example_ids as the paired frontier runs"},
        models={"jev": {"requested": getattr(adapter, "model_id_requested", "jev-1.13.0"),
                        "native_probability": True}},
        prompt_sha256=adapter.prompt_hash, started_at_utc=started,
        concurrency=CONCURRENCY,
        note="Paired Jev run with a §29 manifest; concurrency matched to the frontier runs.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

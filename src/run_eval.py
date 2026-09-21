"""Evaluation runner: retries, idempotent resume, prediction JSONL, run manifest.

Spec §17 (retry), §28 (JSONL), §29 (manifest), §31 (concurrency); errata E16, E17.

Two rules drive the design:

* **Capacity failures are not results.** A 429 is our rate limit, not a model
  defect (errata E17). Such a row is written so the run is auditable, but it is
  NOT treated as final: a later resume pass re-runs it until it resolves.
  Model failures (format errors, refusals, persistent 5xx) ARE final and count
  as wrong in operational BAcc.
* **Resume must never duplicate a final row.** The JSONL is append-only, so the
  reader takes the LAST row per (model, example_id) and the runner skips any
  example whose last row is already final.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models.base import (
    CAPACITY_STATUSES,
    STATUS_CONTEXT_INCOMPATIBLE,
    STATUS_OK,
    STATUS_PROVIDER_ERROR,
    STATUS_RATE_LIMITED_EXHAUSTED,
    STATUS_TIMEOUT,
    DecisionResult,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Spec §17: retry ONLY these, at most MAX_RETRIES times.
RETRYABLE_STATUSES = frozenset({
    STATUS_RATE_LIMITED_EXHAUSTED, STATUS_TIMEOUT, STATUS_PROVIDER_ERROR,
})
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 30.0

# A row with one of these statuses is final: resume will not re-run it.
# Everything else (i.e. capacity) is retried on the next pass (errata E17).
FINAL_STATUSES = frozenset({
    STATUS_OK, STATUS_CONTEXT_INCOMPATIBLE,
    "format_error", "refusal", "unknown_error", "provider_error",
})


@dataclass
class Example:
    example_id: str
    dataset: str
    claim: str
    evidence: str


def predictions_path(run_dir: Path, model_key: str) -> Path:
    return run_dir / f"predictions_{model_key}.jsonl"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn final line from a kill; resume rewrites it
    return rows


def last_row_per_example(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Append-only file: the newest row for an example wins."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[r["example_id"]] = r
    return out


def completed_example_ids(path: Path) -> set[str]:
    """Examples that need no further work (errata E17: capacity is excluded)."""
    return {
        eid for eid, row in last_row_per_example(read_rows(path)).items()
        if row.get("status") in FINAL_STATUSES
    }


def _sleep_for_retry(attempt: int, retry_after: float | None) -> None:
    if retry_after is not None:
        time.sleep(min(retry_after, BACKOFF_MAX_S))
        return
    delay = min(BACKOFF_BASE_S * (2 ** attempt), BACKOFF_MAX_S)
    time.sleep(delay * (0.5 + random.random() * 0.5))  # jitter


def verify_with_retries(adapter: Any, ex: Example, *, max_retries: int = MAX_RETRIES
                        ) -> DecisionResult:
    """Spec §17. Returns the final DecisionResult with a truthful `attempts`."""
    attempts = 0
    result: DecisionResult | None = None
    for attempt in range(max_retries + 1):
        attempts += 1
        result = adapter.verify(
            example_id=ex.example_id, claim=ex.claim, evidence=ex.evidence)
        if result.status not in RETRYABLE_STATUSES:
            break
        if attempt < max_retries:
            _sleep_for_retry(attempt, _retry_after_from(result))
    assert result is not None
    result.attempts = attempts
    return result


def _retry_after_from(result: DecisionResult) -> float | None:
    msg = (result.error_message or "").lower()
    for key in ("retry-after: ", "retry after "):
        if key in msg:
            try:
                return float(msg.split(key, 1)[1].split()[0])
            except (ValueError, IndexError):
                return None
    return None


def run_model(
    adapter: Any,
    examples: Sequence[Example],
    *,
    run_id: str,
    run_dir: Path,
    model_key: str,
    prompt_hash: str,
    config_hash: str,
    concurrency: int = 1,
    resume: bool = True,
) -> dict[str, int]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = predictions_path(run_dir, model_key)
    done = completed_example_ids(path) if resume else set()
    todo = [e for e in examples if e.example_id not in done]

    counts = {"total": len(examples), "skipped_done": len(examples) - len(todo),
              "ok": 0, "capacity_failures": 0, "model_failures": 0,
              "context_incompatible": 0}

    def work(ex: Example) -> DecisionResult:
        return verify_with_retries(adapter, ex)

    # Append as results arrive so a kill mid-run loses at most one row.
    with open(path, "a") as fh:
        if concurrency <= 1:
            results = (work(e) for e in todo)
            for ex, res in zip(todo, results, strict=True):
                _write(fh, ex, res, run_id, prompt_hash, config_hash, counts)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for ex, res in zip(todo, pool.map(work, todo), strict=True):
                    _write(fh, ex, res, run_id, prompt_hash, config_hash, counts)
    return counts


def _write(fh: Any, ex: Example, res: DecisionResult, run_id: str,
           prompt_hash: str, config_hash: str, counts: dict[str, int]) -> None:
    row = res.to_jsonl_row(run_id=run_id, dataset=ex.dataset,
                           prompt_hash=prompt_hash, config_hash=config_hash)
    fh.write(json.dumps(row, default=str) + "\n")
    fh.flush()
    os.fsync(fh.fileno())  # a kill test must not lose an acknowledged row
    if res.status == STATUS_OK:
        counts["ok"] += 1
    elif res.status in CAPACITY_STATUSES:
        counts["capacity_failures"] += 1
    elif res.status == STATUS_CONTEXT_INCOMPATIBLE:
        counts["context_incompatible"] += 1
    else:
        counts["model_failures"] += 1


def write_run_manifest(run_dir: Path, *, run_id: str, dataset: dict[str, Any],
                       models: dict[str, Any], prompt_sha256: str,
                       random_seed: int = 20260918, **extra: Any) -> Path:
    """Spec §29."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark_version": "1.0",
        "run_id": run_id,
        "started_at_utc": extra.pop("started_at_utc", None),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "runner_region": os.environ.get("RUNNER_REGION", "local"),
        "runner_instance": platform.node(),
        "python_version": sys.version.split()[0],
        "dataset": dataset,
        "models": models,
        "prompt_sha256": prompt_sha256,
        "pricing_snapshot_sha256": extra.pop("pricing_snapshot_sha256", None),
        "random_seed": random_seed,
        # Errata E17/B5: cost and capacity logs are only interpretable if the
        # account they were produced under is recorded.
        "credentials_account_note": extra.pop("credentials_account_note", None),
        **extra,
    }
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (run_dir / "git_commit.txt").write_text(_git_head() + "\n")
    (run_dir / "environment.txt").write_text(_environment_text())
    return path


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _environment_text() -> str:
    lock = REPO_ROOT / "uv.lock"
    lock_sha = "missing"
    if lock.is_file():
        from .data.hashes import sha256_file
        lock_sha = sha256_file(lock)
    return "\n".join([
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"node={platform.node()}",
        f"runner_region={os.environ.get('RUNNER_REGION', 'local')}",
        f"uv_lock_sha256={lock_sha}",
    ]) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run models over a split (spec §30).")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--models", required=True, help="comma-separated model keys")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    print("run_eval wiring is completed in the G1 gate; "
          f"parsed args: {vars(args)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

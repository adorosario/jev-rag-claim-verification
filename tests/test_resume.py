"""Retry and idempotent-resume tests (spec §17, §36; errata E17)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.base import (  # noqa: E402
    STATUS_CONTEXT_INCOMPATIBLE,
    STATUS_FORMAT_ERROR,
    STATUS_OK,
    STATUS_RATE_LIMITED_EXHAUSTED,
    STATUS_TIMEOUT,
    DecisionResult,
)
from src.run_eval import (  # noqa: E402
    Example,
    completed_example_ids,
    last_row_per_example,
    predictions_path,
    read_rows,
    run_model,
    verify_with_retries,
)


def _res(eid, status=STATUS_OK, label="SUPPORTED"):
    return DecisionResult(
        example_id=eid, model_key="m", model_id_requested="m1", model_id_reported="m1",
        label=label if status == STATUS_OK else None,
        p_supported=0.9 if status == STATUS_OK else None,
        native_probability=True, input_tokens=10, output_tokens=1,
        provider_cost_usd=None, latency_ms=1.0, attempts=1, status=status)


class _Adapter:
    """Returns a scripted status sequence per example."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def verify(self, *, example_id, claim, evidence):
        self.calls.append(example_id)
        seq = self.script.get(example_id, [STATUS_OK])
        status = seq.pop(0) if len(seq) > 1 else seq[0]
        return _res(example_id, status)


EXAMPLES = [Example(f"e{i}", "D", f"claim{i}", f"doc{i}") for i in range(5)]


# --- retry policy (§17) ---------------------------------------------------

def test_no_retry_on_success():
    a = _Adapter({"e0": [STATUS_OK]})
    r = verify_with_retries(a, EXAMPLES[0])
    assert r.status == STATUS_OK
    assert r.attempts == 1
    assert len(a.calls) == 1


def test_retries_then_succeeds_and_reports_true_attempt_count():
    a = _Adapter({"e0": [STATUS_TIMEOUT, STATUS_TIMEOUT, STATUS_OK]})
    r = verify_with_retries(a, EXAMPLES[0], max_retries=3)
    assert r.status == STATUS_OK
    assert r.attempts == 3
    assert len(a.calls) == 3


def test_retries_are_capped():
    a = _Adapter({"e0": [STATUS_TIMEOUT]})
    r = verify_with_retries(a, EXAMPLES[0], max_retries=2)
    assert r.status == STATUS_TIMEOUT
    assert r.attempts == 3  # initial + 2 retries
    assert len(a.calls) == 3


def test_format_error_is_not_retried():
    """§17 retries transport problems, not a model answering badly."""
    a = _Adapter({"e0": [STATUS_FORMAT_ERROR]})
    r = verify_with_retries(a, EXAMPLES[0], max_retries=3)
    assert r.attempts == 1
    assert len(a.calls) == 1


def test_context_incompatible_is_not_retried():
    a = _Adapter({"e0": [STATUS_CONTEXT_INCOMPATIBLE]})
    r = verify_with_retries(a, EXAMPLES[0], max_retries=3)
    assert r.attempts == 1


# --- resume (§36) ---------------------------------------------------------

def test_kill_halfway_then_resume_without_duplicates(tmp_path):
    """Spec §36: kill a run halfway; resume adds no duplicate rows."""
    a = _Adapter({})
    run_model(a, EXAMPLES[:3], run_id="r", run_dir=tmp_path, model_key="m",
              prompt_hash="p", config_hash="c")
    first = read_rows(predictions_path(tmp_path, "m"))
    assert len(first) == 3

    a2 = _Adapter({})
    counts = run_model(a2, EXAMPLES, run_id="r", run_dir=tmp_path, model_key="m",
                       prompt_hash="p", config_hash="c")
    rows = read_rows(predictions_path(tmp_path, "m"))
    assert counts["skipped_done"] == 3
    assert a2.calls == ["e3", "e4"]          # only the missing ones re-ran
    assert len(rows) == 5
    assert len({r["example_id"] for r in rows}) == 5


def test_resume_is_idempotent_when_nothing_is_missing(tmp_path):
    run_model(_Adapter({}), EXAMPLES, run_id="r", run_dir=tmp_path, model_key="m",
              prompt_hash="p", config_hash="c")
    a2 = _Adapter({})
    counts = run_model(a2, EXAMPLES, run_id="r", run_dir=tmp_path, model_key="m",
                       prompt_hash="p", config_hash="c")
    assert a2.calls == []
    assert counts["skipped_done"] == 5
    assert len(read_rows(predictions_path(tmp_path, "m"))) == 5


def test_capacity_failures_are_rerun_on_resume(tmp_path):
    """Errata E17: a rate limit is ours, so the row is not final."""
    a = _Adapter({"e0": [STATUS_RATE_LIMITED_EXHAUSTED], "e1": [STATUS_OK],
                  "e2": [STATUS_OK], "e3": [STATUS_OK], "e4": [STATUS_OK]})
    c1 = run_model(a, EXAMPLES, run_id="r", run_dir=tmp_path, model_key="m",
                   prompt_hash="p", config_hash="c")
    assert c1["capacity_failures"] == 1

    a2 = _Adapter({})  # capacity resolved
    c2 = run_model(a2, EXAMPLES, run_id="r", run_dir=tmp_path, model_key="m",
                   prompt_hash="p", config_hash="c")
    assert a2.calls == ["e0"]
    assert c2["ok"] == 1
    final = last_row_per_example(read_rows(predictions_path(tmp_path, "m")))
    assert final["e0"]["status"] == STATUS_OK   # the later row wins
    assert len(final) == 5


def test_model_failures_are_final_and_not_rerun(tmp_path):
    """A refusal or format error is the model's answer; rerunning it would be
    resampling until we like the result."""
    a = _Adapter({"e0": [STATUS_FORMAT_ERROR]})
    run_model(a, EXAMPLES, run_id="r", run_dir=tmp_path, model_key="m",
              prompt_hash="p", config_hash="c")
    a2 = _Adapter({})
    run_model(a2, EXAMPLES, run_id="r", run_dir=tmp_path, model_key="m",
              prompt_hash="p", config_hash="c")
    assert a2.calls == []


def test_completed_ids_excludes_capacity(tmp_path):
    a = _Adapter({"e0": [STATUS_RATE_LIMITED_EXHAUSTED]})
    run_model(a, EXAMPLES[:2], run_id="r", run_dir=tmp_path, model_key="m",
              prompt_hash="p", config_hash="c")
    done = completed_example_ids(predictions_path(tmp_path, "m"))
    assert "e0" not in done
    assert "e1" in done


def test_torn_final_line_does_not_break_resume(tmp_path):
    """A kill can leave a partial line; resume must tolerate it."""
    run_model(_Adapter({}), EXAMPLES[:2], run_id="r", run_dir=tmp_path,
              model_key="m", prompt_hash="p", config_hash="c")
    p = predictions_path(tmp_path, "m")
    with open(p, "a") as fh:
        fh.write('{"example_id": "e2", "status": "ok"')  # truncated
    rows = read_rows(p)
    assert len(rows) == 2
    a2 = _Adapter({})
    run_model(a2, EXAMPLES[:3], run_id="r", run_dir=tmp_path, model_key="m",
              prompt_hash="p", config_hash="c")
    assert "e2" in a2.calls


def test_counts_separate_capacity_from_model_failures(tmp_path):
    a = _Adapter({"e0": [STATUS_RATE_LIMITED_EXHAUSTED], "e1": [STATUS_FORMAT_ERROR],
                  "e2": [STATUS_CONTEXT_INCOMPATIBLE]})
    c = run_model(a, EXAMPLES, run_id="r", run_dir=tmp_path, model_key="m",
                  prompt_hash="p", config_hash="c")
    assert c["capacity_failures"] == 1
    assert c["model_failures"] == 1
    assert c["context_incompatible"] == 1
    assert c["ok"] == 2


def test_rows_match_spec_28_schema(tmp_path):
    run_model(_Adapter({}), EXAMPLES[:1], run_id="run7", run_dir=tmp_path,
              model_key="m", prompt_hash="sha256:p", config_hash="sha256:c")
    row = json.loads(predictions_path(tmp_path, "m").read_text().splitlines()[0])
    for k in ("run_id", "example_id", "dataset", "model_key", "model_id_requested",
              "model_id_reported", "label", "p_supported", "native_probability",
              "input_tokens", "output_tokens", "cost_usd", "latency_ms",
              "attempts", "status", "prompt_hash", "config_hash"):
        assert k in row, k
    assert row["run_id"] == "run7"


def test_run_manifest_has_spec_29_fields(tmp_path):
    from src.run_eval import write_run_manifest
    p = write_run_manifest(tmp_path, run_id="r", dataset={"name": "x"},
                           models={"jev": {}}, prompt_sha256="sha256:p")
    m = json.loads(p.read_text())
    for k in ("benchmark_version", "git_commit", "runner_region", "python_version",
              "dataset", "models", "prompt_sha256", "random_seed"):
        assert k in m, k
    assert (tmp_path / "environment.txt").is_file()
    assert (tmp_path / "git_commit.txt").is_file()

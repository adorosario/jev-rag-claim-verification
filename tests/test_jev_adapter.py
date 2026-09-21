"""Offline tests for the Jev adapter's normalization (spec §27, errata E1/E4/E9/E17).

These use a stub client so the mapping is testable without spending API calls
or depending on the network.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.base import (  # noqa: E402
    STATUS_FORMAT_ERROR,
    STATUS_OK,
    STATUS_RATE_LIMITED_EXHAUSTED,
    STATUS_TIMEOUT,
    DecisionResult,
)
from src.models.jev import JEV_MODEL_ID, JevAdapter, load_prompt, prompt_hash  # noqa: E402


class _Usage:
    def __init__(self, i=442, o=35):
        self.input_tokens, self.output_tokens = i, o


class _ChoiceAnswer:
    def __init__(self, choice, probabilities, confidence):
        self.choice, self.probabilities, self.confidence = choice, probabilities, confidence


class _Response:
    def __init__(self, answers, model=JEV_MODEL_ID):
        self.answers, self.model, self.usage = answers, model, _Usage()
        self.request_id = "req_test"

    def model_dump(self, mode="json"):
        return {"model": self.model, "answers": {k: vars(v) for k, v in self.answers.items()}}


class _StubClient:
    def __init__(self, response=None, raise_exc=None):
        self._response, self._raise = response, raise_exc
        self.last_model = None

    def system_one(self, *, state, questions, model):
        self.last_model = model
        if self._raise:
            raise self._raise
        return self._response

    def close(self):
        pass


def _adapter(client):
    return JevAdapter(client=client)


def test_supported_maps_to_native_probability():
    ans = _ChoiceAnswer("supported", {"supported": 0.93, "not_supported": 0.07}, 0.85)
    a = _adapter(_StubClient(_Response({"grounding": ans})))
    r = a.verify(example_id="e1", claim="c", evidence="e")
    assert r.status == STATUS_OK
    assert r.label == "SUPPORTED"
    assert r.p_supported == 0.93
    assert r.native_probability is True
    assert r.attempts == 1


def test_confidence_is_secondary_not_the_probability():
    """Errata E4: confidence must never stand in for p_supported."""
    ans = _ChoiceAnswer("supported", {"supported": 0.62, "not_supported": 0.38}, 0.25)
    r = _adapter(_StubClient(_Response({"grounding": ans}))).verify(
        example_id="e", claim="c", evidence="e")
    assert r.p_supported == 0.62
    assert r.extra["jev_confidence"] == 0.25
    assert r.p_supported != r.extra["jev_confidence"]


def test_not_supported_label_mapping():
    ans = _ChoiceAnswer("not_supported", {"supported": 0.0, "not_supported": 1.0}, 1.0)
    r = _adapter(_StubClient(_Response({"grounding": ans}))).verify(
        example_id="e", claim="c", evidence="e")
    assert r.label == "NOT_SUPPORTED"
    assert r.p_supported == 0.0


def test_saturated_probability_is_preserved_not_clipped():
    """Clipping is a §18.4 analysis decision preregistered at G2A, not an
    adapter behaviour. The raw value must reach the JSONL untouched."""
    ans = _ChoiceAnswer("not_supported", {"supported": 0.0, "not_supported": 1.0}, 1.0)
    r = _adapter(_StubClient(_Response({"grounding": ans}))).verify(
        example_id="e", claim="c", evidence="e")
    assert r.p_supported == 0.0


def test_model_is_always_pinned_explicitly():
    """Errata E9: never ride the moving `jev-latest` alias."""
    ans = _ChoiceAnswer("supported", {"supported": 1.0, "not_supported": 0.0}, 1.0)
    stub = _StubClient(_Response({"grounding": ans}))
    _adapter(stub).verify(example_id="e", claim="c", evidence="e")
    assert stub.last_model == JEV_MODEL_ID


def test_unmapped_choice_is_a_format_error():
    ans = _ChoiceAnswer("maybe", {"maybe": 1.0}, 1.0)
    r = _adapter(_StubClient(_Response({"grounding": ans}))).verify(
        example_id="e", claim="c", evidence="e")
    assert r.status == STATUS_FORMAT_ERROR
    assert r.label is None


def test_missing_answer_is_a_format_error():
    r = _adapter(_StubClient(_Response({}))).verify(example_id="e", claim="c", evidence="e")
    assert r.status == STATUS_FORMAT_ERROR
    assert r.label is None


def test_rate_limit_is_capacity_not_model_failure():
    """Errata E17: our rate limit must not be scored as a wrong answer."""
    import httpx2
    from typesafe_sdk import TypeSafeRateLimitError
    exc = TypeSafeRateLimitError(status=429, body={}, headers=httpx2.Headers({"retry-after": "1"}))
    r = _adapter(_StubClient(raise_exc=exc)).verify(example_id="e", claim="c", evidence="e")
    assert r.status == STATUS_RATE_LIMITED_EXHAUSTED
    assert r.is_capacity_failure is True
    assert r.is_model_failure is False


def test_timeout_is_a_model_failure():
    """TypeSafeAPITimeoutError subclasses TypeSafeAPIConnectionError, so the
    status mapping must check the timeout case FIRST or every timeout would be
    mislabelled provider_error."""
    from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPITimeoutError
    assert issubclass(TypeSafeAPITimeoutError, TypeSafeAPIConnectionError)
    exc = TypeSafeAPITimeoutError(timeout=120.0)
    r = _adapter(_StubClient(raise_exc=exc)).verify(example_id="e", claim="c", evidence="e")
    assert r.status == STATUS_TIMEOUT
    assert r.is_model_failure is True
    assert r.is_capacity_failure is False


def test_connection_error_is_provider_error():
    from typesafe_sdk import TypeSafeAPIConnectionError
    from src.models.base import STATUS_PROVIDER_ERROR
    r = _adapter(_StubClient(raise_exc=TypeSafeAPIConnectionError("boom"))).verify(
        example_id="e", claim="c", evidence="e")
    assert r.status == STATUS_PROVIDER_ERROR


def test_unknown_exception_is_unknown_error():
    from src.models.base import STATUS_UNKNOWN_ERROR
    r = _adapter(_StubClient(raise_exc=RuntimeError("???"))).verify(
        example_id="e", claim="c", evidence="e")
    assert r.status == STATUS_UNKNOWN_ERROR


def test_prompt_hash_is_stable_and_covers_criteria():
    p = load_prompt()
    h1 = prompt_hash(p)
    assert h1 == prompt_hash(load_prompt())
    p2 = dict(p)
    p2["criteria"] = dict(p["criteria"])
    p2["criteria"]["supported"] = p2["criteria"]["supported"] + " "
    assert prompt_hash(p2) != h1


def test_e1_wording_fix_is_present():
    """Errata E1: the ambiguous 'any' phrasing must be gone."""
    c = load_prompt()["criteria"]["not_supported"]
    assert "at least one materially important part" in c
    assert "fails to support any materially important part" not in c


def test_jsonl_row_matches_spec_28():
    ans = _ChoiceAnswer("supported", {"supported": 0.93, "not_supported": 0.07}, 0.85)
    r = _adapter(_StubClient(_Response({"grounding": ans}))).verify(
        example_id="e", claim="c", evidence="e")
    row = r.to_jsonl_row(run_id="r", dataset="d", prompt_hash="sha256:p", config_hash="sha256:c")
    for k in ("run_id", "example_id", "dataset", "model_key", "model_id_requested",
              "model_id_reported", "label", "p_supported", "native_probability",
              "input_tokens", "output_tokens", "cost_usd", "latency_ms", "attempts",
              "status", "prompt_hash", "config_hash"):
        assert k in row, k


def test_decision_result_rejects_invalid_status():
    with pytest.raises(ValueError):
        DecisionResult(example_id="e", model_key="jev", model_id_reported=None, label=None,
                       p_supported=None, native_probability=True, input_tokens=None,
                       output_tokens=None, provider_cost_usd=None, latency_ms=1.0,
                       attempts=1, status="not_a_status")

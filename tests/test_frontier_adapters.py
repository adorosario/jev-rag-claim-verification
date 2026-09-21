"""Offline tests for the OpenAI and Anthropic adapters (spec §11, §12, §14; errata E8, E15)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import frontier_prompt as fp  # noqa: E402
from src.models.anthropic_adapter import AnthropicAdapter  # noqa: E402
from src.models.base import (  # noqa: E402
    STATUS_CONTEXT_INCOMPATIBLE,
    STATUS_FORMAT_ERROR,
    STATUS_OK,
    STATUS_RATE_LIMITED_EXHAUSTED,
    STATUS_REFUSAL,
)
from src.models.openai_adapter import OpenAIAdapter  # noqa: E402


# --- stubs ----------------------------------------------------------------

class _TopLP:
    def __init__(self, token, logprob):
        self.token, self.logprob = token, logprob


class _LPContent:
    def __init__(self, alts):
        self.top_logprobs = [_TopLP(t, lp) for t, lp in alts]


class _LP:
    def __init__(self, alts):
        self.content = [_LPContent(alts)]


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, logprobs=None):
        self.message, self.logprobs = _Msg(content), logprobs


class _Usage:
    def __init__(self):
        self.prompt_tokens, self.completion_tokens = 150, 4
        self.completion_tokens_details = None
        self.input_tokens, self.output_tokens = 150, 4


class _OAResp:
    def __init__(self, content, logprobs=None, model="gpt-6-astra"):
        self.choices, self.model, self.usage = [_Choice(content, logprobs)], model, _Usage()


class _Completions:
    def __init__(self, resp=None, exc=None):
        self._resp, self._exc, self.last = resp, exc, None

    def create(self, **kw):
        self.last = kw
        if self._exc:
            raise self._exc
        return self._resp


class _OAClient:
    def __init__(self, resp=None, exc=None):
        self.chat = type("C", (), {"completions": _Completions(resp, exc)})()


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _AnResp:
    def __init__(self, text, stop_reason="end_turn"):
        self.content, self.model, self.stop_reason = [_Block(text)], "claude-sonnet-5", stop_reason
        self.usage = _Usage()


class _Messages:
    def __init__(self, resp=None, exc=None):
        self._resp, self._exc, self.last = resp, exc, None

    def create(self, **kw):
        self.last = kw
        if self._exc:
            raise self._exc
        return self._resp


class _AnClient:
    def __init__(self, resp=None, exc=None):
        self.messages = _Messages(resp, exc)


# --- shared prompt --------------------------------------------------------

def test_all_frontier_models_share_one_prompt():
    """Spec rule 7: the same semantic prompt for every model."""
    cfg = fp.load_config()
    modes = {m: fp.mode_for(m, cfg) for m in ("gpt-6-astra", "gpt-5.6-sol", "claude-sonnet-5")}
    assert set(modes.values()) == {"enum"}
    hashes = {fp.prompt_hash(m, cfg) for m in modes.values()}
    assert len(hashes) == 1


def test_prompt_contains_both_labels_and_the_grounding_instruction():
    p = fp.render("C", "E", "enum")
    assert "SUPPORTED" in p and "NOT_SUPPORTED" in p
    assert "Judge only from the supplied evidence" in p
    assert "C" in p and "E" in p


def test_e1_parity_frontier_prompt_means_at_least_one():
    """Errata E1 parity check: §11's 'if any ... is missing' is the existential
    reading, which matches the corrected Jev criterion."""
    p = fp.render("C", "E", "enum")
    assert "Use this if any materially important part is missing" in p


def test_prompt_hash_changes_with_mode():
    assert fp.prompt_hash("enum") != fp.prompt_hash("ab_logprobs")


# --- OpenAI ---------------------------------------------------------------

def _oa(resp=None, exc=None, **kw):
    return OpenAIAdapter("gpt-6-astra", model_key="gpt6_astra",
                         client=_OAClient(resp, exc), **kw)


def test_openai_enum_mode_has_no_native_probability():
    """Errata E15: logprobs are unavailable on the required models."""
    a = _oa(_OAResp("SUPPORTED"))
    r = a.verify(example_id="e", claim="c", evidence="v")
    assert r.status == STATUS_OK
    assert r.label == "SUPPORTED"
    assert r.native_probability is False
    assert r.p_supported is None


def test_openai_does_not_send_logprobs_in_enum_mode():
    a = _oa(_OAResp("SUPPORTED"))
    a.verify(example_id="e", claim="c", evidence="v")
    sent = a._client.chat.completions.last
    assert "logprobs" not in sent
    assert "max_completion_tokens" in sent  # max_tokens is rejected by these models
    assert "temperature" not in sent        # rejected at 0 by these models
    assert sent["reasoning_effort"] == "low"


def test_openai_ab_mode_computes_probability_from_logprobs():
    """The §12 path, kept working for any model that does expose logprobs."""
    import math
    lp = _LP([("A", math.log(0.8)), ("B", math.log(0.2))])
    a = OpenAIAdapter("gpt-4.1", model_key="gpt41", client=_OAClient(_OAResp("A", lp)),
                      mode="ab_logprobs")
    r = a.verify(example_id="e", claim="c", evidence="v")
    assert r.native_probability is True
    assert r.p_supported == pytest.approx(0.8, abs=1e-9)
    assert r.label == "SUPPORTED"


def test_openai_ab_mode_never_fabricates_a_probability():
    """§12: if A or B is absent from the alternatives, score the label only."""
    lp = _LP([("X", -0.1), ("Y", -2.0)])
    a = OpenAIAdapter("gpt-4.1", model_key="gpt41", client=_OAClient(_OAResp("A", lp)),
                      mode="ab_logprobs")
    r = a.verify(example_id="e", claim="c", evidence="v")
    assert r.label == "SUPPORTED"
    assert r.p_supported is None
    assert r.extra["native_probability_available"] is False


def test_openai_unparseable_reply_is_format_error():
    r = _oa(_OAResp("I think so maybe")).verify(example_id="e", claim="c", evidence="v")
    assert r.status == STATUS_FORMAT_ERROR
    assert r.label is None


def test_openai_refusal_is_its_own_status():
    r = _oa(_OAResp("I cannot help with that.")).verify(example_id="e", claim="c", evidence="v")
    assert r.status == STATUS_REFUSAL
    assert r.is_model_failure is False or r.status == STATUS_REFUSAL


def test_openai_context_rejection_maps_to_context_incompatible():
    """§14: an over-long example is an exclusion, not a model error."""
    from openai import BadRequestError
    import httpx
    exc = BadRequestError(
        "maximum context length exceeded",
        response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
        body=None)
    r = _oa(exc=exc).verify(example_id="e", claim="c", evidence="v")
    assert r.status == STATUS_CONTEXT_INCOMPATIBLE


def test_openai_rate_limit_is_capacity_failure():
    from openai import RateLimitError
    import httpx
    exc = RateLimitError(
        "429", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None)
    r = _oa(exc=exc).verify(example_id="e", claim="c", evidence="v")
    assert r.status == STATUS_RATE_LIMITED_EXHAUSTED
    assert r.is_capacity_failure is True
    assert r.is_model_failure is False


# --- Anthropic ------------------------------------------------------------

def test_anthropic_is_never_native_probability():
    """Spec §12: Table 3 shows N/A for Claude."""
    a = AnthropicAdapter(client=_AnClient(_AnResp("SUPPORTED")))
    r = a.verify(example_id="e", claim="c", evidence="v")
    assert r.status == STATUS_OK
    assert r.label == "SUPPORTED"
    assert r.native_probability is False
    assert r.p_supported is None
    assert "N/A" in r.extra["calibration"]


def test_anthropic_sends_thinking_disabled_and_no_temperature():
    """Errata E8, plus anthropic 1.6.0 removed temperature from create()."""
    a = AnthropicAdapter(client=_AnClient(_AnResp("SUPPORTED")))
    a.verify(example_id="e", claim="c", evidence="v")
    sent = a._client.messages.last
    assert sent["thinking"] == {"type": "disabled"}
    assert "temperature" not in sent
    assert "tools" not in sent


def test_anthropic_not_supported_parsing():
    a = AnthropicAdapter(client=_AnClient(_AnResp("NOT_SUPPORTED")))
    r = a.verify(example_id="e", claim="c", evidence="v")
    assert r.label == "NOT_SUPPORTED"


def test_anthropic_refusal_status():
    a = AnthropicAdapter(client=_AnClient(_AnResp("I'm unable to assess this.")))
    r = a.verify(example_id="e", claim="c", evidence="v")
    assert r.status == STATUS_REFUSAL


def test_anthropic_format_error():
    a = AnthropicAdapter(client=_AnClient(_AnResp("no idea")))
    r = a.verify(example_id="e", claim="c", evidence="v")
    assert r.status == STATUS_FORMAT_ERROR

"""OpenAI adapter for gpt-6-astra and gpt-5.6-sol (spec §7.2, §7.3, §11, §12; errata E8, E15).

Live findings recorded on 2026-09-19 that shape this adapter:

* **No logprobs.** Both required models reject the `logprobs` parameter, so the
  §12 A/B probability method is unavailable and `native_probability` is False.
  The A/B path is still implemented for any model that does expose logprobs.
* **`max_tokens` is rejected**; these models require `max_completion_tokens`.
* **`temperature=0` is rejected.** Determinism cannot be requested, so the
  provider default is used and recorded (relevant to §18.6 repeatability).
* **`reasoning_effort`** accepts `low` but rejects `minimal` and `none`, so
  `low` is the floor for the §7.2 task-optimized run (errata E8).
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from . import frontier_prompt as fp
from .base import (
    STATUS_CONTEXT_INCOMPATIBLE,
    STATUS_FORMAT_ERROR,
    STATUS_OK,
    STATUS_PROVIDER_ERROR,
    STATUS_RATE_LIMITED_EXHAUSTED,
    STATUS_REFUSAL,
    STATUS_TIMEOUT,
    STATUS_UNKNOWN_ERROR,
    DecisionResult,
    ModelAdapter,
    sha256_of_payload,
)
from .label_parsing import LabelParseError, looks_like_refusal, parse_ab_label, parse_enum_label

DEFAULT_MAX_COMPLETION_TOKENS = 16
DEFAULT_TIMEOUT_S = 120.0


class OpenAIAdapter(ModelAdapter):
    def __init__(
        self,
        model_id: str,
        *,
        model_key: str | None = None,
        client: Any | None = None,
        mode: str | None = None,
        reasoning_effort: str | None = "low",
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model_id_requested = model_id
        self.model_key = model_key or model_id.replace("-", "_").replace(".", "")
        self.cfg = fp.load_config()
        self.mode = mode or fp.mode_for(model_id, self.cfg)
        self.native_probability = bool(self.cfg["modes"][self.mode]["native_probability"])
        self.prompt_hash = fp.prompt_hash(self.mode, self.cfg)
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        # max_retries=0: the runner owns the §17 policy so `attempts` is real
        # and capacity failures stay separable (errata E17).
        self._client = client or OpenAI(
            api_key=os.environ["OPENAI_API_KEY"], timeout=timeout_s, max_retries=0)

    def _request_kwargs(self, prompt: str) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.model_id_requested,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_completion_tokens,
        }
        # temperature is deliberately omitted: these models reject 0 and the
        # provider default is what we record (errata E8).
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        if self.native_probability:
            kw["logprobs"] = True
            kw["top_logprobs"] = 20
        return kw

    def verify(self, *, example_id: str, claim: str, evidence: str) -> DecisionResult:
        prompt = fp.render(claim, evidence, self.mode, self.cfg)
        started = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**self._request_kwargs(prompt))
        except Exception as exc:
            return self._failure(example_id, exc, started)

        latency_ms = (time.perf_counter() - started) * 1000.0
        choice = resp.choices[0]
        text = choice.message.content
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None)
        out_tok = getattr(usage, "completion_tokens", None)

        def _result(**kw: Any) -> DecisionResult:
            base = dict(
                example_id=example_id, model_key=self.model_key,
                model_id_requested=self.model_id_requested,
                model_id_reported=getattr(resp, "model", None),
                native_probability=self.native_probability,
                input_tokens=in_tok, output_tokens=out_tok, provider_cost_usd=None,
                latency_ms=latency_ms, attempts=1,
                raw_response_sha256=sha256_of_payload({"content": text}),
            )
            base.update(kw)
            return DecisionResult(**base)

        if looks_like_refusal(text):
            return _result(label=None, p_supported=None, status=STATUS_REFUSAL,
                           error_type="refusal", error_message=(text or "")[:300])
        try:
            label = (parse_ab_label(text) if self.mode == "ab_logprobs"
                     else parse_enum_label(text))
        except LabelParseError as exc:
            return _result(label=None, p_supported=None, status=STATUS_FORMAT_ERROR,
                           error_type="LabelParseError", error_message=str(exc)[:300])

        p_supported, extra = None, {
            "mode": self.mode,
            "reasoning_effort": self.reasoning_effort,
            "temperature": "provider_default",
            "raw_text": (text or "")[:200],
        }
        rt = getattr(usage, "completion_tokens_details", None)
        if rt is not None:
            extra["reasoning_tokens"] = getattr(rt, "reasoning_tokens", None)

        if self.native_probability:
            p_supported, note = _p_from_logprobs(choice)
            extra["logprob_note"] = note
            if p_supported is None:
                # §12: never fabricate a probability; the label still scores.
                extra["native_probability_available"] = False

        return _result(label=label, p_supported=p_supported, status=STATUS_OK, extra=extra)

    def _failure(self, example_id: str, exc: Exception, started: float) -> DecisionResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        msg = str(exc)
        status = _status_for(exc, msg)
        return DecisionResult(
            example_id=example_id, model_key=self.model_key,
            model_id_requested=self.model_id_requested, model_id_reported=None,
            label=None, p_supported=None, native_probability=self.native_probability,
            input_tokens=None, output_tokens=None, provider_cost_usd=None,
            latency_ms=latency_ms, attempts=1, status=status,
            error_type=type(exc).__name__, error_message=msg[:500],
        )


def _p_from_logprobs(choice: Any) -> tuple[float | None, str]:
    """Spec §12: p = exp(logP(A)) / (exp(logP(A)) + exp(logP(B)))."""
    lp = getattr(choice, "logprobs", None)
    if not lp or not getattr(lp, "content", None):
        return None, "no logprobs returned"
    alts = {t.token.strip().upper(): t.logprob for t in lp.content[0].top_logprobs}
    la, lb = alts.get("A"), alts.get("B")
    if la is None or lb is None:
        return None, "A or B absent from the returned top alternatives"
    ea, eb = math.exp(la), math.exp(lb)
    if ea + eb == 0:
        return None, "degenerate logprobs"
    return ea / (ea + eb), "ok"


def _status_for(exc: Exception, msg: str) -> str:
    if isinstance(exc, RateLimitError):
        return STATUS_RATE_LIMITED_EXHAUSTED  # capacity, errata E17
    if isinstance(exc, APITimeoutError):
        return STATUS_TIMEOUT  # must precede APIConnectionError
    if isinstance(exc, BadRequestError):
        low = msg.lower()
        # §14: a context rejection is an exclusion, not a model error.
        if any(k in low for k in ("context length", "context_length",
                                  "too long", "maximum context", "reduce the length")):
            return STATUS_CONTEXT_INCOMPATIBLE
        return STATUS_PROVIDER_ERROR
    if isinstance(exc, (APIConnectionError, InternalServerError)):
        return STATUS_PROVIDER_ERROR
    return STATUS_UNKNOWN_ERROR

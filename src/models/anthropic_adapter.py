"""Anthropic adapter for claude-sonnet-5 (spec §7.4, §11, §12; errata E8).

Spec §12: Anthropic exposes no token-logprob interface for ordinary responses,
so `native_probability` is always False and Table 3 shows N/A for this row.
A self-reported confidence is explicitly NOT collected here -- §12 forbids
placing one beside a native probability without labelling, and the safest way
to honour that is not to produce one on the primary path at all.
"""

from __future__ import annotations

import os
import time
from typing import Any

from anthropic import (
    Anthropic,
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
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
from .label_parsing import LabelParseError, looks_like_refusal, parse_enum_label

DEFAULT_MAX_TOKENS = 16
DEFAULT_TIMEOUT_S = 120.0


class AnthropicAdapter(ModelAdapter):
    model_key = "claude_sonnet5"
    native_probability = False  # spec §12

    def __init__(
        self,
        model_id: str = "claude-sonnet-5",
        *,
        client: Any | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        thinking_disabled: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model_id_requested = model_id
        self.cfg = fp.load_config()
        self.mode = fp.mode_for(model_id, self.cfg)
        self.prompt_hash = fp.prompt_hash(self.mode, self.cfg)
        self.max_tokens = max_tokens
        self.thinking_disabled = thinking_disabled
        self._client = client or Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"], timeout=timeout_s, max_retries=0)

    def verify(self, *, example_id: str, claim: str, evidence: str) -> DecisionResult:
        prompt = fp.render(claim, evidence, self.mode, self.cfg)
        kw: dict[str, Any] = {
            "model": self.model_id_requested,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # anthropic 1.6.0 removed `temperature` from messages.create entirely,
        # so determinism cannot be requested here either. The provider default
        # is used and recorded, which matches the OpenAI side (errata E8).
        if self.thinking_disabled:
            kw["thinking"] = {"type": "disabled"}
        # No tools on any path (spec §7.4).
        started = time.perf_counter()
        try:
            resp = self._client.messages.create(**kw)
        except Exception as exc:
            return self._failure(example_id, exc, started)

        latency_ms = (time.perf_counter() - started) * 1000.0
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = getattr(resp, "usage", None)

        def _result(**kw2: Any) -> DecisionResult:
            base = dict(
                example_id=example_id, model_key=self.model_key,
                model_id_requested=self.model_id_requested,
                model_id_reported=getattr(resp, "model", None),
                p_supported=None,  # §12: never a native probability
                native_probability=False,
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                provider_cost_usd=None, latency_ms=latency_ms, attempts=1,
                raw_response_sha256=sha256_of_payload({"content": text}),
            )
            base.update(kw2)
            return DecisionResult(**base)

        if getattr(resp, "stop_reason", None) == "refusal" or looks_like_refusal(text):
            return _result(label=None, status=STATUS_REFUSAL,
                           error_type="refusal", error_message=text[:300])
        try:
            label = parse_enum_label(text)
        except LabelParseError as exc:
            return _result(label=None, status=STATUS_FORMAT_ERROR,
                           error_type="LabelParseError", error_message=str(exc)[:300])

        return _result(label=label, status=STATUS_OK, extra={
            "mode": self.mode,
            "temperature": "not_settable_in_sdk_1.6",
            "thinking": "disabled" if self.thinking_disabled else "provider_default",
            "stop_reason": getattr(resp, "stop_reason", None),
            "raw_text": text[:200],
            "calibration": "N/A (spec §12: no native token logprobs)",
        })

    def _failure(self, example_id: str, exc: Exception, started: float) -> DecisionResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        msg = str(exc)
        if isinstance(exc, RateLimitError):
            status = STATUS_RATE_LIMITED_EXHAUSTED
        elif isinstance(exc, APITimeoutError):
            status = STATUS_TIMEOUT
        elif isinstance(exc, BadRequestError):
            low = msg.lower()
            status = (STATUS_CONTEXT_INCOMPATIBLE
                      if any(k in low for k in ("too long", "context", "max_tokens"))
                      else STATUS_PROVIDER_ERROR)
        elif isinstance(exc, (APIConnectionError, InternalServerError)):
            status = STATUS_PROVIDER_ERROR
        else:
            status = STATUS_UNKNOWN_ERROR
        return DecisionResult(
            example_id=example_id, model_key=self.model_key,
            model_id_requested=self.model_id_requested, model_id_reported=None,
            label=None, p_supported=None, native_probability=False,
            input_tokens=None, output_tokens=None, provider_cost_usd=None,
            latency_ms=latency_ms, attempts=1, status=status,
            error_type=type(exc).__name__, error_message=msg[:500],
        )

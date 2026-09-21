"""Jev adapter (spec §7.1, §10, §12; errata E1, E4, E9).

The request and response shapes here were confirmed against typesafe-sdk 0.7.0
and the live API on 2026-09-19; see docs/reference/jev-api.md. The spec's §10
snippet is labelled illustrative, so this module is the authority on what we
actually send and read.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeBadRequestError,
    TypeSafeError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from .base import (
    STATUS_FORMAT_ERROR,
    STATUS_OK,
    STATUS_PROVIDER_ERROR,
    STATUS_RATE_LIMITED_EXHAUSTED,
    STATUS_TIMEOUT,
    STATUS_UNKNOWN_ERROR,
    DecisionResult,
    ModelAdapter,
    sha256_of_payload,
)

# Spec §7.1 pins the version. The SDK default is the moving alias `jev-latest`
# (typesafe_sdk.constants.DEFAULT_MODEL), so the model is ALWAYS passed
# explicitly; errata E9 forbids silently benchmarking an aliased model.
JEV_MODEL_ID = "jev-1.13.0"

# The SDK ships RetryPolicy(max_retries=2) and retries on its own. That would
# make `attempts` wrong and hide capacity failures, so retries are disabled here
# and the runner owns the §17 policy (errata E17).
NO_SDK_RETRY = RetryPolicy(max_retries=0)

# The SDK default timeout is 10s, too low for long AggreFact evidence.
DEFAULT_TIMEOUT_S = 120.0

PROMPT_PATH = Path(__file__).resolve().parents[2] / "configs" / "prompts" / "jev_grounding_v1.json"


def load_prompt(path: Path = PROMPT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def prompt_hash(prompt: dict[str, Any]) -> str:
    """Hash exactly the fields that reach the API (§16: freeze and record)."""
    return sha256_of_payload({
        "instructions": prompt["instructions"],
        "criteria": prompt["criteria"],
        "question_name": prompt["question_name"],
    })


class JevAdapter(ModelAdapter):
    model_key = "jev"
    # Errata E4: probabilities["supported"] is a native model probability and
    # is the only thing calibration may use. `confidence` is logged as a
    # secondary field in `extra`.
    native_probability = True

    def __init__(
        self,
        *,
        client: TypeSafeClient | None = None,
        model_id: str = JEV_MODEL_ID,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        prompt: dict[str, Any] | None = None,
    ) -> None:
        self.model_id_requested = model_id
        self.prompt = prompt or load_prompt()
        self.prompt_hash = prompt_hash(self.prompt)
        self._label_map = self.prompt["label_map"]
        self._owns_client = client is None
        self._client = client or TypeSafeClient(timeout=timeout_s, retry=NO_SDK_RETRY)
        self._questions = {
            self.prompt["question_name"]: Choice(
                instructions=self.prompt["instructions"],
                criteria=dict(self.prompt["criteria"]),
            )
        }

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "JevAdapter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def verify(self, *, example_id: str, claim: str, evidence: str) -> DecisionResult:
        qname = self.prompt["question_name"]
        started = time.perf_counter()
        try:
            response = self._client.system_one(
                state={"claim": claim, "evidence": evidence},
                questions=self._questions,
                model=self.model_id_requested,
            )
        except Exception as exc:  # normalized below; the runner decides on retry
            return self._failure(example_id, exc, started)

        latency_ms = (time.perf_counter() - started) * 1000.0
        answer = response.answers.get(qname)

        if answer is None or not hasattr(answer, "probabilities"):
            return DecisionResult(
                example_id=example_id, model_key=self.model_key,
                model_id_requested=self.model_id_requested,
                model_id_reported=getattr(response, "model", None),
                label=None, p_supported=None, native_probability=self.native_probability,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                provider_cost_usd=None, latency_ms=latency_ms, attempts=1,
                status=STATUS_FORMAT_ERROR,
                error_type="missing_choice_answer",
                error_message=f"no Choice answer for question {qname!r}",
            )

        probs = dict(answer.probabilities)
        label = self._label_map.get(answer.choice)
        if label is None:
            return DecisionResult(
                example_id=example_id, model_key=self.model_key,
                model_id_requested=self.model_id_requested,
                model_id_reported=response.model,
                label=None, p_supported=None, native_probability=self.native_probability,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                provider_cost_usd=None, latency_ms=latency_ms, attempts=1,
                status=STATUS_FORMAT_ERROR,
                error_type="unmapped_choice",
                error_message=f"choice {answer.choice!r} is not in label_map",
            )

        return DecisionResult(
            example_id=example_id,
            model_key=self.model_key,
            model_id_requested=self.model_id_requested,
            model_id_reported=response.model,
            label=label,
            p_supported=probs.get("supported"),
            native_probability=self.native_probability,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            provider_cost_usd=None,  # priced later from configs/pricing_<date>.yaml
            latency_ms=latency_ms,
            attempts=1,
            status=STATUS_OK,
            raw_response_sha256=sha256_of_payload(response.model_dump(mode="json")),
            extra={
                # Errata E4: secondary only, never a stand-in for p_supported.
                "jev_confidence": answer.confidence,
                "p_not_supported": probs.get("not_supported"),
                "probabilities_sum": round(sum(probs.values()), 10),
                "request_id": getattr(response, "request_id", None),
                "raw_choice": answer.choice,
            },
        )

    def _failure(self, example_id: str, exc: Exception, started: float) -> DecisionResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        status = _status_for(exc)
        return DecisionResult(
            example_id=example_id, model_key=self.model_key,
            model_id_requested=self.model_id_requested, model_id_reported=None,
            label=None, p_supported=None, native_probability=self.native_probability,
            input_tokens=None, output_tokens=None, provider_cost_usd=None,
            latency_ms=latency_ms, attempts=1, status=status,
            error_type=type(exc).__name__, error_message=str(exc)[:500],
        )


def _status_for(exc: Exception) -> str:
    """Map an SDK exception onto the §27 status enum.

    Errata E17: a rate limit is OUR capacity problem, so it gets its own status
    and is resumed by the runner rather than scored as a wrong answer.
    """
    if isinstance(exc, TypeSafeRateLimitError):
        return STATUS_RATE_LIMITED_EXHAUSTED
    if isinstance(exc, TypeSafeAPITimeoutError):
        return STATUS_TIMEOUT
    if isinstance(exc, (TypeSafeAPIConnectionError, TypeSafeInternalServerError)):
        return STATUS_PROVIDER_ERROR
    if isinstance(exc, TypeSafeBadRequestError):
        # Context-limit rejections surface here; §14 forbids silent truncation,
        # so the runner inspects the message and may reclassify to
        # context_incompatible. Defaulting to provider_error stays conservative.
        return STATUS_PROVIDER_ERROR
    if isinstance(exc, TypeSafeError):
        return STATUS_PROVIDER_ERROR
    return STATUS_UNKNOWN_ERROR

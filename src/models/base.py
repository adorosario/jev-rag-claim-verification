"""Normalized adapter contract shared by every model in the benchmark.

Spec §27 defines `DecisionResult`; every adapter returns one, and nothing else.
The runner and the metrics layer only ever see this shape, so a provider quirk
can never leak into the analysis.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Final

# Spec §27 status enum. `ok` is the only value that carries a usable decision.
STATUS_OK: Final = "ok"
STATUS_FORMAT_ERROR: Final = "format_error"
STATUS_TIMEOUT: Final = "timeout"
STATUS_RATE_LIMITED_EXHAUSTED: Final = "rate_limited_exhausted"
STATUS_PROVIDER_ERROR: Final = "provider_error"
STATUS_CONTEXT_INCOMPATIBLE: Final = "context_incompatible"
STATUS_REFUSAL: Final = "refusal"
STATUS_UNKNOWN_ERROR: Final = "unknown_error"

VALID_STATUSES: Final[frozenset[str]] = frozenset({
    STATUS_OK,
    STATUS_FORMAT_ERROR,
    STATUS_TIMEOUT,
    STATUS_RATE_LIMITED_EXHAUSTED,
    STATUS_PROVIDER_ERROR,
    STATUS_CONTEXT_INCOMPATIBLE,
    STATUS_REFUSAL,
    STATUS_UNKNOWN_ERROR,
})

# The two benchmark labels. Adapters normalize to these exact strings.
LABEL_SUPPORTED: Final = "SUPPORTED"
LABEL_NOT_SUPPORTED: Final = "NOT_SUPPORTED"
VALID_LABELS: Final[frozenset[str]] = frozenset({LABEL_SUPPORTED, LABEL_NOT_SUPPORTED})

# Errata E17: capacity failures are *our* rate limits, not a model defect. The
# runner resumes them and the report gives them their own column; operational
# BAcc must not count them as wrong.
CAPACITY_STATUSES: Final[frozenset[str]] = frozenset({STATUS_RATE_LIMITED_EXHAUSTED})

# Errata E17: statuses that ARE the model's fault and count as wrong answers in
# operational BAcc.
MODEL_FAILURE_STATUSES: Final[frozenset[str]] = frozenset({
    STATUS_FORMAT_ERROR,
    STATUS_REFUSAL,
    STATUS_PROVIDER_ERROR,
    STATUS_TIMEOUT,
    STATUS_UNKNOWN_ERROR,
})


@dataclass
class DecisionResult:
    """Spec §27. One model's verdict on one example.

    `p_supported` is only comparable across models when `native_probability`
    is True. Spec §12 and errata E4 require that distinction to survive all the
    way into the calibration tables, so it is stored per row, not per model.
    """

    example_id: str
    model_key: str
    model_id_reported: str | None

    label: str | None
    p_supported: float | None
    native_probability: bool

    input_tokens: int | None
    output_tokens: int | None
    provider_cost_usd: float | None

    latency_ms: float
    attempts: int
    status: str

    raw_response_sha256: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    # Not in the §27 dataclass, but required elsewhere in the spec and carried
    # here so the runner never has to reach back into a provider object.
    model_id_requested: str | None = None
    # Jev's `confidence`, OpenAI's logprob availability, over-limit flags, etc.
    # Errata E4 keeps `confidence` strictly secondary, so it lives here rather
    # than being mistaken for a probability.
    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid status {self.status!r}")
        if self.label is not None and self.label not in VALID_LABELS:
            raise ValueError(f"invalid label {self.label!r}")
        if self.p_supported is not None and not 0.0 <= self.p_supported <= 1.0:
            raise ValueError(f"p_supported out of range: {self.p_supported!r}")
        if self.status == STATUS_OK and self.label is None:
            raise ValueError("status 'ok' requires a label")
        # A non-native probability must never be silently treated as calibrated.
        if self.p_supported is not None and self.native_probability is None:
            raise ValueError("native_probability must be set when p_supported is present")

    @property
    def is_capacity_failure(self) -> bool:
        """Errata E17: our rate limit, resumed and reported separately."""
        return self.status in CAPACITY_STATUSES

    @property
    def is_model_failure(self) -> bool:
        """Errata E17: the model's fault, counted as wrong in operational BAcc."""
        return self.status in MODEL_FAILURE_STATUSES

    def to_jsonl_row(self, *, run_id: str, dataset: str, prompt_hash: str,
                     config_hash: str) -> dict[str, Any]:
        """Spec §28 prediction row. Gold labels are joined later (§16)."""
        return {
            "run_id": run_id,
            "example_id": self.example_id,
            "dataset": dataset,
            "model_key": self.model_key,
            "model_id_requested": self.model_id_requested,
            "model_id_reported": self.model_id_reported,
            "label": self.label,
            "p_supported": self.p_supported,
            "native_probability": self.native_probability,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.provider_cost_usd,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "status": self.status,
            "prompt_hash": prompt_hash,
            "config_hash": config_hash,
            "raw_response_sha256": self.raw_response_sha256,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "extra": self.extra,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_of_payload(payload: Any) -> str:
    """Stable sha256 over a JSON-serializable payload, for audit trails."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ModelAdapter:
    """Interface every adapter implements.

    Adapters do not retry; the runner owns the §17 retry policy so that
    `attempts` is accurate and capacity failures stay distinguishable from
    model failures (errata E17).
    """

    model_key: str
    model_id_requested: str
    native_probability: bool

    def verify(self, *, example_id: str, claim: str, evidence: str) -> DecisionResult:
        raise NotImplementedError

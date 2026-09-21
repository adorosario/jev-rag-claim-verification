"""MiniCheck-Flan-T5-Large adapter (spec §7.6, §12; errata E3, E5).

This is a faithful port of the published inference code shipped with the model
(`minicheck_web/inference.py`, MIT), NOT a reimplementation. Errata E3 requires
reproducing the published per-dataset BAcc, so every detail that could move a
number is copied deliberately:

* input is ``"predict: " + doc + <eos> + claim``;
* the document is split into ~500-token chunks on sentence boundaries;
* the claim is repeated once per chunk and is never split (see the upstream
  note: AggreFact-CNN's multi-sentence summaries are scored whole);
* the supported/unsupported logits are read at token ids ``[3, 209]`` and
  softmaxed;
* chunk scores are aggregated with **max**, which is the published default.

Two consequences worth stating plainly, because they affect fairness claims:

1. MiniCheck's chunking is intrinsic to the published system, so it effectively
   never hits a context limit. Spec §14 forbids US adding chunking to the other
   models, so MiniCheck is the only model that can never be excluded for
   context. That asymmetry is real and must be disclosed, not hidden.
2. Its probability is a max over chunks, which is a native probability with
   particular semantics; it is not the same object as Jev's single decision
   probability. Both are native, so both are eligible for Table 3, but the
   difference belongs in the caption.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .base import (
    STATUS_OK,
    STATUS_UNKNOWN_ERROR,
    DecisionResult,
    ModelAdapter,
)

MODEL_ID = "lytang/MiniCheck-Flan-T5-Large"
# Pinned so a silent upstream update cannot move our numbers (§7, E9).
MODEL_REVISION = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"

CHUNK_SIZE_TOKENS = 500      # upstream default_chunk_size
MAX_INPUT_LENGTH = 512       # upstream max_input_length
LABEL_TOKEN_IDS = [3, 209]   # upstream: logits[:, [3, 209]] -> (unsupported, supported)
DECISION_THRESHOLD = 0.5     # p_supported > 0.5 -> SUPPORTED

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sent_tokenize_with_newlines(text: str) -> list[str]:
    """Split into sentences while preserving newline structure, as upstream does."""
    blocks = text.split("\n")
    out: list[str] = []
    for block in blocks:
        if not block.strip():
            out.append("")
            continue
        out.extend(s for s in _SENT_SPLIT.split(block.strip()) if s)
    return [s for s in out if s != ""]


def chunk_document(doc: str, chunk_size: int = CHUNK_SIZE_TOKENS) -> list[str]:
    """~chunk_size whitespace tokens per chunk, never splitting a sentence."""
    sentences = sent_tokenize_with_newlines(doc)
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and count + n > chunk_size:
            chunks.append(" ".join(current))
            current, count = [sentence], n
        else:
            current.append(sentence)
            count += n
    if current:
        chunks.append(" ".join(current))
    cleaned = [c.replace(" \n ", "\n").strip() for c in chunks]
    return [c for c in cleaned if c] or [doc.strip()[:1] or " "]


class MiniCheckAdapter(ModelAdapter):
    model_key = "minicheck"
    model_id_requested = MODEL_ID
    # Spec §12: MiniCheck exposes a native support probability.
    native_probability = True

    def __init__(self, *, device: str | None = None, batch_size: int = 16,
                 max_input_length: int = MAX_INPUT_LENGTH) -> None:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self._torch = torch
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        # use_safetensors=False is load-bearing: transformers 5 prefers
        # model.safetensors, which this repo does not ship at the pinned
        # revision. Left to its own devices it resolves a PULL-REQUEST ref
        # (refs/pr/2) and downloads from there, which both defeats the pin and
        # times out. Force the pytorch_model.bin that the revision actually has.
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = T5ForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_safetensors=False
        ).to(device).eval()
        # Errata E5: MiniCheck is not an API call, so its hardware is recorded
        # and its latency is reported separately from the API models.
        self.hardware = {
            "device": device,
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        }

    def _score_chunks(self, chunks: list[str], claim: str) -> list[float]:
        torch = self._torch
        eos = self.tokenizer.eos_token
        texts = ["predict: " + eos.join([c, claim]) for c in chunks]
        probs: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(batch, max_length=self.max_input_length,
                                 truncation=True, padding=True,
                                 return_tensors="pt").to(self.device)
            with torch.no_grad():
                decoder_ids = torch.zeros(
                    (enc["input_ids"].size(0), 1), dtype=torch.long, device=self.device)
                logits = self.model(**enc, decoder_input_ids=decoder_ids).logits[:, 0, :]
                pair = logits[:, torch.tensor(LABEL_TOKEN_IDS, device=self.device)]
                p = torch.nn.functional.softmax(pair.float(), dim=-1)[:, 1]
            probs.extend(p.cpu().tolist())
        return probs

    def verify(self, *, example_id: str, claim: str, evidence: str) -> DecisionResult:
        started = time.perf_counter()
        try:
            chunks = chunk_document(evidence)
            per_chunk = self._score_chunks(chunks, claim)
            p_supported = max(per_chunk)  # upstream aggregation
        except Exception as exc:
            return DecisionResult(
                example_id=example_id, model_key=self.model_key,
                model_id_requested=MODEL_ID, model_id_reported=MODEL_ID,
                label=None, p_supported=None, native_probability=True,
                input_tokens=None, output_tokens=None, provider_cost_usd=None,
                latency_ms=(time.perf_counter() - started) * 1000.0, attempts=1,
                status=STATUS_UNKNOWN_ERROR, error_type=type(exc).__name__,
                error_message=str(exc)[:500])

        latency_ms = (time.perf_counter() - started) * 1000.0
        return DecisionResult(
            example_id=example_id, model_key=self.model_key,
            model_id_requested=MODEL_ID,
            model_id_reported=f"{MODEL_ID}@{MODEL_REVISION[:12]}",
            label="SUPPORTED" if p_supported > DECISION_THRESHOLD else "NOT_SUPPORTED",
            p_supported=float(p_supported), native_probability=True,
            # Local model: no provider token accounting and no per-call price.
            input_tokens=None, output_tokens=None, provider_cost_usd=None,
            latency_ms=latency_ms, attempts=1, status=STATUS_OK,
            extra={
                "n_chunks": len(chunks),
                "support_prob_per_chunk": [round(x, 6) for x in per_chunk],
                "aggregation": "max",
                "hardware": self.hardware,
                "note": "chunking is intrinsic to the published system (§14 asymmetry)",
            },
        )


class _Ref:
    """Doc anchor: the upstream file this port follows."""

    SOURCE = "https://huggingface.co/lytang/MiniCheck-Flan-T5-Large/blob/main/minicheck_web/inference.py"

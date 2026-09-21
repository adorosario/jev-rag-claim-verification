"""MiniCheck document-chunking tests (spec §7.6; errata E3).

Chunking is part of the PUBLISHED MiniCheck system, so these assert fidelity to
the upstream recipe rather than to a design of our own. Getting this wrong would
break the E3 requirement to reproduce the published per-dataset BAcc.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.minicheck_adapter import (  # noqa: E402
    CHUNK_SIZE_TOKENS,
    DECISION_THRESHOLD,
    LABEL_TOKEN_IDS,
    MODEL_ID,
    MODEL_REVISION,
    chunk_document,
    sent_tokenize_with_newlines,
)


def test_upstream_constants_are_pinned():
    """These four values decide the numbers; they must not drift silently."""
    assert CHUNK_SIZE_TOKENS == 500
    assert LABEL_TOKEN_IDS == [3, 209]
    assert DECISION_THRESHOLD == 0.5
    assert MODEL_ID == "lytang/MiniCheck-Flan-T5-Large"
    assert len(MODEL_REVISION) == 40  # pinned, not "main"


def test_short_document_is_one_chunk():
    assert len(chunk_document("A single short sentence.")) == 1


def test_long_document_splits_near_the_chunk_size():
    doc = ("The cat sat on the mat. " * 300).strip()
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert all(len(c.split()) <= CHUNK_SIZE_TOKENS + 20 for c in chunks)


def test_chunking_never_splits_a_sentence():
    doc = " ".join(f"Sentence number {i} is here." for i in range(400))
    for chunk in chunk_document(doc):
        assert chunk.strip().endswith(".")


def test_all_content_is_preserved_across_chunks():
    """No sentence may be dropped; a lost sentence silently changes a label."""
    sentences = [f"Fact {i} holds." for i in range(250)]
    doc = " ".join(sentences)
    joined = " ".join(chunk_document(doc))
    for s in sentences:
        assert s in joined


def test_newlines_are_respected_as_boundaries():
    out = sent_tokenize_with_newlines("First one. Second one!\nThird after newline.")
    assert out == ["First one.", "Second one!", "Third after newline."]


def test_empty_chunks_are_removed():
    doc = "Real sentence.\n\n\n   \n\nAnother real one."
    chunks = chunk_document(doc)
    assert all(c.strip() for c in chunks)


def test_degenerate_document_still_yields_a_chunk():
    """An empty doc must not produce an empty batch and crash the run."""
    assert len(chunk_document("")) >= 1
    assert len(chunk_document("   ")) >= 1


def test_chunking_is_deterministic():
    doc = ("Alpha beta gamma delta. " * 200).strip()
    assert chunk_document(doc) == chunk_document(doc)

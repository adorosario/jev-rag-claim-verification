"""Label-parsing tests (spec §36: whitespace, newline, lowercase, explanation, empty)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.label_parsing import (  # noqa: E402
    LabelParseError,
    looks_like_refusal,
    parse_ab_label,
    parse_enum_label,
)


# --- the substring trap ---------------------------------------------------

def test_not_supported_is_not_read_as_supported():
    """The single most dangerous parsing bug in this benchmark."""
    assert parse_enum_label("NOT_SUPPORTED") == "NOT_SUPPORTED"
    assert parse_enum_label("NOT SUPPORTED") == "NOT_SUPPORTED"
    assert parse_enum_label("not_supported") == "NOT_SUPPORTED"
    assert parse_enum_label("Not-Supported") == "NOT_SUPPORTED"


def test_supported_is_still_parsed():
    assert parse_enum_label("SUPPORTED") == "SUPPORTED"


# --- spec §36 required cases ---------------------------------------------

def test_leading_whitespace():
    assert parse_enum_label("   SUPPORTED") == "SUPPORTED"
    assert parse_enum_label("\t NOT_SUPPORTED") == "NOT_SUPPORTED"


def test_newlines():
    assert parse_enum_label("\nSUPPORTED\n") == "SUPPORTED"
    assert parse_enum_label("\n\nNOT_SUPPORTED\n") == "NOT_SUPPORTED"


def test_lowercase():
    assert parse_enum_label("supported") == "SUPPORTED"


def test_unexpected_explanation():
    assert parse_enum_label("NOT_SUPPORTED. The evidence omits the date.") == "NOT_SUPPORTED"
    assert parse_enum_label("Answer: SUPPORTED") == "SUPPORTED"


def test_empty_response_raises():
    for bad in ("", "   ", "\n", None):
        with pytest.raises(LabelParseError):
            parse_enum_label(bad)


# --- ambiguity is an error, never a guess --------------------------------

def test_reply_containing_both_labels_is_ambiguous():
    with pytest.raises(LabelParseError):
        parse_enum_label("This is SUPPORTED, not NOT_SUPPORTED")


def test_no_label_at_all_raises():
    with pytest.raises(LabelParseError):
        parse_enum_label("I think the claim is fine.")


def test_hyphenated_word_does_not_false_match():
    with pytest.raises(LabelParseError):
        parse_enum_label("un-supported")


# --- A/B mode (kept for models that do expose logprobs) ------------------

def test_ab_first_token():
    assert parse_ab_label("A") == "SUPPORTED"
    assert parse_ab_label("B") == "NOT_SUPPORTED"


def test_ab_whitespace_then_letter():
    assert parse_ab_label("  A") == "SUPPORTED"
    assert parse_ab_label("\n B ") == "NOT_SUPPORTED"


def test_ab_lowercase():
    assert parse_ab_label("a") == "SUPPORTED"


def test_ab_with_punctuation_prefix():
    assert parse_ab_label('"B"') == "NOT_SUPPORTED"


def test_ab_empty_raises():
    for bad in ("", "  ", None):
        with pytest.raises(LabelParseError):
            parse_ab_label(bad)


def test_ab_no_letter_raises():
    with pytest.raises(LabelParseError):
        parse_ab_label("maybe")


# --- refusals are a distinct failure class (E17) -------------------------

def test_refusal_detection():
    assert looks_like_refusal("I cannot help with that request.")
    assert looks_like_refusal("As an AI, I am unable to judge this.")
    assert not looks_like_refusal("SUPPORTED")
    assert not looks_like_refusal("")

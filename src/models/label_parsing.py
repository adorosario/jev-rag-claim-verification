"""Robust label parsing for free-text model replies (spec §36).

The ordering trap: "NOT_SUPPORTED" contains "SUPPORTED" as a substring, so a
naive `"SUPPORTED" in text` check silently turns every negative into a positive.
Every function here tests the negative first, and a test asserts it.

Parsing is deliberately strict about ambiguity: a reply mentioning both labels
is a format_error rather than a guess, because a wrong guess is indistinguishable
from a real model decision once it reaches the metrics.
"""

from __future__ import annotations

import re

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"

# Word-boundary anchored so NOT_SUPPORTED cannot be read as SUPPORTED.
_NOT_RE = re.compile(r"\bNOT[\s_\-]*SUPPORTED\b", re.IGNORECASE)
_SUP_RE = re.compile(r"(?<![\w\-])SUPPORTED\b", re.IGNORECASE)


class LabelParseError(ValueError):
    """The reply could not be resolved to exactly one label."""


def parse_enum_label(text: str | None) -> str:
    """Parse a §11 enum reply. Raises LabelParseError on empty or ambiguous."""
    if text is None:
        raise LabelParseError("empty response (None)")
    stripped = text.strip()
    if not stripped:
        raise LabelParseError("empty response")

    has_not = _NOT_RE.search(stripped) is not None
    # Remove the negative matches before looking for the positive, so the
    # "SUPPORTED" inside "NOT_SUPPORTED" cannot be counted twice.
    without_not = _NOT_RE.sub(" ", stripped)
    has_sup = _SUP_RE.search(without_not) is not None

    if has_not and has_sup:
        raise LabelParseError(f"ambiguous: both labels present in {stripped[:120]!r}")
    if has_not:
        return NOT_SUPPORTED
    if has_sup:
        return SUPPORTED
    raise LabelParseError(f"no label found in {stripped[:120]!r}")


def parse_ab_label(text: str | None) -> str:
    """Parse a §12 single-character A/B reply."""
    if text is None:
        raise LabelParseError("empty response (None)")
    stripped = text.strip()
    if not stripped:
        raise LabelParseError("empty response")
    first = stripped[0].upper()
    if first == "A":
        return SUPPORTED
    if first == "B":
        return NOT_SUPPORTED
    # Some models prefix with punctuation or a quote before the letter.
    m = re.search(r"\b([AB])\b", stripped.upper())
    if m:
        return SUPPORTED if m.group(1) == "A" else NOT_SUPPORTED
    raise LabelParseError(f"no A/B token found in {stripped[:120]!r}")


def looks_like_refusal(text: str | None) -> bool:
    """Spec §17/E17: a refusal is a MODEL failure, distinct from a parse error."""
    if not text:
        return False
    t = text.strip().lower()
    return any(p in t for p in (
        "i cannot", "i can't", "i'm unable", "i am unable",
        "i won't", "as an ai", "i'm sorry, but i can",
    ))

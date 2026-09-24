"""Which checkout this test run is in, and what may be skipped because of it.

Appendix D of the paper says the released package holds "the analysis code and its tests",
and the builder compares this directory against the repository's file by file so that no
test can go missing quietly. That check compares NAMES. It cannot see that a shipped test
reads something the export deliberately withholds, and for a while the package shipped a
test suite that aborted during collection and then failed thirty times over: the paper
source, the builder script and the vendor's agreement are all cited by tests here and none
of the three is republished.

Dropping those tests from the package would make the file-by-file comparison false. So they
ship and they skip, and the skip is keyed on a POSITIVE marker that the builder writes into
the package (``.public-export``) rather than on the absence of the input. That matters: a
skip keyed on "is the file missing?" would turn a deleted paper section or a moved script
into a silent pass in this repository, which is the failure mode these tests exist to catch.
In this repository the marker does not exist, nothing below is skipped, and a missing input
is still an error.

The second gate is credentials rather than content. The gold labels come from the gated
LLM-AggreFact dataset, which needs an HF token the package's .env.example documents and
cannot ship. Those tests skip only in the package and only when no token is present, so a
reader who has accepted the dataset licence runs them in full.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Written by scripts/build_public_export.sh into the package it assembles.
EXPORT_MARKER = REPO_ROOT / ".public-export"

PUBLIC_EXPORT = EXPORT_MARKER.is_file()

_WITHHELD_REASON = (
    "this test reads an input the public package deliberately withholds (the paper "
    "source, the builder script or the vendor's agreement); see Appendix D and the "
    "conflict-of-interest statement. It runs in full in the repository."
)

#: Applies to a test whose inputs are withheld from the released package.
repo_only = pytest.mark.skipif(PUBLIC_EXPORT, reason=_WITHHELD_REASON)

#: Applies to a test that needs the gated LLM-AggreFact parquet.
needs_gold_labels = pytest.mark.skipif(
    PUBLIC_EXPORT and not os.environ.get("HF_TOKEN"),
    reason=(
        "the gold labels come from the gated lytang/LLM-AggreFact dataset. Set HF_TOKEN "
        "in .env, as .env.example describes, and this runs."
    ),
)

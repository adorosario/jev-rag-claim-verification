"""Appendix D's release claims, checked against the script that is supposed to enforce them.

Appendix D names nine categories of artefact in the released package and then says: "After
assembling the package the builder re-checks a named path for each of those categories ...
so the inventory is enforced by the script and not asserted in prose."

That sentence was false for three of the nine. The prompts, the analysis code and the tests
were copied into the export and never verified, scripts/verify/check_public_release.sh named
a prompt file that has never existed in this repository, and the builder itself did not ship,
so a reader invited to check the script against the package could not read the script. No
test covered any of it, which is how a sentence about a gate outlived the gate.

These tests are the mapping from the sentence to the paths, so that editing the prose, the
builder or the release gate without the other two fails here rather than in review.

    docker compose run --rm dev uv run pytest tests/test_public_export.py -v
"""

import json
import re
from pathlib import Path

import pytest

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _export import repo_only  # noqa: E402

# This module reads scripts/build_public_export.sh and Appendix D, and Appendix D
# says in as many words that the builder is the one script the package withholds.
pytestmark = repo_only

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts/build_public_export.sh"
RELEASE_GATE = REPO / "scripts/verify/check_public_release.sh"
APPENDIX = REPO / "arxiv/source/sections/close.tex"

# The private adapter's test is the one file the export withholds, by name, in the builder.
WITHHELD = "tests/test_customgpt_verifier.py"

# Appendix D's sentence, phrase by phrase, against one path the builder must verify for each.
# A phrase that leaves the paper must leave this table, and a category that enters the paper
# has to arrive with a path the script checks.
CATEGORIES = {
    "the raw predictions": "runs/paired-jev-20260921/predictions_jev.jsonl",
    "the prompts": "configs/prompts/binary_v1.txt",
    "the pinned model registry": "configs/models.yaml",
    "the pricing snapshot": "configs/pricing_2026-09-20.yaml",
    "the example-ID list": "runs/explore-jev-dev/jev_dev_sample.jsonl",
    "the leaderboard capture": "data/manifests/llm_aggrefact_published_leaderboard.json",
    "the incident log": "docs/reference/incident-log.md",
    "the analysis code": "scripts/verify/paired_analysis.py",
    "its tests": "tests/test_metrics.py",
}


def _shell_list(text: str, opener: str, closer: str) -> list[str]:
    """The words of one shell list, given the text that opens and closes it."""
    start = text.index(opener) + len(opener)
    return text[start:text.index(closer, start)].split()


@pytest.fixture(scope="module")
def builder() -> str:
    return BUILDER.read_text()


@pytest.fixture(scope="module")
def checked(builder: str) -> list[str]:
    """The paths the builder verifies after assembly, from its check_present call."""
    call = builder.index("\ncheck_present \\")
    block = builder[call:]
    block = block[:block.index("\n\n")]
    return [w for w in block.split() if w not in ("check_present", "\\")]


@pytest.fixture(scope="module")
def release_paths() -> list[str]:
    return _shell_list(RELEASE_GATE.read_text(), "PATHS=(", ")")


@pytest.fixture(scope="module")
def appendix_paragraph() -> str:
    """The Appendix D paragraph that makes the inventory claim."""
    text = APPENDIX.read_text()
    paragraphs = [p for p in re.split(r"\n\s*\n", text)
                  if "build\\_public\\_export.sh" in p and "It holds" in p]
    assert len(paragraphs) == 1, (
        "Appendix D's release paragraph could not be located, so the claim this file tests "
        "cannot be read out of the paper")
    return paragraphs[0]


def test_every_category_appendix_d_names_has_a_path_the_builder_verifies(
        appendix_paragraph, checked):
    """The three that were missing were the prompts, the analysis code and the tests."""
    for phrase, path in CATEGORIES.items():
        assert phrase in appendix_paragraph, (
            f"Appendix D no longer names {phrase!r}; update CATEGORIES so this test keeps "
            "describing the paper")
        assert path in checked, (
            f"Appendix D names {phrase!r} and the builder's check_present list does not "
            f"verify {path}. Either verify it or stop claiming the script enforces it")


def test_the_prompts_and_the_tests_are_compared_directory_against_directory(builder):
    """Naming one file out of a directory lets the rest go missing quietly, which is what
    "the prompts" and "its tests" are: directories, not paths."""
    assert "check_dir_matches configs/prompts" in builder
    assert f"check_dir_matches tests {Path(WITHHELD).name}" in builder


def test_the_reader_side_half_of_the_inventory_check_ships(builder, checked, release_paths):
    """Appendix D says the inventory is enforced by a script rather than asserted in prose.
    A reader with no script has to take that on trust. The builder cannot be that script:
    it names the internal service the export exists to strip, so publishing it would trip
    the builder's own guard. check_public_release.sh carries the same list, names nothing
    private, and checks it from the reader's side, so that is what ships."""
    gate = "scripts/verify/check_public_release.sh"
    assert f"copy {gate}" in builder
    assert gate in checked
    assert gate in release_paths
    assert "copy scripts/build_public_export.sh" not in builder, (
        "the builder names the internal service in its sed line and in its own guard, so "
        "shipping it would publish that name and abort the build")


def test_appendix_d_says_which_script_ships_and_why_the_builder_does_not(appendix_paragraph):
    """The paragraph used to invite a reader to check the builder against the package while
    the builder was not in the package. Whatever it says now has to match what ships."""
    assert "check\\_public\\_release.sh" in appendix_paragraph
    assert "tests/test\\_public\\_export.py" in appendix_paragraph
    assert "the one script we do not publish" in appendix_paragraph


def test_the_builder_runs_the_shipped_suite_inside_the_package(builder, appendix_paragraph):
    """The directory comparison above compares NAMES. It cannot see that a shipped test
    imports a module the copy list omits, and that is exactly what happened: the package
    shipped tests/test_frontier_adapters.py without src/models/anthropic_adapter.py, so
    pytest in the published repository aborted with "Interrupted: 1 error during
    collection" before running one of the other tests, and deselecting that one left
    thirty failures on inputs the package withholds by design. Appendix D offers the
    shipped tests as part of what makes the release checkable, so the builder has to run
    them where a reader would."""
    assert "-e PYTHONPATH=" in builder, (
        "the dev service sets PYTHONPATH=/app and mounts this repository there, so a run "
        "that only changes the working directory imports src/ from here and passes with "
        "a module the package does not contain; the first version of this gate did")
    assert "python -m pytest" in builder, (
        "the builder assembles a test suite it never runs; a suite that cannot be "
        "collected still passes every check this file made before this one")
    assert 'fail "the tests in the export do not pass inside the export"' in builder
    assert "anthropic_adapter.py" in builder, (
        "tests/test_frontier_adapters.py ships and imports this adapter; without it "
        "pytest cannot collect the package at all")
    assert 'cat > "$OUT/.public-export"' in builder, (
        "the shipped tests key their skips on this marker, so a package without it runs "
        "the tests that read the paper source and the builder, and fails")
    assert "copy .gitattributes" in builder, (
        "the published repository has to store the hashed evidence bytes exactly, which "
        "is what tests/test_evidence_integrity.py checks in it")
    # And the paper says so, rather than leaving a reader to discover it.
    for phrase in ("also runs the shipped tests inside the assembled package",
                   "they skip in the package on a marker the builder writes"):
        assert phrase in re.sub(r"\s+", " ", appendix_paragraph), phrase


def test_the_skips_in_the_package_are_keyed_on_a_marker_and_not_on_a_missing_file():
    """A skip written as "if the input is missing" would turn a deleted paper section or a
    moved script into a silent pass HERE, which is the failure mode these tests exist to
    catch. tests/_export.py keys on a file the builder writes into the package instead, so
    nothing is skipped in this repository."""
    from _export import PUBLIC_EXPORT, EXPORT_MARKER
    assert not PUBLIC_EXPORT, (
        f"{EXPORT_MARKER} exists in the repository, so the tests that check the paper "
        "against its evidence are being skipped here")
    helper = (REPO / "tests/_export.py").read_text()
    assert "EXPORT_MARKER.is_file()" in helper


def test_the_release_gate_names_only_paths_this_repository_has(release_paths):
    """check_public_release.sh listed configs/prompts/claim_verification_frontier.txt, which
    has never existed here, so that row could only ever report 404 and no build could fix
    it. A path the builder cannot produce is not a release check, it is a permanent FAIL."""
    missing = [p for p in release_paths if not (REPO / p).exists()]
    assert not missing, (
        "check_public_release.sh fetches these from the public repository and nothing in "
        f"this one can put them there: {missing}")


def test_the_builder_and_the_release_gate_check_the_same_inventory(checked, release_paths):
    """The builder's own comment says to keep the two lists in step. Until now nothing did."""
    only_in_builder = sorted(set(checked) - set(release_paths))
    assert not only_in_builder, (
        "the builder verifies these in the built package and check_public_release.sh never "
        f"asks whether they were published: {only_in_builder}")


def test_the_package_ships_the_article_that_was_published_not_the_draft(
        builder, checked, release_paths):
    """The builder read `copy medium/article.md` for every build of this package. That file
    is the v1 draft. It was never published, and line 197 of it reads "Escalate two thirds
    and you beat it outright", which is the accuracy claim the paper's introduction
    explicitly declines to make and which arxiv/FINDING-cascade-out-of-sample.md lists under
    "Does not survive". So the repository Appendix D sends a reader to shipped, beside the
    paper, an article contradicting it.

    What ships is the article as published, whose cascade operating point the paper
    supersedes in Appendix C and in the incident log. The swap is pinned here, in both
    directions, so it cannot revert quietly.
    """
    assert re.search(r"^copy medium/article-v3\.md$", builder, re.M), (
        "the builder no longer copies the published article")
    assert not re.search(r"^copy medium/article\.md$", builder, re.M), (
        "the builder is shipping the unpublished v1 draft again")
    # Dropping the copy line is not enough. The builder writes into an existing checkout
    # and `copy` replaces only the paths it writes, so the first build after the swap left
    # the v1 draft in the package beside the article that replaced it. It has to be
    # removed, and the removal has to be checked after assembly.
    assert "for withdrawn in medium/article.md; do" in builder, (
        "the builder never removes the v1 draft from an export built over an older one")
    assert 'fail "the unpublished v1 article is in the export"' in builder, (
        "nothing checks the withdrawal after the package is assembled")
    for path in ("medium/article-v3.md", "medium/story-link.md"):
        assert path in checked, f"{path} is copied but never verified after assembly"
        assert path in release_paths, f"{path} is not checked in the published repository"

    v1 = (REPO / "medium/article.md").read_text()
    published = (REPO / "medium/article-v3.md").read_text()
    assert "beat it outright" in v1, (
        "the v1 draft no longer carries the claim this swap exists to keep out of the "
        "package; re-read both files and rewrite this test deliberately")
    assert "beat it outright" not in published, (
        "the published article now carries the accuracy claim the paper refuses, so "
        "shipping it needs a decision rather than a test")
    # And the paper discloses it rather than leaving a reader to find the difference.
    incidents = (REPO / "arxiv/source/sections/close.tex").read_text()
    assert "medium/article-v3.md" in incidents.replace("\\_", "_"), (
        "Appendix C must name the published article whose operating point it supersedes")


def test_the_logprobs_evidence_ships_and_the_paper_cites_it(builder, checked, release_paths):
    """Contribution 3 is that the escalation target cannot supply a probability. The
    evidence for it is runs/model_live_check.json, which carries the provider's verbatim
    400 for the frontier models; it was in neither the export nor the release gate, so the
    claim travelled as a sentence. The Claude-family entry in that file is NOT a probe, and
    Section 8.2 said "the Claude-family models we checked expose nothing either", so the
    file also bounds what may be claimed."""
    path = "runs/model_live_check.json"
    assert f"copy {path}" in builder
    assert path in checked and path in release_paths
    live = json.loads((REPO / path).read_text())["models"]
    assert "logprobs_error" in live["gpt6_astra"] and "400" in live["gpt6_astra"]["logprobs_error"]
    assert "logprobs_error" not in live["claude_sonnet5"], (
        "the Claude entry now carries a probe, so Section 8.2 may describe it as one")
    src = {p: (REPO / "arxiv/source/sections" / p).read_text()
           for p in ("cascade.tex", "setup.tex")}
    for name, text in src.items():
        assert "model\\_live\\_check.json" in text, (
            f"{name} states what the provider exposes and cites no record of it")
    flat = re.sub(r"\s+", " ", src["cascade.tex"])
    assert "the Claude-family models we checked expose nothing" not in flat, (
        "the retracted claim is back: the record holds no probe of that model")
    assert "registry decision" in flat, (
        "Section 8.2 must say what the Claude entry is, since it is not a measurement")


def test_every_verified_path_exists_here_and_is_not_the_withheld_one(checked):
    """A check_present entry for a path this repository cannot produce would abort every
    build, and one for the private adapter's test would contradict the line that deletes it."""
    assert WITHHELD not in checked
    missing = [p for p in checked if not (REPO / p).exists()]
    assert not missing, f"the builder would abort on paths this repository lacks: {missing}"

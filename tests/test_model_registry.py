"""The pinned model registry must describe the runs the paper reports.

Appendix D lists "the pinned model registry" among the artefacts the released package
holds, and sends a reader there to reproduce the frontier runs. For three days it
advertised `max_completion_tokens: 16` for gpt-6-astra: the cap of the DISCARDED first
paired pass, where the model's internal reasoning consumed the whole budget and 84 of 495
claims came back with no answer. The cap was raised to 256 and 4096 and the runs repeated,
but only in the run scripts, which hard-code the value. So the paper's Appendix C said the
cap had been fixed while the registry it points at still published the broken one, and
nothing failed, because no test compared the registry to the runs.

These tests compare them. The run manifests are the record of what was actually sent, and
they are inside the evidence lock, so they are the side that wins.
"""
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
REGISTRY = yaml.safe_load((REPO / "configs/models.yaml").read_text())["models"]

# The reported frontier configurations, and the run whose manifest records what each sent.
REPORTED = {
    "low": "runs/paired-astra-lowthink-20260921-p2b/run_manifest.json",
    "high": "runs/paired-astra-highthink-20260921-p2b/run_manifest.json",
}


def _manifest(path: str) -> dict:
    return json.loads((REPO / path).read_text())["models"]["gpt6_astra"]


@pytest.mark.parametrize("effort,manifest_path", sorted(REPORTED.items()))
def test_the_registry_cap_is_the_cap_the_reported_run_sent(effort, manifest_path):
    declared = REGISTRY["gpt6_astra"]["settings"]["max_completion_tokens"]
    assert isinstance(declared, dict), (
        "the two reported configurations do not share one cap, so a scalar here can "
        "only be right for one of them and was right for neither")
    assert declared[effort] == _manifest(manifest_path)["max_completion_tokens"], (
        f"the registry's {effort}-effort cap is not what that run sent. The manifest is "
        "inside the evidence lock and the registry is not, so the registry is wrong")


def test_the_registry_does_not_still_advertise_the_cap_that_failed():
    """16 is the value that wiped 84 of 495 answers. It must not be the published
    setting for a model the paper reports, whatever else changes around it."""
    declared = REGISTRY["gpt6_astra"]["settings"]["max_completion_tokens"]
    assert 16 not in declared.values(), (
        "the discarded pass's cap is back in the registry for a reported model")


def test_the_incident_is_named_where_a_reader_of_the_registry_will_see_it():
    notes = " ".join(REGISTRY["gpt6_astra"].get("notes", []))
    assert "16" in notes and "84" in notes, (
        "the registry carries notes for errata E9 and E15; the cap that silently wiped "
        "84 answers is at least as much a reader's business as those, and correcting a "
        "published value without saying it was wrong is the quieter version of the bug")


def test_every_reasoning_effort_the_registry_declares_has_a_reported_run():
    declared = set(REGISTRY["gpt6_astra"]["settings"]["max_completion_tokens"])
    assert declared == set(REPORTED), (
        "a cap declared for an effort with no reported run is unverifiable, and an "
        "effort with a run and no cap is the drift this file exists to catch")


def test_the_registry_still_explains_itself_after_the_export_strips_its_comments():
    """The published registry is written by scripts/_export_models_yaml.py, which
    re-serialises this file with yaml.safe_dump. safe_dump drops every `#` comment, so an
    explanation written as a comment is an explanation the reader of the published package
    never sees. Both explanations that matter here once lived in comments: why the cap is
    declared per reasoning effort, and why gpt56_sol is still pinned at the value that
    failed. This test re-serialises the registry the way the export does and reads the
    result, so a comment cannot become load-bearing again.

    It deliberately does not import the export script: the published package ships this
    test and not that script, so the check has to run from the registry alone.
    """
    exported = yaml.safe_dump(
        yaml.safe_load((REPO / "configs/models.yaml").read_text()), sort_keys=False, width=100)
    published = yaml.safe_load(exported)["models"]

    astra = " ".join(published["gpt6_astra"].get("notes", []))
    assert "16" in astra and "84" in astra, "the incident does not survive the export"
    assert "per reasoning effort" in astra, (
        "the published registry declares two caps and, without this note, nothing in it "
        "says why or where the two values came from")

    sol = " ".join(published["gpt56_sol"].get("notes", []))
    assert published["gpt56_sol"]["settings"]["max_completion_tokens"] == 16
    assert "no run reported" in sol.lower() and "untested" in sol.lower(), (
        "gpt56_sol is published pinned at the cap the paper calls a bug; a reader who "
        "cannot see that it was never an arm reads that as the bug, unfixed")

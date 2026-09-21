"""Every branch of the test-split guard (errata E14).

The guard is the only thing standing between an agent and the test split, so
each refusal path gets its own test and the happy path is proven reachable --
a guard that always refuses would pass a weaker suite while being useless.

`test_split_status` is imported under an alias: pytest would otherwise collect
the imported function itself as a test case.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.guard import TestSplitLocked, assert_test_split_unlocked  # noqa: E402
from src.data.guard import test_split_status as split_status  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway repo with one commit and a benchmark-protocol-v1 tag.

    `data/manifests/` is gitignored in the fixture so a test can rewrite the
    manifests without git refusing to switch commits. The guard only ever reads
    them from the working tree, so tracked-ness is irrelevant to what is tested.
    """
    r = tmp_path / "repo"
    (r / "data" / "manifests").mkdir(parents=True)
    _git(r.parent, "init", "-q", "-b", "main", str(r))
    _git(r, "config", "user.email", "test@example.com")
    _git(r, "config", "user.name", "test")
    _git(r, "config", "commit.gpgsign", "false")
    (r / ".gitignore").write_text("data/manifests/\n")
    (r / "README.md").write_text("x")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "init")
    _git(r, "tag", "-a", "benchmark-protocol-v1", "-m", "protocol")
    return r


def _sha(repo: Path, rev: str = "benchmark-protocol-v1^{commit}") -> str:
    return _git(repo, "rev-parse", rev)


def _write_freeze(repo: Path, *, tag="benchmark-protocol-v1", sha=None):
    sha = sha if sha is not None else _sha(repo)
    (repo / "data/manifests/protocol_freeze.json").write_text(
        json.dumps({"tag": tag, "tag_sha": sha}))


def _write_push(repo: Path, *, tag="benchmark-protocol-v1", sha=None):
    sha = sha if sha is not None else _sha(repo)
    (repo / "data/manifests/protocol_push.json").write_text(
        json.dumps({"tag": tag, "ls_remote_sha": sha}))


# --- the happy path must be reachable ------------------------------------

def test_unlocks_when_all_four_conditions_hold(repo):
    _write_freeze(repo)
    _write_push(repo)
    s = split_status(repo)
    assert s.unlocked is True, s.reasons
    assert_test_split_unlocked(repo)  # must not raise


# --- condition 1: freeze manifest ----------------------------------------

def test_refuses_without_freeze_manifest(repo):
    assert not split_status(repo)
    assert "protocol_freeze.json" in split_status(repo).reasons[0]


def test_refuses_when_freeze_has_no_tag(repo):
    (repo / "data/manifests/protocol_freeze.json").write_text(json.dumps({"tag_sha": _sha(repo)}))
    assert "no 'tag'" in " ".join(split_status(repo).reasons)


def test_refuses_when_freeze_sha_is_not_a_sha(repo):
    _write_freeze(repo, sha="not-a-sha")
    assert "40-hex" in " ".join(split_status(repo).reasons)


def test_refuses_when_tag_name_is_not_a_protocol_tag(repo):
    _write_freeze(repo, tag="v1.0.0")
    assert "not a benchmark-protocol-vN tag" in " ".join(split_status(repo).reasons)


def test_refuses_on_malformed_freeze_json(repo):
    (repo / "data/manifests/protocol_freeze.json").write_text("{not json")
    s = split_status(repo)
    assert not s.unlocked and "guard error" in s.reasons[0]


# --- condition 2: push manifest (the on-origin proof) --------------------

def test_refuses_without_push_manifest(repo):
    """The whole point: a local tag is not proof it reached origin."""
    _write_freeze(repo)
    assert "not confirmed on origin" in " ".join(split_status(repo).reasons)


def test_refuses_when_push_sha_is_not_a_sha(repo):
    _write_freeze(repo)
    _write_push(repo, sha="nope")
    assert "40-hex" in " ".join(split_status(repo).reasons)


def test_refuses_when_push_tag_differs_from_freeze_tag(repo):
    _write_freeze(repo)
    _write_push(repo, tag="benchmark-protocol-v2")
    assert "!=" in " ".join(split_status(repo).reasons)


def test_refuses_when_origin_sha_differs_from_frozen_sha(repo):
    _write_freeze(repo)
    _write_push(repo, sha="0" * 40)
    assert "origin sha" in " ".join(split_status(repo).reasons)


# --- condition 3: local tag and HEAD ancestry ----------------------------

def test_refuses_when_local_tag_moved(repo):
    """A retagged commit must not silently unlock."""
    _write_freeze(repo)
    _write_push(repo)
    (repo / "b.txt").write_text("b")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "second")
    _git(repo, "tag", "-d", "benchmark-protocol-v1")
    _git(repo, "tag", "-a", "benchmark-protocol-v1", "-m", "moved")
    assert "expected" in " ".join(split_status(repo).reasons)


def test_refuses_when_head_does_not_contain_the_tag(repo):
    """The protocol was tagged on a commit that is not in HEAD's history.

    Everything else is consistent -- the manifests agree with each other and
    with the local tag -- so this isolates the ancestry check itself.
    """
    _git(repo, "switch", "-q", "-c", "other")
    (repo / "c.txt").write_text("c")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "on other branch")
    _git(repo, "tag", "-a", "benchmark-protocol-v2", "-m", "p2")
    side_sha = _sha(repo, "benchmark-protocol-v2^{commit}")
    _git(repo, "switch", "-q", "main")

    _write_freeze(repo, tag="benchmark-protocol-v2", sha=side_sha)
    _write_push(repo, tag="benchmark-protocol-v2", sha=side_sha)

    s = split_status(repo)
    assert not s.unlocked
    assert "does not contain" in " ".join(s.reasons), s.reasons


def test_refuses_when_tag_missing_locally(repo):
    _write_freeze(repo)
    _write_push(repo)
    _git(repo, "tag", "-d", "benchmark-protocol-v1")
    assert not split_status(repo)


# --- condition 4: a superseding protocol blocks the old one --------------

def test_refuses_while_a_newer_protocol_is_being_prepared(repo):
    """v2 exists locally while the manifests still point at v1."""
    _write_freeze(repo)
    _write_push(repo)
    assert split_status(repo).unlocked is True
    _git(repo, "tag", "-a", "benchmark-protocol-v2", "-m", "next protocol")
    s = split_status(repo)
    assert not s.unlocked
    assert "supersedes" in " ".join(s.reasons)


def test_version_order_not_date_order(repo):
    """v10 must outrank v2 even though v2 is tagged later."""
    _git(repo, "tag", "-a", "benchmark-protocol-v10", "-m", "ten")
    _git(repo, "tag", "-a", "benchmark-protocol-v2", "-m", "two")
    _write_freeze(repo, tag="benchmark-protocol-v10")
    _write_push(repo, tag="benchmark-protocol-v10")
    assert split_status(repo).unlocked is True, split_status(repo).reasons


def test_unrelated_tags_are_ignored(repo):
    _git(repo, "tag", "-a", "results-lock-devrun", "-m", "lock")
    _write_freeze(repo)
    _write_push(repo)
    assert split_status(repo).unlocked is True


# --- structural guarantees ------------------------------------------------

def test_guard_makes_no_network_calls():
    """The container has no GitHub credentials; a network git call would either
    hang or fail in a way the guard must never depend on."""
    from src.data.guard import _git as guard_git
    for forbidden in ("ls-remote", "fetch", "push", "clone", "pull"):
        with pytest.raises(ValueError):
            guard_git(Path("/app"), forbidden)


def test_fails_closed_on_nonexistent_repo(tmp_path):
    assert split_status(tmp_path / "does-not-exist").unlocked is False


def test_assert_raises_with_reasons(repo):
    with pytest.raises(TestSplitLocked) as e:
        assert_test_split_unlocked(repo)
    assert "LOCKED" in str(e.value)

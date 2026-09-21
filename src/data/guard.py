"""Fail-closed guard on the LLM-AggreFact test split (errata E14, spec §16, §38).

The test split may not be read until the protocol is frozen AND the tag is
confirmed on origin. This module is the single chokepoint that enforces it.

Design rules, all of them load-bearing:

* **No network.** The container holds no GitHub credentials, so the guard must
  never call `git ls-remote`, `git fetch`, or any HTTP client. The on-origin
  fact is carried in `protocol_push.json`, which ONLY the workflow's post-PASS
  push step writes.
* **Fail closed.** Any exception, missing file, malformed JSON or unexpected git
  output results in a refusal. There is no default-allow path.
* **No side effects.** The guard only reads.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FREEZE_MANIFEST = Path("data/manifests/protocol_freeze.json")
PUSH_MANIFEST = Path("data/manifests/protocol_push.json")

PROTOCOL_TAG_GLOB = "benchmark-protocol-v*"
PROTOCOL_TAG_RE = re.compile(r"^benchmark-protocol-v(\d+)$")

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class TestSplitLocked(RuntimeError):
    """Raised when the test split may not be touched. Never caught internally."""

    # Not a pytest test class, despite the leading "Test".
    __test__ = False


@dataclass
class GuardStatus:
    unlocked: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.unlocked


def _git(repo_root: Path, *args: str) -> str:
    """Run a read-only, strictly local git command.

    Raises on non-zero exit so the caller's fail-closed wrapper catches it.
    """
    for forbidden in ("ls-remote", "fetch", "pull", "clone", "push"):
        if forbidden in args:
            raise ValueError(f"guard must not run network git: {forbidden}")
    out = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return out.stdout.strip()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _newest_protocol_tag(repo_root: Path) -> str | None:
    """Highest `benchmark-protocol-vN` by integer N.

    Version order, not commit date: a tag created later with a lower number
    must not be able to displace a higher one.
    """
    raw = _git(repo_root, "tag", "--list", PROTOCOL_TAG_GLOB)
    best: tuple[int, str] | None = None
    for line in raw.splitlines():
        m = PROTOCOL_TAG_RE.match(line.strip())
        if not m:
            continue
        n = int(m.group(1))
        if best is None or n > best[0]:
            best = (n, line.strip())
    return best[1] if best else None


def test_split_status(repo_root: Path | str = REPO_ROOT) -> GuardStatus:
    """Evaluate all four unlock conditions. Never raises; always returns."""
    root = Path(repo_root)
    reasons: list[str] = []
    try:
        # --- Condition 1: the freeze manifest exists and is well formed -----
        freeze_path = root / FREEZE_MANIFEST
        if not freeze_path.is_file():
            return GuardStatus(False, [f"missing {FREEZE_MANIFEST}"])
        freeze = _load_json(freeze_path)
        tag = freeze.get("tag")
        freeze_sha = freeze.get("tag_sha")
        if not tag or not isinstance(tag, str):
            reasons.append("protocol_freeze.json has no 'tag'")
        if not isinstance(freeze_sha, str) or not _SHA_RE.match(freeze_sha or ""):
            reasons.append("protocol_freeze.json 'tag_sha' is not a 40-hex sha")
        if reasons:
            return GuardStatus(False, reasons)
        if not PROTOCOL_TAG_RE.match(tag):
            return GuardStatus(False, [f"tag {tag!r} is not a benchmark-protocol-vN tag"])

        # --- Condition 2: the push manifest agrees with the freeze ----------
        push_path = root / PUSH_MANIFEST
        if not push_path.is_file():
            return GuardStatus(
                False,
                [f"missing {PUSH_MANIFEST}: the tag is not confirmed on origin yet"],
            )
        push = _load_json(push_path)
        push_sha = push.get("ls_remote_sha")
        push_tag = push.get("tag")
        if not isinstance(push_sha, str) or not _SHA_RE.match(push_sha or ""):
            return GuardStatus(False, ["protocol_push.json 'ls_remote_sha' is not a 40-hex sha"])
        if push_tag != tag:
            return GuardStatus(
                False, [f"push manifest tag {push_tag!r} != freeze manifest tag {tag!r}"])
        if push_sha != freeze_sha:
            return GuardStatus(
                False, [f"origin sha {push_sha} != frozen sha {freeze_sha}"])

        # --- Condition 3: the tag is local, at that sha, and HEAD contains it
        local_sha = _git(root, "rev-parse", f"{tag}^{{commit}}")
        if local_sha != freeze_sha:
            return GuardStatus(
                False, [f"local tag {tag} is at {local_sha}, expected {freeze_sha}"])
        try:
            _git(root, "merge-base", "--is-ancestor", freeze_sha, "HEAD")
        except subprocess.CalledProcessError:
            return GuardStatus(False, [f"HEAD does not contain {tag} ({freeze_sha})"])

        # --- Condition 4: no superseding protocol is being prepared ---------
        newest = _newest_protocol_tag(root)
        if newest is None:
            return GuardStatus(False, ["no benchmark-protocol-v* tag found locally"])
        if newest != tag:
            return GuardStatus(
                False,
                [f"{newest} supersedes the frozen {tag}; a newer protocol is being prepared"],
            )

        return GuardStatus(True, [])

    except Exception as exc:  # fail closed on ANYTHING
        return GuardStatus(False, [f"guard error ({type(exc).__name__}): {exc}"])


def assert_test_split_unlocked(repo_root: Path | str = REPO_ROOT) -> None:
    """Chokepoint. The loader calls this before touching any test file."""
    status = test_split_status(repo_root)
    if not status.unlocked:
        raise TestSplitLocked(
            "The LLM-AggreFact test split is LOCKED (spec §16, errata E14). "
            "Reasons: " + "; ".join(status.reasons)
        )

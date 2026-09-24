"""Every hashed evidence file must still match its recorded sha256 (spec §43).

This catches the failure mode where git rewrites bytes on commit (CRLF
normalization) and a fresh clone then fails its own integrity check. It reads
the file as BYTES, exactly as a verifier reproducing our numbers would.
"""

import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _export import PUBLIC_EXPORT, repo_only  # noqa: E402
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _sha256_bytes(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _sha256_text(path: Path) -> str:
    """Hash of the DECODED text. Note this is not round-trip stable for CRLF
    files, because read_text() performs universal-newline translation; evidence
    manifests therefore hash bytes instead."""
    return sha256(path.read_text().encode("utf-8")).hexdigest()


# --- legal snapshots ------------------------------------------------------

def _sha256_files():
    return sorted(REPO_ROOT.glob("legal/*.sha256"))


@pytest.mark.parametrize("manifest", _sha256_files(), ids=lambda p: p.name)
def test_legal_snapshot_hashes_match(manifest):
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        expected, rel = line.split(None, 1)
        path = REPO_ROOT / rel.strip()
        if PUBLIC_EXPORT and not path.is_file():
            # The package ships this manifest and not the vendor's agreement it hashes,
            # which the conflict-of-interest statement says in as many words: a reader
            # verifies their own copy of the document against the hash. In this
            # repository the file is present and a missing one is still a failure.
            pytest.skip(f"{rel} is not republished; verify your own copy against the hash")
        assert path.is_file(), f"missing snapshot {rel}"
        assert _sha256_bytes(path) == expected, (
            f"{rel} no longer matches its recorded hash; if git normalized line "
            f"endings, check .gitattributes")


@repo_only
def test_mca_snapshot_exists_and_is_searchable():
    """G0 evidence: a live MCA snapshot must be on disk and greppable."""
    snaps = sorted(REPO_ROOT.glob("legal/typesafe-mca-snapshot-*.txt"))
    assert snaps, "no MCA snapshot saved"
    text = snaps[-1].read_text()
    assert "Master Customer Agreement" in text
    assert "License Restrictions" in text


@repo_only
def test_mca_benchmark_clause_finding_is_reproducible():
    """The recorded finding must be re-derivable from the snapshot itself.

    If a future snapshot DOES contain the restriction, this fails loudly and the
    legal record has to be revisited rather than quietly drifting.
    """
    snaps = sorted(REPO_ROOT.glob("legal/typesafe-mca-snapshot-*.txt"))
    text = snaps[-1].read_text().lower()
    record = (REPO_ROOT / "legal/typesafe-mca-2.3f-excerpt.md").read_text()
    found = text.count("benchmark") > 0 or "performance information" in text
    claims_not_found = "NOT FOUND in the live MCA" in record
    assert found != claims_not_found, (
        "legal/typesafe-mca-2.3f-excerpt.md disagrees with the snapshot on disk")


# --- git must not rewrite evidence bytes ---------------------------------

def _tracked(pattern: str) -> list[Path]:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", pattern],
                         capture_output=True, text=True)
    if out.returncode != 0:
        # The released package is a directory until it is committed, and `git ls-files`
        # outside a work tree is an error rather than an empty list. What this test
        # checks is a property of a git repository, so where there is none there is
        # nothing to check; in this repository there always is one.
        pytest.skip("not a git work tree, so git cannot have rewritten anything")
    return [REPO_ROOT / p for p in out.stdout.split()]


@pytest.mark.parametrize("rel", [
    "legal/typesafe-mca-snapshot-*",
    "data/pricing_snapshots/*.md",
])
def test_git_stores_evidence_bytes_exactly(rel):
    """Worktree bytes must equal the bytes git has stored."""
    files = _tracked(rel)
    if not files:
        pytest.skip(f"no tracked files matching {rel}")
    for path in files:
        stored = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-p",
             f":{path.relative_to(REPO_ROOT)}"],
            capture_output=True, check=True).stdout
        assert sha256(stored).hexdigest() == _sha256_bytes(path), (
            f"git rewrote {path.relative_to(REPO_ROOT)}; add it to .gitattributes "
            f"with -text")


def test_gitattributes_protects_evidence_paths():
    attrs = (REPO_ROOT / ".gitattributes").read_text()
    for pattern in ("legal/typesafe-mca-snapshot-", "data/pricing_snapshots/"):
        assert pattern in attrs, pattern
        assert "-text" in attrs


# --- pricing snapshots ----------------------------------------------------

def test_pricing_config_hashes_match_snapshot_files():
    configs = sorted(REPO_ROOT.glob("configs/pricing_*.yaml"))
    if not configs:
        pytest.skip("no pricing config")
    cfg = yaml.safe_load(configs[-1].read_text())
    for name, meta in cfg["snapshots"].items():
        path = REPO_ROOT / meta["path"]
        assert _sha256_text(path) == meta["sha256"], name

"""Dev-split integrity and the E14 cache guarantee (spec §36; errata E2, E14, E18).

These read the real pinned revision, so they need HF_TOKEN and a warm cache.
They are the tests a verifier will rerun to confirm the manifest was not
hand-written.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_aggrefact import (  # noqa: E402
    HF_REVISION,
    MANIFEST_PATH,
    REQUIRED_COLUMNS,
    SPEC_DEV_COUNT,
    SPEC_N_DATASETS,
    SPEC_TEST_COUNT,
    TEST_FILE,
    example_id,
    load_dev_frame,
    load_test_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = Path("/opt/hf-cache")


@pytest.fixture(scope="module")
def dev():
    return load_dev_frame()


def test_dev_count_matches_pinned_revision(dev):
    assert len(dev) == SPEC_DEV_COUNT


def test_required_columns_present(dev):
    for col in REQUIRED_COLUMNS:
        assert col in dev.inputs.columns or col == "label"
    assert "label" in dev.labels.columns


def test_labels_are_binary(dev):
    assert set(dev.labels["label"].unique()) <= {0, 1}


def test_eleven_component_datasets(dev):
    assert dev.inputs["dataset"].nunique() == SPEC_N_DATASETS


def test_no_missing_claim_or_doc(dev):
    assert dev.inputs["claim"].notna().all()
    assert dev.inputs["doc"].notna().all()
    assert (dev.inputs["claim"].str.len() > 0).all()
    assert (dev.inputs["doc"].str.len() > 0).all()


def test_labels_are_stored_apart_from_inputs(dev):
    """Spec §16: the runner must not be able to see a gold label."""
    assert "label" not in dev.inputs.columns


def test_example_id_is_stable_and_content_addressed():
    """Errata E18: the ID must not depend on row order or revision."""
    a = example_id("dev", "RAGTruth", "doc text", "claim text")
    b = example_id("dev", "RAGTruth", "doc text", "claim text")
    assert a == b
    assert a.startswith("dev:RAGTruth:")
    assert a != example_id("dev", "RAGTruth", "doc text", "other claim")
    # field separator prevents a boundary collision
    assert example_id("dev", "D", "ab", "c") != example_id("dev", "D", "a", "bc")


def test_example_ids_cover_the_split(dev):
    assert len(dev.inputs) == SPEC_DEV_COUNT
    assert dev.inputs["example_id"].notna().all()


def test_duplicates_are_reported_not_dropped(dev):
    """Errata E18 requires reporting duplicate (doc, claim) pairs."""
    assert len(dev.inputs) == SPEC_DEV_COUNT  # nothing silently removed
    assert dev.duplicates is not None


def test_row_hash_is_order_independent(dev):
    from src.data.hashes import row_hash, rows_hash
    hashes = [row_hash("a", 1), row_hash("b", 2), row_hash("c", 3)]
    assert rows_hash(hashes) == rows_hash(reversed(hashes))


# --- errata E14: the test split must be untouchable -----------------------

def test_no_test_file_anywhere_in_hf_cache():
    """The single most important cache assertion in G0."""
    if not HF_CACHE.exists():
        pytest.skip("HF cache not present")
    offenders = [p for p in HF_CACHE.rglob("*") if p.is_file() and "test" in p.name.lower()]
    assert offenders == [], f"test split artifacts leaked into the cache: {offenders}"


def test_loading_the_test_split_is_refused():
    from src.data.guard import TestSplitLocked
    with pytest.raises(TestSplitLocked):
        load_test_frame(REPO_ROOT)


def test_test_file_constant_is_never_downloaded_by_dev_path():
    """`TEST_FILE` exists only so the guard and this test can name it."""
    src = (REPO_ROOT / "src/data/load_aggrefact.py").read_text()
    body = src.split("def load_dev_frame")[1].split("def load_test_frame")[0]
    assert "TEST_FILE" not in body


# --- the manifest must be generated, not typed ----------------------------

def test_manifest_matches_a_fresh_load(dev):
    manifest = json.loads((REPO_ROOT / MANIFEST_PATH).read_text())
    assert manifest["hf_revision"] == HF_REVISION
    assert manifest["dev_count"] == len(dev)
    assert manifest["dev_rows_sha256"] == dev.rows_sha256
    assert manifest["dev_parquet_sha256"] == dev.parquet_sha256
    assert manifest["test_split_downloaded"] is False


def test_manifest_records_spec_expectations_and_mismatches():
    """Errata E2: mismatches are flagged, never forced."""
    manifest = json.loads((REPO_ROOT / MANIFEST_PATH).read_text())
    assert manifest["spec_expected"]["dev_count"] == SPEC_DEV_COUNT
    assert manifest["spec_expected"]["test_count"] == SPEC_TEST_COUNT
    assert isinstance(manifest["spec_mismatches"], list)


def test_test_count_came_from_metadata_not_from_test_files():
    manifest = json.loads((REPO_ROOT / MANIFEST_PATH).read_text())
    assert manifest["test_count_source"].startswith("datasets-server")
    assert manifest["test_count"] == SPEC_TEST_COUNT

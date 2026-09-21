"""LLM-AggreFact loader, dev-only by default (spec §6.1, §16; errata E2, E14, E18).

Hard rules enforced here:

* **Dev files only.** `hf_hub_download` is called with an explicit filename.
  `load_dataset` is never used, because it resolves and fetches every split and
  would place the test parquet in the cache where an agent could read it (E14).
* **The test count never comes from a test file.** It is read from the
  datasets-server `/size` endpoint, which is metadata, not data.
* **The test split goes through the guard.** `load_test_frame` calls
  `assert_test_split_unlocked` before it does anything else.
* **Labels are stored apart from model inputs** (§16), so the runner cannot see
  a gold label even by accident.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .guard import assert_test_split_unlocked
from .hashes import row_hash, rows_hash, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]

HF_REPO_ID = "lytang/LLM-AggreFact"
# Pinned 2026-09-19. Re-pinning is a protocol change, not a maintenance task.
HF_REVISION = "981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4"

DEV_FILE = "data/dev-00000-of-00001.parquet"
# Named only so the guard and the cache test can assert it is absent. It is
# never passed to hf_hub_download outside load_test_frame().
TEST_FILE = "data/test-00000-of-00001.parquet"

# Spec §6.1. Errata E2: assert against the pinned revision and FLAG any
# mismatch; never quietly force the spec's numbers onto reality.
SPEC_DEV_COUNT = 30_420
SPEC_TEST_COUNT = 29_320
SPEC_N_DATASETS = 11
REQUIRED_COLUMNS = ("dataset", "doc", "claim", "label")

MANIFEST_PATH = Path("data/manifests/aggrefact_revision.json")
SIZE_ENDPOINT = "https://datasets-server.huggingface.co/size"


@dataclass
class DevSplit:
    """Model inputs and gold labels, deliberately held apart (§16)."""

    inputs: pd.DataFrame   # example_id, dataset, doc, claim, row_index
    labels: pd.DataFrame   # example_id, label
    parquet_path: Path
    parquet_sha256: str
    rows_sha256: str
    duplicates: pd.DataFrame

    def __len__(self) -> int:
        return len(self.inputs)


def example_id(split: str, dataset: str, doc: str, claim: str) -> str:
    """Errata E18: a stable ID, not a revision-dependent row index."""
    return f"{split}:{dataset}:{row_hash(doc, claim)[:16]}"


def _download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        revision=HF_REVISION,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    ))


def _to_frame(path: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, str, pd.DataFrame]:
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{split} split is missing required columns: {missing}")

    df = df.reset_index(drop=True)
    df["row_index"] = df.index
    df["example_id"] = [
        example_id(split, d, doc, claim)
        for d, doc, claim in zip(df["dataset"], df["doc"], df["claim"], strict=True)
    ]

    bad = sorted(set(df["label"].unique()) - {0, 1})
    if bad:
        raise ValueError(f"{split} split has labels outside {{0,1}}: {bad}")

    # Errata E18: report duplicate (doc, claim) pairs rather than dropping them.
    dup_mask = df.duplicated(subset=["example_id"], keep=False)
    duplicates = df.loc[dup_mask, ["example_id", "dataset", "row_index", "label"]].copy()

    rows_sha = rows_hash(
        row_hash(d, doc, claim, int(lbl))
        for d, doc, claim, lbl in zip(
            df["dataset"], df["doc"], df["claim"], df["label"], strict=True)
    )

    inputs = df[["example_id", "dataset", "doc", "claim", "row_index"]].copy()
    labels = df[["example_id", "label"]].copy()
    return inputs, labels, rows_sha, duplicates


def load_dev_frame() -> DevSplit:
    path = _download(DEV_FILE)
    inputs, labels, rows_sha, duplicates = _to_frame(path, "dev")
    return DevSplit(
        inputs=inputs, labels=labels, parquet_path=path,
        parquet_sha256=sha256_file(path), rows_sha256=rows_sha, duplicates=duplicates,
    )


def load_test_frame(repo_root: Path | str = REPO_ROOT):
    """Refuses until the protocol tag is confirmed on origin (errata E14)."""
    assert_test_split_unlocked(repo_root)
    path = _download(TEST_FILE)
    return _to_frame(path, "test")


def fetch_test_count(timeout: float = 30.0) -> tuple[int | None, str]:
    """Test count from METADATA only -- never from a test file (errata E14)."""
    import httpx

    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.get(SIZE_ENDPOINT, params={"dataset": HF_REPO_ID},
                      headers=headers, timeout=timeout)
        r.raise_for_status()
        for split in r.json().get("size", {}).get("splits", []):
            if split.get("split") == "test":
                return int(split["num_rows"]), "datasets-server/size"
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return None, f"datasets-server unavailable: {type(exc).__name__}: {exc}"
    return None, "datasets-server returned no test split"


def build_manifest(repo_root: Path | str = REPO_ROOT) -> dict:
    root = Path(repo_root)
    dev = load_dev_frame()
    test_count, test_count_source = fetch_test_count()

    n_datasets = int(dev.inputs["dataset"].nunique())
    per_dataset = (dev.inputs.groupby("dataset").size()
                   .sort_index().astype(int).to_dict())

    # Errata E2: flag, do not force.
    mismatches: list[str] = []
    if len(dev) != SPEC_DEV_COUNT:
        mismatches.append(f"dev count {len(dev)} != spec §6.1 {SPEC_DEV_COUNT}")
    if test_count is not None and test_count != SPEC_TEST_COUNT:
        mismatches.append(f"test count {test_count} != spec §6.1 {SPEC_TEST_COUNT}")
    if n_datasets != SPEC_N_DATASETS:
        mismatches.append(f"{n_datasets} component datasets != spec §6.1 {SPEC_N_DATASETS}")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hf_repo_id": HF_REPO_ID,
        "hf_revision": HF_REVISION,
        "dev_file": DEV_FILE,
        "dev_parquet_sha256": dev.parquet_sha256,
        "dev_rows_sha256": dev.rows_sha256,
        "dev_count": len(dev),
        "test_count": test_count,
        "test_count_source": test_count_source,
        "n_component_datasets": n_datasets,
        "per_dataset_dev_counts": per_dataset,
        "label_values": sorted(int(v) for v in dev.labels["label"].unique()),
        "duplicate_example_ids": int(dev.duplicates["example_id"].nunique()),
        "duplicate_rows": int(len(dev.duplicates)),
        # A (doc, claim) pair carrying both labels is irreducible: no model can
        # score better than chance on it. Small, but it caps achievable BAcc and
        # belongs in Limitations rather than being discovered by a reviewer.
        "duplicate_example_ids_with_conflicting_labels": int(
            (dev.duplicates.groupby("example_id")["label"].nunique() > 1).sum()
        ),
        "spec_expected": {
            "dev_count": SPEC_DEV_COUNT,
            "test_count": SPEC_TEST_COUNT,
            "n_component_datasets": SPEC_N_DATASETS,
        },
        "spec_mismatches": mismatches,
        "example_id_scheme": "split:dataset:sha256(doc␟claim)[:16] (errata E18)",
        "test_split_downloaded": False,
    }
    out = root / MANIFEST_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    m = build_manifest()
    print(json.dumps({k: v for k, v in m.items() if k != "per_dataset_dev_counts"}, indent=2))
    print("\nper-dataset dev counts:")
    for k, v in m["per_dataset_dev_counts"].items():
        print(f"  {k:<28} {v:>6}")

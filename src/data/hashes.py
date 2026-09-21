"""Stable content hashes for datasets and manifests (spec §6.1 redistribution, §43).

Row hashes are order-independent: the same content must hash the same whether
or not the loader, a future datasets version, or a shuffle changed row order.
Anything order-dependent would produce spurious "dataset changed" alarms.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

# U+241F SYMBOL FOR UNIT SEPARATOR: a visible character that cannot occur in
# benchmark text, so "ab"+"c" and "a"+"bc" can never collide.
FIELD_SEP = "␟"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def join_fields(*parts: object) -> str:
    return FIELD_SEP.join("" if p is None else str(p) for p in parts)


def row_hash(*parts: object) -> str:
    return sha256_text(join_fields(*parts))


def rows_hash(row_hashes: Iterable[str]) -> str:
    """Order-independent hash over a collection of per-row hashes."""
    return sha256_text("\n".join(sorted(row_hashes)))

"""Shared prompt construction for the general-purpose LLMs (spec §11, §12).

Spec rule 7 requires the same semantic prompt for every model, so all frontier
adapters render from this one module and the resulting prompt hash is recorded
per prediction row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import sha256_of_payload

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"
MODES_PATH = PROMPTS_DIR / "frontier_response_modes.json"


def load_config(path: Path = MODES_PATH) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    cfg["_base_prompt"] = (PROMPTS_DIR / cfg["base_prompt_file"]).read_text()
    return cfg


def mode_for(model_id: str, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    return cfg["mode_by_model"].get(model_id, "enum")


def render(claim: str, evidence: str, mode: str, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    if mode not in cfg["modes"]:
        raise ValueError(f"unknown response mode {mode!r}")
    base = cfg["_base_prompt"].replace("{claim}", claim).replace("{evidence}", evidence)
    return base.rstrip() + "\n\n" + cfg["modes"][mode]["suffix"]


def prompt_hash(mode: str, cfg: dict[str, Any] | None = None) -> str:
    """Hash the template and suffix, not a rendered example."""
    cfg = cfg or load_config()
    return sha256_of_payload({
        "base_prompt": cfg["_base_prompt"],
        "suffix": cfg["modes"][mode]["suffix"],
        "mode": mode,
    })

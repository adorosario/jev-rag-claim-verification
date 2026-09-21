"""Pricing snapshots and the priced cost model (spec §20, §22; errata E5, E9).

CLAUDE.md rule 4 forbids hand-typed numbers, so no price is ever written into
this file. Every price is parsed out of an archived, sha256-hashed copy of the
provider's own pricing page, and `configs/pricing_<date>.yaml` is generated from
those snapshots alone.

Spec §22 and rule 7: standard service tier, uncached, for the headline figures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from .data.hashes import sha256_text

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = Path("data/pricing_snapshots")

SOURCES: dict[str, str] = {
    "openai": "https://platform.openai.com/docs/pricing.md",
    "anthropic": "https://docs.claude.com/en/docs/about-claude/pricing.md",
    "typesafe": "https://docs.typesafe.ai/models.md",
}

# Which row we need out of each page. Parsing is anchored on the model name so a
# reordered table cannot silently shift us onto the wrong row.
OPENAI_MODELS = ("gpt-6-astra", "gpt-5.6-sol")
ANTHROPIC_MODELS = ("Claude Sonnet 5", "Claude Sonnet 4.5")

_MONEY = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass
class Price:
    model: str
    input_per_mtok_usd: float
    output_per_mtok_usd: float
    tier: str
    source: str
    notes: list[str] = field(default_factory=list)


def fetch_snapshots(repo_root: Path | str = REPO_ROOT,
                    on: date | None = None) -> dict[str, dict[str, str]]:
    """Download each pricing page and archive it with its sha256."""
    root = Path(repo_root)
    day = (on or datetime.now(timezone.utc).date()).isoformat()
    out_dir = root / SNAPSHOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict[str, str]] = {}
    for name, url in SOURCES.items():
        r = httpx.get(url, timeout=60.0, follow_redirects=True)
        r.raise_for_status()
        text = r.text
        path = out_dir / f"{name}-{day}.md"
        path.write_text(text)
        meta[name] = {
            "url": url,
            "path": str(SNAPSHOT_DIR / path.name),
            "sha256": sha256_text(text),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "bytes": str(len(text.encode("utf-8"))),
        }
    return meta


def _section(md: str, heading: str) -> str:
    """Text from `heading` up to the next '### ' heading."""
    i = md.find(heading)
    if i < 0:
        raise ValueError(f"section {heading!r} not found in snapshot")
    j = md.find("\n### ", i + len(heading))
    return md[i: j if j > 0 else len(md)]


def parse_openai(md: str) -> list[Price]:
    """Standard tier, short-context, uncached (spec §22).

    The page also lists Batch, Flex and Fast tiers; anchoring on the
    '### Standard pricing data' section keeps us on the required one.
    """
    section = _section(md, "### Standard pricing data")
    prices: list[Price] = []
    for model in OPENAI_MODELS:
        row = next((ln for ln in section.splitlines()
                    if ln.strip().startswith(f"| {model} ")), None)
        if row is None:
            raise ValueError(f"no standard-pricing row for {model}")
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        # columns: input, cached input, cache writes, output, then long-context
        nums = [float(_MONEY.search(c).group(1)) if _MONEY.search(c) else None
                for c in cells[1:5]]
        if nums[0] is None or nums[3] is None:
            raise ValueError(f"could not parse prices for {model}: {cells[:5]}")
        prices.append(Price(
            model=model, input_per_mtok_usd=nums[0], output_per_mtok_usd=nums[3],
            tier="standard/short-context/uncached", source="openai",
            notes=["short-context column; every dev example is far below any "
                   "long-context threshold (max ~34k tokens)"],
        ))
    return prices


def parse_anthropic(md: str) -> list[Price]:
    prices: list[Price] = []
    for model in ANTHROPIC_MODELS:
        row = next((ln for ln in md.splitlines()
                    if ln.strip().startswith(f"| {model} ")
                    and "MTok" in ln and ln.count("|") >= 6), None)
        if row is None:
            raise ValueError(f"no pricing row for {model}")
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        # columns: base input, 5m cache writes, 1h cache writes, cache hits, output
        base_in = _MONEY.search(cells[1])
        out = _MONEY.search(cells[-1])
        if not base_in or not out:
            raise ValueError(f"could not parse prices for {model}: {cells}")
        prices.append(Price(
            model=model, input_per_mtok_usd=float(base_in.group(1)),
            output_per_mtok_usd=float(out.group(1)),
            tier="standard/uncached", source="anthropic",
            notes=["base input (uncached) and output columns"],
        ))
    return prices


def parse_typesafe(md: str) -> list[Price]:
    """Jev prices per Btok/Mtok; output tokens are free."""
    row = next((ln for ln in md.splitlines()
                if "Price" in ln and "tok" in ln and "$" in ln), None)
    if row is None:
        raise ValueError("no TypeSafe price row found")
    nums = _MONEY.findall(row)
    if len(nums) < 2:
        raise ValueError(f"could not parse TypeSafe price row: {row!r}")
    # "$42 / $0.042" -> per Btok / per Mtok; we use per Mtok.
    per_mtok = float(nums[1])
    return [Price(
        model="jev-1.13.0", input_per_mtok_usd=per_mtok, output_per_mtok_usd=0.0,
        tier="standard/uncached", source="typesafe",
        notes=["TypeSafe charges input tokens only; output tokens are free"],
    )]


def build(repo_root: Path | str = REPO_ROOT, on: date | None = None) -> Path:
    root = Path(repo_root)
    day = (on or datetime.now(timezone.utc).date()).isoformat()
    meta = fetch_snapshots(root, on=on)

    prices: list[Price] = []
    prices += parse_openai((root / meta["openai"]["path"]).read_text())
    prices += parse_anthropic((root / meta["anthropic"]["path"]).read_text())
    prices += parse_typesafe((root / meta["typesafe"]["path"]).read_text())

    doc: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tier_policy": "standard service tier, uncached, per spec §22 and rule 7",
        "snapshots": meta,
        "prices_per_mtok_usd": {
            p.model: {
                "input": p.input_per_mtok_usd,
                "output": p.output_per_mtok_usd,
                "tier": p.tier,
                "source": p.source,
                "notes": p.notes,
            } for p in prices
        },
        "local_models": {
            "lytang/MiniCheck-Flan-T5-Large": {
                "api_price_usd": 0.0,
                # Errata E5: MiniCheck is not an API call, so its cost is compute
                # time on recorded hardware, not a token price.
                "cost_model": "local compute; record hardware and wall-clock, "
                              "then price against the chosen instance rate",
                "hardware": "recorded per run in run_manifest.json",
            }
        },
        "caveats": [
            "Anthropic documents that Claude 4.7+ use a newer tokenizer producing "
            "~30% more tokens for the same text than Sonnet 4.6 and earlier. Cost "
            "is therefore computed from MEASURED token usage per example, never "
            "from a price-per-token comparison across models.",
            "claude-sonnet-4-5 is priced here because it is the production "
            "verifier's model (spec §7.5, §33), not because it is a baseline.",
        ],
    }
    out = root / "configs" / f"pricing_{day}.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return out


def cost_usd(input_tokens: int | None, output_tokens: int | None,
             price_in_per_mtok: float, price_out_per_mtok: float) -> float | None:
    """Spec §20 cost formula. Returns None when usage is unknown."""
    if input_tokens is None and output_tokens is None:
        return None
    i = (input_tokens or 0) / 1_000_000 * price_in_per_mtok
    o = (output_tokens or 0) / 1_000_000 * price_out_per_mtok
    return i + o


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a dated pricing config from snapshots.")
    ap.add_argument("--date", default=None)
    args = ap.parse_args(argv)
    on = date.fromisoformat(args.date) if args.date else None
    path = build(on=on)
    print(f"wrote {path}")
    print(json.dumps(yaml.safe_load(path.read_text())["prices_per_mtok_usd"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Cost model and pricing-snapshot tests (spec §20, §22, §36; errata E5).

CLAUDE.md rule 4: no hand-typed numbers. These tests therefore verify that the
generated pricing config still matches the ARCHIVED snapshot bytes, so a price
cannot be edited into the config without the snapshot changing too.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.hashes import sha256_text  # noqa: E402
from src.pricing import (  # noqa: E402
    cost_usd,
    parse_anthropic,
    parse_openai,
    parse_typesafe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _configs():
    return sorted(REPO_ROOT.glob("configs/pricing_*.yaml"))


@pytest.fixture(scope="module")
def cfg():
    files = _configs()
    if not files:
        pytest.skip("no pricing config generated yet")
    return yaml.safe_load(files[-1].read_text())


# --- §36: synthetic token counts -> dollars ------------------------------

def test_cost_math_with_synthetic_tokens():
    # 1,000,000 input tokens at $10/MTok = $10 exactly
    assert cost_usd(1_000_000, 0, 10.0, 50.0) == pytest.approx(10.0)
    # 500 in + 100 out at $4 / $20 per MTok
    expected = 500 / 1e6 * 4.0 + 100 / 1e6 * 20.0
    assert cost_usd(500, 100, 4.0, 20.0) == pytest.approx(expected)


def test_cost_is_zero_for_zero_tokens():
    assert cost_usd(0, 0, 10.0, 50.0) == pytest.approx(0.0)


def test_cost_is_none_when_usage_unknown():
    """A local model reports no provider tokens; that must not become $0.00."""
    assert cost_usd(None, None, 10.0, 50.0) is None


def test_missing_output_tokens_counts_as_zero_not_none():
    assert cost_usd(1000, None, 10.0, 50.0) == pytest.approx(1000 / 1e6 * 10.0)


def test_jev_output_tokens_are_free():
    """TypeSafe bills input only; output at $0 must not inflate Jev's cost."""
    assert cost_usd(1000, 99999, 0.042, 0.0) == pytest.approx(1000 / 1e6 * 0.042)


# --- the config must be derived from the snapshots -----------------------

def test_snapshot_hashes_still_match_the_archived_files(cfg):
    """If someone edits a price into the config, this fails."""
    for name, meta in cfg["snapshots"].items():
        path = REPO_ROOT / meta["path"]
        assert path.is_file(), f"missing snapshot for {name}"
        assert sha256_text(path.read_text()) == meta["sha256"], name


def test_prices_reparse_from_the_snapshots(cfg):
    """Re-derive every price from the archived bytes and compare."""
    prices = cfg["prices_per_mtok_usd"]
    oa = {p.model: p for p in parse_openai(
        (REPO_ROOT / cfg["snapshots"]["openai"]["path"]).read_text())}
    an = {p.model: p for p in parse_anthropic(
        (REPO_ROOT / cfg["snapshots"]["anthropic"]["path"]).read_text())}
    ts = {p.model: p for p in parse_typesafe(
        (REPO_ROOT / cfg["snapshots"]["typesafe"]["path"]).read_text())}
    for model, p in {**oa, **an, **ts}.items():
        assert prices[model]["input"] == pytest.approx(p.input_per_mtok_usd), model
        assert prices[model]["output"] == pytest.approx(p.output_per_mtok_usd), model


def test_every_required_model_is_priced(cfg):
    prices = cfg["prices_per_mtok_usd"]
    for model in ("jev-1.13.0", "gpt-6-astra", "gpt-5.6-sol", "Claude Sonnet 5"):
        assert model in prices, model
        assert prices[model]["input"] > 0 or model == "jev-1.13.0"


def test_production_verifier_model_is_priced(cfg):
    """Spec §7.5/§33: production runs Claude Sonnet 4.5, not the Sonnet 5 baseline."""
    assert "Claude Sonnet 4.5" in cfg["prices_per_mtok_usd"]


def test_standard_tier_not_batch_or_flex(cfg):
    """Spec §22 and rule 7. Batch and Flex are half price, Fast is double, so
    picking the wrong table would silently halve every OpenAI cost figure."""
    prices = cfg["prices_per_mtok_usd"]
    assert "standard" in prices["gpt-6-astra"]["tier"]
    # The batch table lists gpt-6-astra at 5.00; standard is 10.00.
    assert prices["gpt-6-astra"]["input"] == pytest.approx(10.0)
    assert prices["gpt-5.6-sol"]["input"] == pytest.approx(4.0)


def test_minicheck_is_recorded_as_local_compute(cfg):
    """Errata E5: MiniCheck is not an API call and has no token price."""
    local = cfg["local_models"]["lytang/MiniCheck-Flan-T5-Large"]
    assert local["api_price_usd"] == 0.0
    assert "compute" in local["cost_model"]


def test_tokenizer_caveat_is_recorded(cfg):
    """Anthropic's newer tokenizer produces ~30% more tokens for the same text,
    so cross-model price-per-token comparisons are invalid."""
    assert any("tokenizer" in c for c in cfg["caveats"])


# --- parser robustness ----------------------------------------------------

def test_openai_parser_rejects_a_missing_section():
    with pytest.raises(ValueError):
        parse_openai("# Pricing\n\nno tables here\n")


def test_typesafe_parser_rejects_a_page_without_prices():
    with pytest.raises(ValueError):
        parse_typesafe("# Models\n\nno price row\n")

"""Live model-ID and capability check (errata E9, E15; G0 evidence).

Writes a machine-readable record of what each provider ACTUALLY reports, so no
number in the paper about model identity or probability availability is ever
hand-typed. Run it again on the day of the final run (§2, §29).

Errata E9: if a required model is missing or silently aliased, that is a
blocking issue. This script flags it; it never substitutes a model.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path("runs/model_live_check.json")

REQUIRED = {
    "jev": "jev-1.13.0",
    "gpt6_astra": "gpt-6-astra",
    "gpt56_sol": "gpt-5.6-sol",
    "claude_sonnet5": "claude-sonnet-5",
}

# A single-token A/B probe (spec §12). Short and cheap on purpose.
AB_PROBE = "Reply with exactly one character: A or B."


def check_jev() -> dict[str, Any]:
    from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

    out: dict[str, Any] = {"requested": REQUIRED["jev"], "provider": "typesafe"}
    try:
        q = {"probe": Choice(instructions="Is the sky mentioned?",
                             criteria={"yes": "mentioned", "no": "not mentioned"})}
        with TypeSafeClient(timeout=60.0, retry=RetryPolicy(max_retries=0)) as c:
            r = c.system_one(state={"text": "the sky is blue"}, questions=q,
                             model=REQUIRED["jev"])
            a = r.answers["probe"]
            out.update(
                exists=True, reported=r.model,
                alias_mismatch=(r.model != REQUIRED["jev"]),
                native_probability=True,
                probability_source="probabilities['supported'] (errata E4)",
                probabilities_sum=round(sum(a.probabilities.values()), 10),
                has_confidence=hasattr(a, "confidence"),
            )
            # Record what the moving alias currently resolves to.
            r2 = c.system_one(state={"text": "x"}, questions=q, model="jev-latest")
            out["jev_latest_resolves_to"] = r2.model
    except Exception as exc:
        out.update(exists=False, error=f"{type(exc).__name__}: {exc}"[:400])
    return out


def check_openai(model: str) -> dict[str, Any]:
    from openai import OpenAI

    out: dict[str, Any] = {"requested": model, "provider": "openai"}
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        ids = {m.id for m in client.models.list()}
        out["exists"] = model in ids
    except Exception as exc:
        out.update(exists=None, error=f"list failed: {type(exc).__name__}: {exc}"[:200])
        return out

    # Errata E15: logprob availability decides whether this model can have a
    # native probability at all, and therefore whether Table 3 shows N/A.
    try:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": AB_PROBE}],
            max_completion_tokens=5, logprobs=True, top_logprobs=20)
        lp = r.choices[0].logprobs
        got = bool(lp and lp.content)
        out.update(
            reported=r.model, alias_mismatch=(r.model != model),
            logprobs_supported=got, native_probability=got,
            n_top_logprobs=len(lp.content[0].top_logprobs) if got else 0,
        )
    except Exception as exc:
        out.update(
            logprobs_supported=False, native_probability=False,
            logprobs_error=f"{type(exc).__name__}: {exc}"[:300],
        )
        # Still confirm the model answers at all, without logprobs.
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": AB_PROBE}],
                max_completion_tokens=5)
            out.update(reported=r.model, alias_mismatch=(r.model != model),
                       answers_without_logprobs=True)
        except Exception as exc2:
            out["error"] = f"{type(exc2).__name__}: {exc2}"[:300]
    return out


def check_anthropic(model: str) -> dict[str, Any]:
    from anthropic import Anthropic

    out: dict[str, Any] = {"requested": model, "provider": "anthropic"}
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    try:
        out["exists"] = model in {m.id for m in client.models.list(limit=100)}
        r = client.messages.create(model=model, max_tokens=5,
                                   messages=[{"role": "user", "content": AB_PROBE}])
        out.update(
            reported=r.model, alias_mismatch=(r.model != model),
            # Spec §12: Anthropic exposes no token-logprob interface, so
            # calibration is N/A rather than self-reported.
            logprobs_supported=False, native_probability=False,
            note="spec §12: no native token logprobs; Table 3 shows N/A",
        )
    except Exception as exc:
        out.update(exists=out.get("exists"), error=f"{type(exc).__name__}: {exc}"[:300])
    return out


def run(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    models = {
        "jev": check_jev(),
        "gpt6_astra": check_openai(REQUIRED["gpt6_astra"]),
        "gpt56_sol": check_openai(REQUIRED["gpt56_sol"]),
        "claude_sonnet5": check_anthropic(REQUIRED["claude_sonnet5"]),
    }

    blocking: list[str] = []
    for key, info in models.items():
        if info.get("exists") is False:
            blocking.append(f"{key}: required model {info['requested']} does not exist (E9)")
        if info.get("alias_mismatch"):
            blocking.append(
                f"{key}: requested {info['requested']} but provider reported "
                f"{info.get('reported')} (E9 silent aliasing)")

    native = sorted(k for k, v in models.items() if v.get("native_probability"))
    no_native = sorted(k for k, v in models.items() if not v.get("native_probability"))

    record = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "models_with_native_probability": native,
        "models_without_native_probability": no_native,
        "calibration_note": (
            "Errata E15: models without a native probability show N/A in Table 3. "
            "They are still scored on labels, BAcc, cost and latency."
        ),
        "blocking_issues": blocking,
    }
    out = Path(repo_root) / OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


if __name__ == "__main__":
    rec = run()
    for key, info in rec["models"].items():
        print(f"{key:<16} exists={info.get('exists')!s:<5} "
              f"reported={info.get('reported')!s:<16} "
              f"native_prob={info.get('native_probability')}")
    print(f"\nnative probability : {rec['models_with_native_probability']}")
    print(f"N/A in Table 3     : {rec['models_without_native_probability']}")
    print(f"blocking issues    : {rec['blocking_issues'] or 'none'}")

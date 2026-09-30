#!/usr/bin/env python3
"""Generate docs/reference/limitations-in-full.md from Appendix E's LaTeX.

Section 12 of the paper names every limitation. Appendix E gives each a short entry. The
long-form accounting behind them lives in the released package as this document, because
the appendix was eight pages and a reader who wants the derivations can fetch them.

Nothing here is written by hand: the text is Appendix E's own prose with every macro
expanded to the value arxiv/generated/numbers.tex carries, so the document cannot drift
from the paper's numbers. Regenerate after any change to the appendix or the numbers:

    docker compose run --rm dev uv run python scripts/make_limitations_doc.py
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
MACROS = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}",
                         (REPO / "arxiv/generated/numbers.tex").read_text()))

NAMES = {
    "sec:intro": "Section 1", "sec:related": "Section 2", "sec:metrics": "Section 3",
    "sec:systems": "Section 4", "sec:setup": "Section 5", "sec:same-score": "Section 6",
    "sec:different-errors": "Section 7", "sec:calibration": "Section 8",
    "sec:cascade": "Section 9", "sec:cascade-inference": "Section 9.3",
    "sec:cascade-vs-jev": "Section 9.4", "subsec:random-control": "Section 9.5",
    "sec:tuned": "Section 10", "sec:discussion": "Section 11",
    "sec:limitations": "Section 12", "app:limitations": "Appendix E",
    "app:per-dataset": "Appendix A", "app:prompts": "Appendix B",
    "app:incidents": "Appendix C", "app:repro": "Appendix D",
    "tab:headline": "Table 1", "tab:paired-contrasts": "Table 2",
    "tab:gating": "Table 6", "tab:cascade": "Table 7",
    "tab:cascade-vs-astra": "Table 8", "tab:random": "Table 9", "tab:holm": "Table 11",
}

HEAD = """# Limitations in full

Long-form accounting behind Section 12 of *Jev-as-a-Judge for RAG Claim Verification:
Confidence-Gated Cascading Between a Non-Generative Verifier and a Frontier LLM*.

Section 12 of the paper names every limitation with the number that matters for it, and
Appendix E gives each one a short entry. This is the version with the derivations: the
family-boundary sensitivity of the Holm correction, the MiniCheck anchor's full diagnosis,
the same-prompt repeat's coverage and its reasoning-token accounting. It lives here rather
than in the paper so the paper stays readable. Nothing was dropped in the move.

Every value is expanded from `arxiv/generated/numbers.tex`, which
`scripts/verify/paired_analysis.py` regenerates from the locked predictions, so this
document cannot disagree with the paper. It is generated, not written:

    docker compose run --rm dev uv run python scripts/make_limitations_doc.py

"""


def convert(src: str) -> str:
    t = src
    t = t.replace("\\section{Limitations in full}", "").replace("\\label{app:limitations}", "")
    t = re.sub(r"(?m)%.*$", "", t)
    # A cross-reference and the word before it: "Section~\ref{sec:setup}" is one name.
    t = re.sub(r"(?:Section|Appendix|Table|Figure)~\\ref\{([^}]*)\}",
               lambda m: NAMES.get(m.group(1), m.group(1)), t)
    t = re.sub(r"\\ref\{([^}]*)\}", lambda m: NAMES.get(m.group(1), m.group(1)), t)
    t = re.sub(r"\\citep\{([^}]*)\}", lambda m: "(" + m.group(1).replace(",", ";") + ")", t)
    t = re.sub(r"\\citet\{([^}]*)\}", lambda m: m.group(1), t)
    # Macros, longest name first so \JevFVRTwoDP is not eaten as \JevFVR.
    for name in sorted(MACROS, key=len, reverse=True):
        t = t.replace("\\" + name + "\\ ", MACROS[name] + " ")
        t = re.sub(r"\\" + name + r"(?![A-Za-z])", MACROS[name], t)
    t = re.sub(r"\\paragraph\{([^}]*)\}",
               lambda m: "\n### " + " ".join(m.group(1).split()).rstrip(".") + "\n", t)
    t = re.sub(r"\\textsc\{([^}]*)\}", lambda m: m.group(1).replace("\\_", "_").upper(), t)
    t = re.sub(r"\\texttt\{([^}]*)\}", lambda m: "`" + m.group(1).replace("\\_", "_") + "`", t)
    t = re.sub(r"\\emph\{([^}]*)\}", lambda m: "*" + m.group(1) + "*", t)
    t = t.replace("\\$", "$").replace("\\%", "%").replace("\\_", "_").replace("\\&", "&")
    t = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("{", "").replace("}", ""), t)
    t = re.sub(r"\\[A-Za-z]+", "", t)
    t = t.replace("{", "").replace("}", "").replace("~", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" +\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def main() -> int:
    close = (REPO / "arxiv/source/sections/close.tex").read_text()
    i = close.index("\\section{Limitations in full}")
    j = close.find("\\section{", i + 10)
    body = convert(close[i: j if j > 0 else len(close)])
    left = sorted(set(re.findall(r"\\[A-Za-z]+", body)))
    if left:
        raise SystemExit(f"unexpanded LaTeX remains, so the document would be wrong: {left}")
    out = REPO / "docs/reference/limitations-in-full.md"
    out.write_text(HEAD + body + "\n")
    print(f"wrote {out.relative_to(REPO)} ({len(body.split())} words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

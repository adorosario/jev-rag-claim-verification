"""The six paper figures must be reproducible, vector, and free of hand-typed numbers.

Four properties are checked, because all four are things a reviewer can test
for themselves:

1. Running ``scripts/paper_figures.py`` regenerates all six PDFs from
   ``arxiv/generated/numbers.json`` alone. If the script grew a second data
   source, this fails.
2. Each PDF embeds TrueType outlines and contains no Type 3 font and no raster
   image. arXiv rejects Type 3, and a raster would mean the figure was
   screenshotted rather than drawn.
3. The figure source quotes no literal percentage or dollar figure. Every
   reported value has to arrive through the numbers file (CLAUDE.md rule 4),
   and the paper takes no em dash in any form.
4. A rebuild is byte-identical, and the committed PDFs are what the script
   produces from the current numbers file.
"""

import ast
import importlib.util
import json
import re
import subprocess
import sys
import zlib
from hashlib import sha256
from pathlib import Path

import pytest

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _export import repo_only  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "paper_figures.py"
NUMBERS = REPO_ROOT / "arxiv" / "generated" / "numbers.json"
EXPECTED = 6


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    outdir = tmp_path_factory.mktemp("figures")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--outdir", str(outdir)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return outdir


def _decompressed(pdf: Path) -> bytes:
    raw = pdf.read_bytes()
    out = bytearray(raw)
    for match in re.finditer(rb"stream\r?\n", raw):
        start = match.end()
        end = raw.find(b"endstream", start)
        try:
            out += zlib.decompress(raw[start:end])
        except zlib.error:
            continue
    return bytes(out)


def test_all_six_figures_are_built(built: Path) -> None:
    pdfs = sorted(built.glob("F*.pdf"))
    assert len(pdfs) == EXPECTED, [p.name for p in pdfs]
    assert {p.name[:2] for p in pdfs} == {f"F{i}" for i in range(1, EXPECTED + 1)}


def test_figures_are_vector_with_embedded_truetype(built: Path) -> None:
    for pdf in sorted(built.glob("F*.pdf")):
        body = _decompressed(pdf)
        assert b"FontFile2" in body, f"{pdf.name} embeds no TrueType outlines"
        assert b"/Type3" not in body, f"{pdf.name} contains a Type 3 font, which arXiv rejects"
        assert b"DCTDecode" not in body and b"/Subtype /Image" not in body, f"{pdf.name} contains a raster image"


def test_numbers_json_is_the_only_data_source() -> None:
    source = SCRIPT.read_text()
    opened = re.findall(r"""(?:read_text|open)\(\s*([A-Za-z_][A-Za-z0-9_.]*)""", source)
    assert set(opened) <= {"NUMBERS"}, f"the figure script reads something other than numbers.json: {opened}"


def test_no_reported_value_is_typed_by_hand() -> None:
    """A percentage or dollar amount written straight into the source would be a
    number the evidence file cannot vouch for."""
    text_literals = re.findall(r'(?:f?"""(?:.|\n)*?"""|f?"(?:[^"\\\n]|\\.)*"|f?\'(?:[^\'\\\n]|\\.)*\')', SCRIPT.read_text())
    offenders = []
    for literal in text_literals:
        # "95% interval" names the nominal confidence level of an interval, which is a
        # property of the procedure rather than a measurement read off the data, so it
        # is the one percentage allowed to appear as text.
        literal = literal.replace("95% interval", "the stated interval")
        # a digit immediately before a percent sign, or after a dollar sign, is a
        # claim about the data; format placeholders like {x:.1f}% are fine
        if re.search(r"\d\s*%", literal) or re.search(r"\$\s*\d", literal):
            offenders.append(literal.strip()[:80])
    assert not offenders, f"hand-typed reported values in the figure source: {offenders}"


# A digit-bearing literal is legitimate only when it describes the axis rather than
# the data. Three kinds qualify: a hex colour, a tick label (a mark on the scale), and
# the phrases below, which name a scale endpoint and the chance level of a balanced
# accuracy. Everything else has to arrive through numbers.json.
AXIS_SCALE_PHRASES = ("chance (50)", "on the full 0 to 100 axis")
HEX_COLOUR = re.compile(r"^#[0-9A-Fa-f]{6}$")
TICK_LABEL = re.compile(r"^\d+(?:\.\d+)?$")
# a run of two or more digits that is not part of a decimal or a thousands group
BARE_COUNT = re.compile(r"(?<![\d.,])\d{2,}(?![\d.,])")


def _drawn_string_literals() -> list[str]:
    """Every string constant in the figure source except the docstrings.

    Walking the AST rather than the raw text means an f-string is seen as its literal
    segments, so a format specifier cannot be mistaken for a number about the data,
    and a value interpolated from numbers.json is correctly invisible here."""
    tree = ast.parse(SCRIPT.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]


def test_no_count_is_typed_by_hand() -> None:
    """The sibling check only sees a digit next to a percent or dollar sign. A count
    such as the number of paired examples carries neither, so it needs its own check:
    it is still a measurement, and the evidence file is still the only place it may
    come from."""
    offenders = []
    for literal in _drawn_string_literals():
        if HEX_COLOUR.match(literal) or TICK_LABEL.match(literal):
            continue
        cleaned = literal.replace("95% interval", "the stated interval")
        for phrase in AXIS_SCALE_PHRASES:
            cleaned = cleaned.replace(phrase, "")
        if BARE_COUNT.search(cleaned):
            offenders.append(literal.strip()[:80])
    assert not offenders, f"hand-typed counts in the figure source: {offenders}"


def _figure_module():
    """The figure script, imported rather than run, so a single figure can be exercised."""
    spec = importlib.util.spec_from_file_location("paper_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figure_two_range_note_bounds_every_point_it_plots(tmp_path, monkeypatch) -> None:
    """Figure 2 says in words what its labels say in numbers, and the two disagreed.

    The note read its endpoints off the seven in-sample sweep points and printed them at
    one decimal, so it told a reader "every configuration falls between 73.3 and 75.8"
    while the same figure labelled its lowest point 73.27 and its highest 75.82. Neither
    endpoint was inside the range the figure stated. check_paper_macros.py cannot see it:
    its sources are arxiv/source/**/*.tex and this sentence is drawn into a PDF.

    This captures the note as rendered and checks its endpoints against every accuracy
    value the figure plots, enumerated here from numbers.json rather than from the script.
    """
    module = _figure_module()
    rendered: list[str] = []
    real_note = module.note

    def capture(ax, text, **kwargs):
        rendered.append(text)
        return real_note(ax, text, **kwargs)

    monkeypatch.setattr(module, "note", capture)
    data = json.loads(NUMBERS.read_text())
    module.figure_2(data, tmp_path)

    text = next((t for t in rendered if "falls between" in t), None)
    assert text, f"figure 2 no longer states a vertical range: {rendered}"
    match = re.search(r"falls between (\d+\.\d+) and (\d+\.\d+)", text)
    assert match, f"the range note prints no two endpoints: {text!r}"
    # Same precision as every accuracy label in the figure, which is what failed before.
    assert re.search(r"falls between \d+\.\d{2} and \d+\.\d{2}\b", text), (
        f"the range endpoints are not printed to two decimals: {text!r}")

    plotted = [100.0 * p["macro_bacc"] for p in data["cascade"]["in_sample_sweep"]]
    plotted.append(100.0 * data["cascade"]["in_sample_best"]["macro_bacc"])
    plotted.append(100.0 * data["cascade"]["crossfit"]["macro_bacc_mean"])
    plotted += [100.0 * data["configurations"][k]["macro_bacc"]
                for k in ("jev", "astra", "astra_high")]
    lo, hi = float(match.group(1)), float(match.group(2))
    assert (lo, hi) == (round(min(plotted), 2), round(max(plotted), 2)), (
        f"the note states [{lo}, {hi}] while the figure plots points from "
        f"{round(min(plotted), 2)} to {round(max(plotted), 2)}")


def test_figure_source_has_no_em_dash() -> None:
    source = SCRIPT.read_text()
    for forbidden in ("—", "–", "---"):
        assert forbidden not in source, f"the paper takes no em dash, found {forbidden!r}"


def test_numbers_file_exists() -> None:
    assert NUMBERS.exists(), "the figures have no evidence file to read"


def test_rebuilding_gives_byte_identical_pdfs(tmp_path) -> None:
    """The PDFs carry no timestamp, so anyone can rebuild them and diff the bytes
    against what is committed."""
    digests = []
    for run in ("first", "second"):
        outdir = tmp_path / run
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--outdir", str(outdir)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        digests.append({p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(outdir.glob("F*.pdf"))})
    assert digests[0] == digests[1]


@repo_only  # the package ships the figure script, not the built PDFs
def test_committed_figures_match_a_fresh_build(tmp_path) -> None:
    outdir = tmp_path / "fresh"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--outdir", str(outdir)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    committed = REPO_ROOT / "arxiv" / "figures"
    for fresh in sorted(outdir.glob("F*.pdf")):
        on_disk = committed / fresh.name
        assert on_disk.exists(), f"{fresh.name} has never been committed"
        assert sha256(on_disk.read_bytes()).hexdigest() == sha256(fresh.read_bytes()).hexdigest(), (
            f"{fresh.name} in the repo is not what the script produces now"
        )

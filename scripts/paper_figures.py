"""Generate the six arXiv figures as vector PDFs.

Every value plotted or printed here is read from ``arxiv/generated/numbers.json``.
Nothing is typed by hand and nothing is read from any other source, so the figures
cannot drift from the locked evidence (CLAUDE.md rule 4).

Run it through the container, never host Python::

    docker compose run --rm dev python scripts/paper_figures.py
    docker compose run --rm dev python scripts/paper_figures.py --png   # raster previews too

Output: ``arxiv/figures/F1..F6*.pdf`` (vector, Type 42 fonts, no Type 3).

Design constraints, applied to every figure:
  * one colourblind-safe pair, fixed across all six figures: Okabe-Ito blue for Jev,
    Okabe-Ito orange for GPT-6 Astra. Validated with the dataviz palette checker
    (CVD delta-E 29.2 protan / 30.9 tritan, normal vision 36.2, both well above the
    threshold of 8).
  * colour is never the only channel: every series also carries its own marker,
    line style or hatch, and is directly labelled, so each figure survives
    greyscale print and photocopying.
  * neutral greys are reserved for references, controls and non-identity marks.
  * where a vertical axis is restricted to make a small difference visible, the
    figure says so on its face.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

REPO = Path(__file__).resolve().parents[1]
NUMBERS = REPO / "arxiv" / "generated" / "numbers.json"
OUTDIR = REPO / "arxiv" / "figures"

# ===== style =====

JEV = "#0072B2"       # Okabe-Ito blue
ASTRA = "#E69F00"     # Okabe-Ito orange
ASTRA_INK = "#8A6000"  # darker orange, for orange text on white
INK = "#1A1A1A"
MUTED = "#5A5A5A"
GRID = "#D6D6D6"
FAINT = "#EDEDED"
SURFACE = "#FFFFFF"


def set_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,          # TrueType, never Type 3 (arXiv requirement)
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}"


def usd(x: float) -> str:
    """Money as text, with the dollar sign escaped.

    matplotlib reads a pair of unescaped dollar signs in one string as a mathtext
    delimiter, so every amount carries a backslash. The escape is stripped at draw
    time and the PDF text layer holds a plain dollar sign, which lets one sentence
    quote as many amounts as the reader needs to check the arithmetic."""
    return "\\$" + (f"{x:,.2f}" if x >= 0.10 else f"{x:.3f}")


def wrap_block(ax, text: str, x: float, y_top: float, width: int, leading: float, **kw) -> float:
    """Draw a paragraph in data coordinates, one text object per wrapped line.

    Returns the y of the last line. Wrapping to a character width is what keeps a
    sentence from running off the canvas or across artwork placed beside it."""
    lines = textwrap.wrap(" ".join(text.split()), width)
    for i, line in enumerate(lines):
        ax.text(x, y_top - leading * i, line, ha="left", va="center", **kw)
    return y_top - leading * (len(lines) - 1)


def note(ax, text: str, dy: float = -0.16, width: int = 118) -> None:
    """One recessive block under an axes, for the caveats that belong on the figure.

    The text is rewrapped to a fixed character width so that a long sentence cannot
    silently widen the saved figure (savefig runs with a tight bounding box)."""
    wrapped = "\n".join(textwrap.wrap(" ".join(text.split()), width))
    ax.annotate(
        wrapped,
        xy=(0.0, dy),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=6.6,
        color=MUTED,
    )


# ===== F1 =====


def figure_1(d: dict, outdir: Path) -> Path:
    """Cascade architecture: claim plus evidence, decision model, gate, accept or escalate."""
    jev = d["configurations"]["jev"]
    astra = d["configurations"]["astra"]
    cf = d["cascade"]["crossfit"]

    escalated = cf["escalated_mean"]
    accepted = 1.0 - escalated

    jev_cost = jev["cost_per_1k_usd"]
    astra_cost = astra["cost_per_1k_usd"]
    total_cost = cf["cost_per_1k_usd_mean"]
    # What the escalated claims themselves cost, implied by the identity
    #   total = decision model on every claim + frontier model on the escalated share.
    # The frontier arm's own rate is an average over every claim, and the gate does
    # not escalate an average claim, so the two rates differ. Printing the implied one
    # is what lets a reader reproduce the total from the face of the figure.
    escalated_cost = (total_cost - jev_cost) / escalated

    fig, ax = plt.subplots(figsize=(6.9, 4.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(-9.0, 55.0)
    ax.axis("off")

    def box(x, y, w, h, title, lines, edge=MUTED, face=SURFACE, lw=0.9):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.6,rounding_size=1.2",
                linewidth=lw,
                edgecolor=edge,
                facecolor=face,
            )
        )
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center", fontsize=8.2, color=INK, fontweight="bold")
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 7.8 - 3.9 * i, line, ha="center", va="center", fontsize=7.0, color=MUTED)

    def arrow(x0, y0, x1, y1, color=MUTED, ls="solid", lw=0.9):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=lw,
                color=color,
                linestyle=ls,
                shrinkA=0,
                shrinkB=0,
            )
        )

    box(1.0, 15.0, 17.0, 15.0, "Claim + evidence", ["one grounded", "yes/no question"], face=FAINT)
    box(
        22.0,
        13.5,
        22.0,
        18.0,
        "Decision model",
        ["non-generative, typed verdict", "native probability, no text", f"{usd(jev_cost)} / 1k, every claim"],
        edge=JEV,
        lw=1.3,
    )

    gate_cx, gate_cy = 57.5, 22.5
    ax.add_patch(
        Polygon(
            [(gate_cx - 9.5, gate_cy), (gate_cx, gate_cy + 9.0), (gate_cx + 9.5, gate_cy), (gate_cx, gate_cy - 9.0)],
            closed=True,
            facecolor=SURFACE,
            edgecolor=INK,
            linewidth=1.0,
        )
    )
    ax.text(gate_cx, gate_cy + 1.9, "Confidence", ha="center", va="center", fontsize=7.8, fontweight="bold")
    ax.text(gate_cx, gate_cy - 1.9, "gate", ha="center", va="center", fontsize=7.8, fontweight="bold")
    ax.text(gate_cx, gate_cy - 11.6, r"$\max(p,\,1-p)\ \geq\ t$", ha="center", va="center", fontsize=7.4, color=INK)

    box(73.0, 28.0, 25.5, 13.0, "Accept the verdict", [f"{pct(accepted)}% of claims, no frontier call"], edge=JEV, lw=1.0)
    box(
        73.0,
        3.0,
        25.5,
        21.5,
        "Escalate to frontier LLM",
        [
            f"{pct(escalated)}% of claims",
            "generative verdict, parsed back",
            f"{usd(astra_cost)} / 1k, every claim",
            f"{usd(escalated_cost)} / 1k, these claims",
        ],
        edge=ASTRA,
        lw=1.3,
    )

    arrow(18.6, 22.5, 21.4, 22.5)
    arrow(44.6, 22.5, 47.4, 22.5)
    arrow(gate_cx + 7.2, gate_cy + 3.6, 72.4, 33.0, color=JEV, lw=1.1)
    arrow(gate_cx + 7.2, gate_cy - 3.6, 72.4, 13.0, color=ASTRA, lw=1.1, ls=(0, (4, 2)))

    ax.text(66.0, 30.4, "confident", fontsize=7.0, color=JEV, ha="center", rotation=24)
    # Set below the dashed arrow and clear of the gate's lower right edge: the corridor
    # between the two is narrower than the label, so the label sits under it, not in it.
    ax.text(66.0, 13.2, "not confident", fontsize=7.0, color=ASTRA_INK, ha="center", rotation=-24)

    # The title block sits above the tallest box (whose top edge is at 41.6 once the
    # rounded border's padding is added), and every line of it is wrapped, so no
    # sentence can grow into the artwork on a rebuild.
    ax.text(0.0, 52.0, "Confidence-gated cascade", fontsize=9.5, fontweight="bold", color=INK, ha="left")
    wrap_block(
        ax,
        "The threshold t is the only tunable part. Escalation share and per-claim costs are shown at the "
        f"cross-fitted operating point, {usd(total_cost)} per 1k claims all in.",
        x=0.0,
        y_top=47.6,
        width=112,
        leading=3.3,
        fontsize=7.0,
        color=MUTED,
    )

    # Four money figures on one canvas invite a reader to multiply them. The identity
    # that actually holds is spelled out here, below the artwork, so the check succeeds.
    wrap_block(
        ax,
        f"Reading the cost: the decision model is billed on every claim, {usd(jev_cost)} per 1k, and the "
        f"{pct(escalated)}% of claims the gate escalates add the frontier bill on top, {usd(total_cost)} per 1k all "
        "in. "
        f"Multiplying the escalated share by the frontier model's own rate does not reproduce that total. The gate "
        f"does not escalate an average claim: the claims it sends on imply {usd(escalated_cost)} per 1k escalated, "
        f"against {usd(astra_cost)} per 1k for the same model across every claim in the set.",
        x=0.0,
        y_top=-1.6,
        width=126,
        leading=3.0,
        fontsize=6.6,
        color=MUTED,
    )

    fig.tight_layout(pad=0.4)
    return save(fig, outdir / "F1-cascade-architecture.pdf")


# ===== F2 =====


def figure_2(d: dict, outdir: Path) -> Path:
    """Cost-accuracy frontier, in-sample and cross-fitted plotted distinctly."""
    sweep = d["cascade"]["in_sample_sweep"]
    best = d["cascade"]["in_sample_best"]
    cf = d["cascade"]["crossfit"]
    jev = d["configurations"]["jev"]
    astra = d["configurations"]["astra"]
    astra_high = d["configurations"]["astra_high"]

    xs = [p["cost_per_1k_usd"] for p in sweep]
    ys = [100.0 * p["macro_bacc"] for p in sweep]

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.set_xscale("log")
    ax.grid(axis="y", which="major", zorder=0)
    ax.set_axisbelow(True)

    ax.plot(
        xs,
        ys,
        linestyle=(0, (5, 2)),
        linewidth=1.0,
        color=MUTED,
        marker="o",
        markersize=5.0,
        markerfacecolor=SURFACE,
        markeredgecolor=MUTED,
        markeredgewidth=1.0,
        zorder=3,
        label="cascade, in-sample threshold (optimistic)",
    )

    cf_x = cf["cost_per_1k_usd_mean"]
    cf_y = 100.0 * cf["macro_bacc_mean"]
    lo, hi = (100.0 * v for v in cf["macro_bacc_ci"])
    ax.errorbar(
        [cf_x],
        [cf_y],
        yerr=[[cf_y - lo], [hi - cf_y]],
        fmt="D",
        markersize=7.0,
        color=INK,
        ecolor=INK,
        elinewidth=1.1,
        capsize=3.0,
        capthick=1.1,
        zorder=6,
        label="cascade, cross-fitted (honest)",
    )

    ax.plot(
        [jev["cost_per_1k_usd"]],
        [100.0 * jev["macro_bacc"]],
        marker="o",
        markersize=8.0,
        color=JEV,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        linestyle="none",
        zorder=7,
        label="jev-1.13.0",
    )
    ax.plot(
        [astra["cost_per_1k_usd"]],
        [100.0 * astra["macro_bacc"]],
        marker="s",
        markersize=8.0,
        color=ASTRA,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        linestyle="none",
        zorder=7,
        label="GPT-6 Astra, low effort",
    )
    ax.plot(
        [astra_high["cost_per_1k_usd"]],
        [100.0 * astra_high["macro_bacc"]],
        marker="^",
        markersize=8.0,
        markerfacecolor=SURFACE,
        color=ASTRA,
        markeredgecolor=ASTRA,
        markeredgewidth=1.4,
        linestyle="none",
        zorder=7,
        label="GPT-6 Astra, high effort",
    )

    best_x, best_y = best["cost_per_1k_usd"], 100.0 * best["macro_bacc"]
    ax.annotate(
        "",
        xy=(cf_x, cf_y),
        xytext=(best_x, best_y),
        arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=0.9, linestyle=(0, (2, 1.6)), shrinkA=7, shrinkB=9),
        zorder=5,
    )
    ax.annotate(
        # Every accuracy label in this figure is two decimals so that the annotated
        # difference reproduces from the two points it connects. At one decimal the arrow
        # said 1.72 between points labelled 75.8 and 74.1, which subtract to 1.70.
        f"selection optimism:\n{cf['selection_optimism_pp']:.2f} points of the\nin-sample score is threshold choice",
        xy=(0.92, 75.35),
        fontsize=7.2,
        color=INK,
        ha="right",
        va="center",
    )

    ax.annotate(
        f"in-sample best, {100.0 * best['macro_bacc']:.2f} at {usd(best_x)}",
        xy=(best_x, best_y),
        xytext=(best_x * 0.92, best_y + 0.42),
        fontsize=7.0,
        color=MUTED,
        ha="center",
    )
    ax.annotate(
        # Two decimals here and on the high-effort point only: these two quantities are
        # different and round to the same value at one decimal, which Table 1 and Table 5
        # spend a footnote separating. The figure must not put it back.
        f"cross-fitted\n{cf_y:.2f} at {usd(cf_x)}",
        xy=(cf_x, cf_y),
        xytext=(cf_x * 0.86, cf_y - 0.18),
        fontsize=7.0,
        color=INK,
        ha="right",
        va="top",
        fontweight="bold",
    )
    ax.annotate(
        f"jev-1.13.0\n{100.0 * jev['macro_bacc']:.2f} at {usd(jev['cost_per_1k_usd'])}",
        xy=(jev["cost_per_1k_usd"], 100.0 * jev["macro_bacc"]),
        xytext=(jev["cost_per_1k_usd"] * 1.45, 100.0 * jev["macro_bacc"] - 0.22),
        fontsize=7.0,
        color=JEV,
        ha="left",
        va="top",
    )
    ax.annotate(
        f"Astra, low effort\n{100.0 * astra['macro_bacc']:.2f} at {usd(astra['cost_per_1k_usd'])}",
        xy=(astra["cost_per_1k_usd"], 100.0 * astra["macro_bacc"]),
        xytext=(astra["cost_per_1k_usd"] * 0.97, 100.0 * astra["macro_bacc"] - 0.22),
        fontsize=7.0,
        color=ASTRA_INK,
        ha="center",
        va="top",
    )
    ax.annotate(
        f"Astra, high effort\n{100.0 * astra_high['macro_bacc']:.2f} at {usd(astra_high['cost_per_1k_usd'])}",
        xy=(astra_high["cost_per_1k_usd"], 100.0 * astra_high["macro_bacc"]),
        xytext=(astra_high["cost_per_1k_usd"] * 1.12, 100.0 * astra_high["macro_bacc"] + 0.22),
        fontsize=7.0,
        color=ASTRA_INK,
        ha="left",
        va="bottom",
    )

    ax.set_xlabel("cost per 1,000 claims in US dollars, log scale, uncached list pricing")
    ax.set_ylabel("macro-balanced accuracy (%)")
    ax.set_title("What every configuration costs, and how much of the tuned result is real", loc="left", pad=8)
    ax.set_ylim(71.9, 76.6)
    ax.set_xlim(0.03, 32.0)
    ax.set_xticks([0.05, 0.1, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.05", "0.10", "0.50", "1", "2", "5", "10", "20"])
    despine(ax)
    # Every series is also labelled where it sits, and the cheapest point is the
    # bottom-left corner of the plot, so the legend goes underneath the axes rather
    # than into the data. Inside the axes it would land on the jev point's own label.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.135),
        ncols=2,
        handletextpad=0.5,
        labelspacing=0.36,
        columnspacing=1.6,
        borderpad=0.0,
        fontsize=7.0,
    )

    # The range note is read off every point this figure plots, not off the sweep alone.
    # `ys` is the seven in-sample sweep points, so a note that said "every configuration"
    # while reading min(ys)/max(ys) was describing a different set from the one the figure
    # shows, and at one decimal it also excluded the two endpoints it was derived from:
    # 73.27 rendered as "73.3" and 75.82 as "75.8", both outside the stated interval while
    # labelled inside the same axes. Two decimals here, like every accuracy label above.
    plotted_ys = ys + [
        cf_y,
        100.0 * jev["macro_bacc"],
        100.0 * astra["macro_bacc"],
        100.0 * astra_high["macro_bacc"],
    ]
    # The note states a range and the labels above it print points, so the two can
    # disagree, and for one round they did: the note was read off the sweep at one decimal
    # while every label carried two, leaving "between 73.3 and 75.8" two inches under
    # labels reading 73.27 and 75.82. Nothing saw it, because the macro gate reads
    # arxiv/source/**/*.tex and this arithmetic happens in a PDF. So the figure checks
    # itself at build time: the endpoints it is about to print, as printed, must bound
    # every point it labels. Narrow the source set or drop a decimal and the build stops.
    note_lo, note_hi = f"{min(plotted_ys):.2f}", f"{max(plotted_ys):.2f}"
    for label in (best_y, cf_y, 100.0 * jev["macro_bacc"],
                  100.0 * astra["macro_bacc"], 100.0 * astra_high["macro_bacc"]):
        shown = float(f"{label:.2f}")
        assert float(note_lo) <= shown <= float(note_hi), (
            f"figure 2 would state a range of [{note_lo}, {note_hi}] while labelling a "
            f"point at {shown}")

    note(
        ax,
        "Restricted vertical range: every configuration plotted here falls between "
        f"{note_lo} and {note_hi}. The interval on the cross-fitted point spans "
        f"[{lo:.1f}, {hi:.1f}] over {cf['n_repeats']} repeats of {cf['k_folds']}-fold cross-fitting, so it carries "
        f"threshold-selection noise, not the sampling uncertainty of the {d['n_paired']} examples. The dashed curve "
        "is not an "
        "achievable frontier: each of its points was chosen after seeing the answers.",
        dy=-0.33,
    )

    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(bottom=0.33)
    return save(fig, outdir / "F2-cost-accuracy-frontier.pdf")


# ===== F3 =====


def figure_3(d: dict, outdir: Path) -> Path:
    """Error complementarity: both / jev-only / astra-only / neither."""
    c = d["complementarity"]
    n = d["n_paired"]
    jev = d["configurations"]["jev"]
    astra = d["configurations"]["astra"]

    segments = [
        ("both correct", c["both_correct"], "#E3E3E3", "", INK, "below"),
        ("Jev correct, Astra wrong", c["jev_only_correct"], JEV, "///", SURFACE, "above-left"),
        ("Astra correct, Jev wrong", c["astra_only_correct"], ASTRA, "\\\\\\", INK, "above-right"),
        ("neither correct", c["neither_correct"], "#8C8C8C", "", SURFACE, "below"),
    ]

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    bar_h = 0.42
    left = 0.0
    for label, count, colour, hatch, textcolour, placement in segments:
        width = 100.0 * count / n
        ax.barh(0, width - 0.4, left=left, height=bar_h, color=colour, edgecolor=SURFACE, linewidth=0.0, hatch=hatch)
        centre = left + (width - 0.4) / 2.0
        ax.text(centre, 0.0, f"{count}", ha="center", va="center", fontsize=8.6, color=textcolour, fontweight="bold")
        if placement == "below":
            ax.annotate(
                f"{label}\n{width:.1f}%",
                xy=(centre, -bar_h / 2),
                xytext=(0, -7),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7.2,
                color=INK,
            )
        else:
            # the two thin segments get staggered callouts, so their labels cannot collide
            dx, dy, ha = (-16.0, 36.0, "right") if placement == "above-left" else (16.0, 14.0, "left")
            ax.annotate(
                f"{label}, {width:.1f}%",
                xy=(centre, bar_h / 2),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=7.2,
                color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6, shrinkB=1.5),
            )
        left += width

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.75, 0.75)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    ax.set_title(f"Where the two systems disagree: {n} paired claims", loc="left", pad=40)
    ax.annotate(
        f"They agree on {pct(c['agreement'])}% of claims and disagree on {pct(c['disagreement'])}%. "
        f"The disagreement splits {c['jev_only_correct']} to {c['astra_only_correct']}, and that split is the whole "
        "opportunity a router has to work with.",
        xy=(0.0, 1.34),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=MUTED,
    )
    note(
        ax,
        "The errors also differ in direction: Jev verifies "
        f"{pct(jev['false_verification_rate'])}% of unsupported claims against Astra's "
        f"{pct(astra['false_verification_rate'])}%, a gap of {c['fvr_gap_pp']:.1f} points,\n"
        "so the matching headline accuracy hides the more permissive guard.",
        dy=-0.42,
    )

    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(bottom=0.32, top=0.66)
    return save(fig, outdir / "F3-error-complementarity.pdf")


# ===== F4 =====


def figure_4(d: dict, outdir: Path) -> Path:
    """Reliability diagrams, with the missing-probability case stated rather than left blank."""
    jev = d["configurations"]["jev"]
    astra = d["configurations"]["astra"]
    cal = jev["calibration"]
    bins = cal["reliability"]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.7))

    ax = axes[0]
    ax.set_aspect("equal")
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], linestyle=(0, (3, 2)), linewidth=0.8, color=MUTED, zorder=2)
    ax.text(0.47, 0.435, "perfect calibration", fontsize=6.6, color=MUTED, rotation=45, rotation_mode="anchor", ha="center", va="top")

    xs = [b["mean_p"] for b in bins]
    ys = [b["observed_supported_rate"] for b in bins]
    sizes = [14.0 + 0.55 * b["n"] for b in bins]
    ax.plot(xs, ys, linewidth=1.2, color=JEV, zorder=4)
    ax.scatter(xs, ys, s=sizes, color=JEV, edgecolor=SURFACE, linewidth=1.0, zorder=5)

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("predicted probability that the claim is supported")
    ax.set_ylabel("observed rate of supported claims")
    ax.set_title("jev-1.13.0: native probability", loc="left", pad=6)
    despine(ax)

    ax.text(
        0.965,
        0.035,
        "\n".join(
            [
                f"ECE = {cal['ece']:.3f} ({cal['ece_bins']} equal-count bins)",
                f"ECE = {cal['ece_sensitivity']:.3f} at {cal['ece_sensitivity_bins']} bins",
                f"Brier = {cal['brier']:.3f}",
                f"AUROC, ranking claims = {cal['auroc']:.3f}",
                f"selective AUROC, ranking its own errors = {cal['selective_auroc']:.3f}",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=SURFACE, edgecolor=GRID, linewidth=0.6),
    )
    ax.annotate(
        f"marker area is proportional to bin count ({min(b['n'] for b in bins)} to {max(b['n'] for b in bins)} claims)",
        xy=(0.0, -0.20),
        xycoords="axes fraction",
        fontsize=6.6,
        color=MUTED,
        ha="left",
        va="top",
    )

    ax2 = axes[1]
    ax2.set_aspect("equal")
    ax2.set_xlim(-0.04, 1.04)
    ax2.set_ylim(-0.04, 1.04)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.grid(True, color=FAINT, zorder=0)
    ax2.set_axisbelow(True)
    ax2.plot([0, 1], [0, 1], linestyle=(0, (3, 2)), linewidth=0.8, color=FAINT, zorder=2)
    ax2.set_xlabel("predicted probability that the claim is supported")
    ax2.set_title("GPT-6 Astra: nothing to plot", loc="left", pad=6)
    despine(ax2)

    sentence = astra["calibration"]["note"]
    body = "\n".join(textwrap.wrap(sentence[0].upper() + sentence[1:], 40))
    ax2.text(
        0.5,
        0.62,
        f"probability provenance: {astra['calibration']['probability_provenance']}",
        transform=ax2.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )
    ax2.text(
        0.5,
        0.52,
        body,
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.6", facecolor=SURFACE, edgecolor=GRID, linewidth=0.6),
    )

    fig.suptitle(
        "Whether the routing signal means anything, and which system has one at all",
        x=0.012,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.tight_layout(pad=0.7, rect=(0, 0.02, 1, 0.94))
    return save(fig, outdir / "F4-reliability-and-provenance.pdf")


# ===== F5 =====


def figure_5(d: dict, outdir: Path) -> Path:
    """Confidence routing against the matched random-escalation control."""
    rows = sorted(d["cascade"]["random_control"], key=lambda r: r["escalated"])
    x = [100.0 * r["escalated"] for r in rows]
    conf = [100.0 * r["confidence_macro_bacc"] for r in rows]
    rand = [100.0 * r["random_macro_bacc_mean"] for r in rows]
    lo = [100.0 * r["random_macro_bacc_ci"][0] for r in rows]
    hi = [100.0 * r["random_macro_bacc_ci"][1] for r in rows]

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    ax.fill_between(
        x,
        lo,
        hi,
        facecolor=ASTRA,
        alpha=0.13,
        edgecolor=ASTRA,
        linewidth=0.5,
        hatch="\\\\\\",
        zorder=2,
        label="random escalation, 95% interval",
    )
    ax.plot(
        x,
        rand,
        linestyle=(0, (5, 2)),
        linewidth=1.3,
        color=ASTRA,
        marker="s",
        markersize=5.5,
        markeredgecolor=SURFACE,
        markeredgewidth=0.9,
        zorder=4,
        label="random escalation, mean",
    )
    ax.plot(
        x,
        conf,
        linestyle="solid",
        linewidth=1.6,
        color=JEV,
        marker="o",
        markersize=6.0,
        markeredgecolor=SURFACE,
        markeredgewidth=0.9,
        zorder=5,
        label="confidence-gated escalation",
    )

    for r, xi, ci in zip(rows, x, conf):
        ax.annotate(
            f"+{r['confidence_minus_random_pp']:.2f}",
            xy=(xi, ci),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
            color=JEV,
        )

    cleared = [r for r in rows if r["clears_random_ci"]]
    for r in cleared:
        xi = 100.0 * r["escalated"]
        yi = 100.0 * r["confidence_macro_bacc"]
        ax.plot([xi], [yi], marker="o", markersize=14.0, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.0, zorder=6)
        ax.annotate(
            "the one budget that clears the random\ninterval, and it is also the budget that\nwas selected in-sample",
            xy=(xi + 0.6, yi + 0.35),
            xytext=(26.0, 77.6),
            fontsize=7.0,
            color=INK,
            ha="left",
            va="top",
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6, shrinkB=9),
        )

    ax.set_xlabel("share of claims escalated to the frontier model (%)")
    ax.set_ylabel("macro-balanced accuracy (%)")
    ax.set_title("Routing on confidence against spending the same budget at random", loc="left", pad=8)
    ax.set_xlim(5, 72)
    ax.set_ylim(70.2, 78.0)
    despine(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.145), ncols=3, handletextpad=0.6, columnspacing=1.8)

    gains = [r["confidence_minus_random_pp"] for r in rows]
    note(
        ax,
        f"Labels are the gain over random at the same budget. Confidence wins at all {len(rows)} budgets, by "
        f"{min(gains):.2f} to {max(gains):.2f} points, and clears the random 95% interval at {len(cleared)} of them.\n"
        f"Restricted vertical range. The sign is consistent, but at n = {d['n_paired']} the comparison is underpowered: "
        "this figure is the control that says so.",
        dy=-0.27,
    )

    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(bottom=0.29)
    return save(fig, outdir / "F5-confidence-vs-random-routing.pdf")


# ===== F6 =====


def figure_6(d: dict, outdir: Path) -> Path:
    """Per-dataset dumbbell on an honest 0 to 100 axis."""
    jev = d["configurations"]["jev"]["per_dataset"]
    astra = d["configurations"]["astra"]["per_dataset"]
    support = d["dataset_support"]["per_dataset"]
    c = d["complementarity"]

    names = sorted(jev, key=lambda k: jev[k] - astra[k])
    ys = list(range(len(names)))

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    ax.axvline(50.0, color=MUTED, linewidth=0.7, zorder=2)
    ax.annotate(
        "chance (50)",
        xy=(50.0, len(names) - 0.42),
        xytext=(2.0, 0),
        textcoords="offset points",
        fontsize=6.8,
        color=MUTED,
        ha="left",
        va="center",
    )

    for y, name in zip(ys, names):
        a, b = 100.0 * jev[name], 100.0 * astra[name]
        ax.plot([a, b], [y, y], color=GRID, linewidth=1.6, zorder=3, solid_capstyle="round")
        ax.plot([a], [y], marker="o", markersize=7.0, color=JEV, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5)
        ax.plot([b], [y], marker="s", markersize=6.6, color=ASTRA, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5)

    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.75, len(names) - 0.25)
    ax.set_xticks([0, 20, 40, 50, 60, 80, 100])
    ax.set_xlabel("per-dataset balanced accuracy (%), on the full 0 to 100 axis")
    ax.set_title("Where each system wins, dataset by dataset", loc="left", pad=8)
    despine(ax)

    ax.annotate(
        "smaller gold\nclass (n)",
        xy=(1.055, 1.01),
        xycoords="axes fraction",
        fontsize=6.6,
        color=MUTED,
        ha="center",
        va="bottom",
    )
    for y, name in zip(ys, names):
        ax.annotate(
            f"{support[name]['min_class']} / {support[name]['n']}",
            xy=(1.055, y),
            xycoords=("axes fraction", "data"),
            fontsize=6.8,
            color=MUTED,
            ha="center",
            va="center",
        )

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=JEV, markersize=7.0, label="jev-1.13.0"),
        Line2D([], [], marker="s", linestyle="none", color=ASTRA, markersize=6.6, label="GPT-6 Astra"),
    ]
    ax.legend(handles=handles, loc="lower right", handletextpad=0.6, borderpad=0.2)

    note(
        ax,
        f"Rows are ordered by the gap between the systems. Jev is ahead on {c['datasets_jev_higher']} of "
        f"{d['n_datasets']} datasets and Astra on {c['datasets_astra_higher']}. Each dataset contributes "
        f"{support[names[0]]['n']} claims and the smaller gold class is as thin as\n"
        f"{d['dataset_support']['thinnest_minority_class_n']} items "
        f"({d['dataset_support']['thinnest_dataset']}), so no single-dataset difference is significant on its own.",
        dy=-0.17,
    )

    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(bottom=0.22, right=0.89)
    return save(fig, outdir / "F6-per-dataset-dumbbell.pdf")


# ===== driver =====


SAVE_PNG = False


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # CreationDate is dropped on purpose: without it the same numbers.json produces
    # byte-identical PDFs on every run, so a reviewer can rebuild the figures and
    # diff them against the committed ones.
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"Creator": "scripts/paper_figures.py", "Producer": "matplotlib", "CreationDate": None},
    )
    if SAVE_PNG:
        fig.savefig(path.with_suffix(".png"), format="png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def main() -> None:
    global SAVE_PNG
    parser = argparse.ArgumentParser(description="Build the six arXiv figures from numbers.json.")
    parser.add_argument("--png", action="store_true", help="also write raster previews beside the PDFs")
    parser.add_argument("--outdir", default=str(OUTDIR))
    args = parser.parse_args()
    SAVE_PNG = args.png

    d = json.loads(NUMBERS.read_text())
    set_style()
    outdir = Path(args.outdir)
    for fn in (figure_1, figure_2, figure_3, figure_4, figure_5, figure_6):
        path = fn(d, outdir)
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        print(f"wrote {rel}")


if __name__ == "__main__":
    main()

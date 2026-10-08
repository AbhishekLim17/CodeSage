r"""Figures for the thesis from the analysis tables (research plan, step 3.8).

    python eval/analyze.py RESULTS_DIR ... --out analysis/
    python eval/figures.py analysis/          # writes analysis/figures/*.png (to look at) and *.pdf (for the thesis)

One figure per question the analysis answers, drawn only when its table has data:

* ``primary``: every primary comparison with its 95% interval and its reference line (no effect, the margin, the H5
  threshold, 50%), one panel per kind of quantity, so differences, kappas and a share never share an axis;
* ``grounded_by_model``: grounded rate under P0 and P1 for each model, smallest model first (H1);
* ``judges``: each judge's kappa over all claims and over cited claims, and how often it says "supported" next to how
  often the main grader does (RQ3);
* ``error_sources``: for each condition, the not-fully-correct answers that had a gold file in their context and those
  that did not (RQ4);
* ``power``: the power of H2 by disagreement and sample size (from ``power_simulation.py``; needs no data).

Colours are the first three slots of a palette checked for colour-blind separation; the third (green) is below 3:1
contrast on white, so wherever it is used, its series is also labelled directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import power_simulation

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, green: validated as a set (all pairs), light mode
INK, INK_2, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ROW_INCHES = 0.9  # height per category row in bar charts, so bars stay thin however many rows there are

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.labelcolor": INK_2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
})


def read(folder: Path, name: str) -> pd.DataFrame | None:
    path = folder / f"{name}.csv"
    try:
        table = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    return table if len(table) else None


def save(fig, out: Path, name: str) -> list[Path]:
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def reference(ax, x: float, label: str, side: str = "right") -> None:
    """A vertical reference line with its name written beside it (right or left of the line), in muted ink."""
    ax.axvline(x, color=MUTED, linewidth=1, zorder=1)
    ax.annotate(label, (x, 1), xycoords=("data", "axes fraction"), xytext=(3 if side == "right" else -3, -2),
                textcoords="offset points", va="top", ha="left" if side == "right" else "right", fontsize=8, color=MUTED)


def bottom_legend(fig, ax, ncol: int) -> None:
    """The legend under the plot, clear of the title and the data; room is made for it below the axis label."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=8)
    fig.tight_layout(rect=(0, 0.35 / fig.get_figheight(), 1, 1))


def interval_panel(ax, rows: pd.DataFrame, title: str, refs: list[tuple], xlim: tuple[float, float]) -> None:
    """Dot (estimate) and line (95% interval) per row, top to bottom, with the Holm p-value at the right."""
    y = np.arange(len(rows))[::-1]
    ax.hlines(y, rows["ci_low"], rows["ci_high"], color=SERIES[0], linewidth=2, capstyle="round", zorder=3)
    ax.scatter(rows["estimate"], y, s=40, color=SERIES[0], edgecolor="white", linewidth=1.5, zorder=4)
    ax.set_yticks(y, [label_of(r) for r in rows.itertuples()])
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    for ref in refs:
        reference(ax, *ref)
    for yy, r in zip(y, rows.itertuples(), strict=True):
        ax.annotate(f"p (Holm) {r.p_holm:.3f}", (1, yy), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_title(title, loc="left")


def label_of(row) -> str:
    return f"{row.id}  ({row.n})"


def primary_figure(table: pd.DataFrame, out: Path) -> list[Path]:
    panels = [
        ("Paired differences", table[table["id"].str.match(r"C[123]")], [(0.0, "no difference")], (-1.05, 1.05)),
        ("Judge vs main grader: Cohen's kappa", table[table["id"].str.startswith("C4")],
         [(0.4, "H5: 0.4 (up to 4B)", "left"), (0.7, "H5: 0.7 (7-8B)")], (-0.2, 1.05)),
        ("Not fully correct, yet faithful (share)", table[table["id"] == "C5"], [(0.5, "50%")], (0, 1.05)),
    ]
    panels = [p for p in panels if len(p[1])]
    margin = table.loc[table["id"] == "C2", "test"]
    if len(margin):  # the C2 margin, read from the test it was run with ("one-sided, > -0.10"); labelled on the left
        panels[0][2].append((float(margin.iloc[0].split(">")[1]), "C2 margin", "left"))
    heights = [max(1.2, 0.45 * len(rows) + 0.6) for _, rows, _, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7, sum(heights) + 0.4), gridspec_kw={"height_ratios": heights},
                             squeeze=False)
    for ax, (title, rows, refs, xlim) in zip(axes[:, 0], panels, strict=True):
        interval_panel(ax, rows, title, refs, xlim)
    fig.suptitle("Primary comparisons, with 95% intervals", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, out, "primary")


def grounded_by_model_figure(table: pd.DataFrame, out: Path) -> list[Path]:
    table = table.sort_values("size_b", ascending=False)  # smallest model at the top
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(table) + 1.3))
    ax.hlines(y, table["P0"], table["P1"], color=GRID, linewidth=2, zorder=2)
    for column, colour, name in (("P0", SERIES[1], "P0: system prompt only"), ("P1", SERIES[0], "P1: with the reminder")):
        ax.scatter(table[column], y, s=48, color=colour, edgecolor="white", linewidth=1.5, zorder=3, label=name)
    for yy, r in zip(y, table.itertuples(), strict=True):
        ax.annotate(f"{r.gain:+.0%}", (max(r.P0, r.P1), yy), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK_2)
    ax.set_yticks(y, [f"{m}  ({n} q)" for m, n in zip(table["model"], table["questions"], strict=True)])
    ax.set_xlim(-0.03, 1.12)  # a dot at 0% or 100% is drawn whole, with room for its label
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("answers that cite a supplied source (grounded)")
    ax.set_title("Grounded rate without and with the citation reminder (H1)", loc="left")
    bottom_legend(fig, ax, ncol=2)
    return save(fig, out, "grounded_by_model")


def grouped_bars(ax, categories, series: list[tuple[str, list[float], str]], xlabel: str, xlim) -> None:
    """Horizontal bars, one group per category, thin, with a white gap between neighbours and the value at the tip.

    A value that cannot be computed (a kappa when every label is the same) gets no bar and says "n/a".
    """
    height = 0.5 / len(series)  # with ROW_INCHES per category, a bar stays under about a quarter of an inch
    y = np.arange(len(categories))[::-1]
    for i, (name, values, colour) in enumerate(series):
        offset = (len(series) - 1) / 2 * height - i * height
        shown = [0.0 if np.isnan(v) else v for v in values]
        bars = ax.barh(y + offset, shown, height=height, color=colour, edgecolor="white", linewidth=2, label=name)
        for bar, value in zip(bars, values, strict=True):
            text = "n/a" if np.isnan(value) else f"{value:.2f}"
            ax.annotate(text, (0 if np.isnan(value) else max(value, 0), bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
    ax.set_ylim(-0.6, len(categories) - 0.4)
    ax.set_yticks(y, categories)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=BASELINE, linewidth=0.8)


def judges_figure(table: pd.DataFrame, out: Path) -> list[Path]:
    judges = sorted(table["judge"].unique())

    def value(judge, subset, column):
        found = table[(table["judge"] == judge) & (table["subset"] == subset)][column]
        return float(found.iloc[0]) if len(found) else float("nan")

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.5, ROW_INCHES * len(judges) + 1.7), sharey=True)
    grouped_bars(left, judges, [("over all claims", [value(j, "all claims", "kappa") for j in judges], SERIES[0]),
                                ("over cited claims only", [value(j, "cited claims", "kappa") for j in judges], SERIES[1])],
                 "Cohen's kappa with the main grader", (-0.2, 1.1))
    left.set_title("Agreement with the main grader (RQ3)", loc="left")
    left.legend(loc="lower right", fontsize=8)
    grouped_bars(right, judges, [("judge", [value(j, "all claims", "judge_says_supported") for j in judges], SERIES[0]),
                                 ("main grader", [value(j, "all claims", "human_says_supported") for j in judges], SERIES[1])],
                 "share of claims labelled supported", (0, 1.15))
    right.set_title("How often each says \"supported\"", loc="left")
    right.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return save(fig, out, "judges")


def error_sources_figure(table: pd.DataFrame, out: Path) -> list[Path]:
    names = [f"{m} {p}-{c}" for m, p, c in zip(table["model"], table["prompt"], table["context"], strict=True)]
    y = np.arange(len(table))[::-1]
    fig, ax = plt.subplots(figsize=(7, ROW_INCHES * len(table) + 1.6))
    generation, retrieval = table["generation_failures"], table["retrieval_failures"]
    ax.barh(y, generation, height=0.3, color=SERIES[0], edgecolor="white", linewidth=2,
            label="generation: the gold file was in the context")
    ax.barh(y, retrieval, left=generation, height=0.3, color=SERIES[1], edgecolor="white", linewidth=2,
            label="retrieval: the gold file was not supplied")
    for yy, total in zip(y, table["not_fully_correct"], strict=True):
        ax.annotate(str(total), (total, yy), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, names)
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.set_xlim(0, table["not_fully_correct"].max() * 1.1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # counts of answers
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("answers graded not fully correct")
    ax.set_title("Where the wrong answers went wrong (RQ4)", loc="left")
    bottom_legend(fig, ax, ncol=2)
    return save(fig, out, "error_sources")


def power_figure(out: Path, margin: float = -0.10, seed: int = 0) -> list[Path]:
    rng = np.random.default_rng(seed)
    disagreement = np.arange(0.06, 0.32, 0.02)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    for ax, truth, title in ((axes[0], 0.0, "If P1 is truly no worse"), (axes[1], -0.03, "If P1 is truly 3 points worse")):
        for n, colour in zip((100, 150, 200), SERIES, strict=True):
            power = [power_simulation.noninferiority_power(rng, n, d, truth, margin) for d in disagreement]
            ax.plot(disagreement, power, color=colour, linewidth=2, solid_capstyle="round", label=f"{n} questions")
            ax.annotate(f"{n} q", (disagreement[-1], power[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        ax.axhline(0.8, color=MUTED, linewidth=1)
        ax.annotate("80%", (disagreement[0], 0.8), xytext=(0, 3), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_xlim(disagreement[0], disagreement[-1] + 0.03)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("questions where P0 and P1 disagree on fully correct")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel(f"power to show non-inferiority (margin {margin:+.0%})")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("H2: chance the study shows P1 is non-inferior", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, out, "power")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis_dir", type=Path, help="the --out folder of analyze.py")
    parser.add_argument("--no-power", action="store_true", help="skip the power figure (it simulates for a few seconds)")
    args = parser.parse_args(argv)
    out = args.analysis_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    drawn, skipped = [], []
    for name, draw in (("primary", primary_figure), ("per_model_gain", grounded_by_model_figure),
                       ("judges", judges_figure), ("error_sources", error_sources_figure)):
        table = read(args.analysis_dir, name)
        if table is None:
            skipped.append(name)
        else:
            drawn += draw(table, out)
    if not args.no_power:
        drawn += power_figure(out)
    for path in drawn:
        print(f"wrote {path}")
    if skipped:
        print(f"no data yet for: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

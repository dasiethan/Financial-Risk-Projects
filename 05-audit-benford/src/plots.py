"""Figures saved as PNG in the project's figures/ folder."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_SOFT, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "lines.linewidth": 2,
})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def benford_first_digit(tables: dict[str, pd.DataFrame], mads: dict[str, float], path: Path) -> None:
    """Observed first-digit shares for two samples against Benford's law."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    digits = next(iter(tables.values())).index.to_numpy()
    for i, ((name, table), color) in enumerate(zip(tables.items(), [BLUE, ORANGE])):
        ax.bar(digits + (i - 0.5) * 0.36, table["observed"], width=0.34, color=color,
               label=f"{name} (MAD {mads[name]:.4f})")
    ax.plot(digits, tables[next(iter(tables))]["expected"], color=INK, marker="o",
            markersize=5, linewidth=1.5, label="Benford's law")
    ax.set_xticks(digits)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("First significant digit of the contract amount")
    ax.set_ylabel("Share of contracts")
    ax.set_title("First-digit test")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, path)


def benford_two_digits(table: pd.DataFrame, top: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(table.index, table["observed"], width=0.8, color=BLUE, label="observed")
    ax.plot(table.index, table["expected"], color=INK, linewidth=1.5, label="Benford's law")
    for d in top.index:
        ax.annotate(str(d), (d, table.loc[d, "observed"]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=INK_SOFT)
    ax.set_xlim(9, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.set_xlabel("First two digits of the contract amount")
    ax.set_ylabel("Share of contracts")
    ax.set_title("First-two-digits test: spikes at round numbers (largest z-scores labelled)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    _save(fig, path)


def thresholds(bins: dict[int, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (threshold, table) in zip(axes.flat, bins.items()):
        x = table["left"].to_numpy() + 0.005
        colors = [ORANGE if 0.95 <= left < 1.0 else BLUE for left in table["left"].round(4)]
        ax.bar(x, table["count"], width=0.009, color=colors)
        ax.plot(x, table["expected"], color=INK, linewidth=1.5, label="smooth counterfactual")
        ax.axvline(1.0, color=INK, linestyle="--", linewidth=1)
        ax.set_title(f"Threshold {threshold:,.0f} € excl. VAT", fontsize=11)
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Contract amount / threshold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for ax in axes[:, 0]:
        ax.set_ylabel("Contracts per 1% bin")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Bunching around procurement thresholds, amounts that are not multiples of "
                 "1,000 € (orange: 5% just below)", fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path)


def flags(counts: pd.Series, total: int, path: Path) -> None:
    data = counts[counts > 0].sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(data.index.str.replace("_", " "), data.values, color=BLUE, height=0.65)
    for y, v in enumerate(data.values):
        ax.text(v, y, f" {v:,} ({v / total:.1%})", va="center", fontsize=8, color=INK_SOFT)
    ax.set_xscale("log")
    ax.set_xlabel("Contracts flagged (log scale)")
    ax.set_title("Audit controls: contracts flagged by each rule")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=data.max() * 8)
    _save(fig, path)

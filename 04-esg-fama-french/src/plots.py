"""Figures saved as PNG in the project's figures/ folder."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SOFT, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PAIR_COLORS = [BLUE, ORANGE, AQUA, YELLOW]

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


def cumulative_growth(returns: pd.DataFrame, pairs: dict[str, str], path: Path) -> None:
    """Growth of $1 for each ESG ETF and its conventional twin, from the ESG inception."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (esg, conv) in zip(axes.flat, pairs.items()):
        sample = returns[[esg, conv]].dropna()
        start = pd.DataFrame(1.0, index=[sample.index[0] - 1], columns=sample.columns)
        growth = pd.concat([start, (1 + sample).cumprod()])
        dates = growth.index.to_timestamp()
        ax.plot(dates, growth[esg], color=BLUE, label=f"{esg} (ESG)")
        ax.plot(dates, growth[conv], color=ORANGE, label=f"{conv} (conventional)")
        ax.set_title(f"{esg} vs {conv}", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:g}"))
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Growth of $1 invested at the ESG fund's inception (total return)",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path)


def alphas(table: pd.DataFrame, path: Path) -> None:
    """Annual alpha with 95% confidence interval: ESG fund, conventional twin, difference."""
    kinds = [("ESG", BLUE), ("conventional", ORANGE), ("ESG minus conventional", INK)]
    labels = table["pair"].unique()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, (kind, color) in enumerate(kinds):
        rows = table[table["kind"] == kind].set_index("pair").loc[labels]
        y = np.arange(len(labels)) + (k - 1) * 0.22
        ax.errorbar(rows["alpha_annual"], y,
                    xerr=[rows["alpha_annual"] - rows["alpha_ci_low"],
                          rows["alpha_ci_high"] - rows["alpha_annual"]],
                    fmt="o", color=color, markersize=6, capsize=3, linewidth=1.5, label=kind)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.set_xlabel("Annualised alpha, FF5 + momentum (95% CI, Newey-West)")
    ax.set_title("Annualised alpha by fund and pair")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
    _save(fig, path)


def factor_tilts(table: pd.DataFrame, factors: list[str], path: Path) -> None:
    """Factor loadings of the ESG-minus-conventional portfolios, with 95% CI."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(factors))
    n = len(table)
    for i, ((_, row), color) in enumerate(zip(table.iterrows(), PAIR_COLORS)):
        offset = (i - (n - 1) / 2) * 0.17
        betas = np.array([row[f"beta_{f}"] for f in factors])
        errors = 1.96 * np.array([row[f"se_{f}"] for f in factors])
        ax.errorbar(x + offset, betas, yerr=errors, fmt="o", color=color, markersize=6,
                    capsize=3, linewidth=1.5, label=row["pair"])
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(x, factors)
    ax.set_ylabel("Loading of ESG minus conventional")
    ax.set_title("Factor tilts of ESG funds relative to their conventional twin")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    _save(fig, path)

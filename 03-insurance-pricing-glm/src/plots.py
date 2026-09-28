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
MODEL_COLORS = {"GLM": BLUE, "LightGBM": ORANGE}

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


def severity_distribution(amounts: pd.Series, cap: float, path: Path) -> None:
    bins = np.logspace(0, np.log10(amounts.max()) + 0.1, 80)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(amounts, bins=bins, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.axvline(cap, color=INK, linestyle="--", linewidth=1)
    ax.annotate(f"cap = {cap:,.0f} €", (cap, ax.get_ylim()[1] * 0.85), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Average cost per claim of each policy (€, uncapped, log scale)")
    ax.set_ylabel("Number of policies (log scale)")
    ax.set_title("Claim severity: a heavy right tail")
    _save(fig, path)


def one_way(tables: dict[str, pd.DataFrame], path: Path) -> None:
    """Observed vs predicted claim frequency by band, with exposure as grey bars."""
    fig, axes = plt.subplots(1, len(tables), figsize=(6 * len(tables), 4.5))
    for ax, (title, table) in zip(np.atleast_1d(axes), tables.items()):
        x = np.arange(len(table))
        twin = ax.twinx()
        twin.bar(x, table["exposure"], color=GRID, width=0.7, zorder=0)
        twin.set_yticks([])
        twin.spines[["top", "right"]].set_visible(False)
        ax.set_zorder(twin.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(x, table["observed"], color=INK, marker="o", markersize=5, linewidth=1.5,
                label="observed")
        for model, color in MODEL_COLORS.items():
            ax.plot(x, table[model], color=color, marker="o", markersize=4, label=model)
        ax.set_xticks(x, table.index, rotation=45 if len(table) > 6 else 0)
        ax.set_ylabel("Claims per policy-year")
        ax.set_title(f"Frequency by {title}", fontsize=11)
    np.atleast_1d(axes)[0].legend(loc="upper right")
    fig.text(0.5, -0.02, "Grey bars: exposure (policy-years) in each band, test set",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout()
    _save(fig, path)


def lorenz(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for model, (x, y, gini) in curves.items():
        ax.plot(x, y, color=MODEL_COLORS[model], label=f"{model} (Gini {gini:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="no segmentation")
    ax.set_xlabel("Cumulative share of exposure\n(policies sorted by predicted pure premium)")
    ax.set_ylabel("Cumulative share of claim cost")
    ax.set_title("Lorenz curves (test set)")
    ax.legend(loc="upper left")
    _save(fig, path)


def premium_deciles(tables: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, len(tables), figsize=(11, 4.2), sharey=True)
    for ax, (model, table) in zip(axes, tables.items()):
        x = table.index.to_numpy()
        ax.bar(x, table["observed_pure_premium"], color=GRID, width=0.7, label="observed")
        ax.plot(x, table["predicted_pure_premium"], color=MODEL_COLORS[model], marker="o",
                markersize=5, label="predicted")
        ax.set_xticks(x)
        ax.set_xlabel("Decile of predicted pure premium (equal exposure)")
        ax.set_title(model, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Cost per policy-year (€, capped claims)")
    fig.suptitle("Pure premium: predicted vs observed by decile (test set)",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path)


def importance(table: pd.DataFrame, path: Path) -> None:
    """Grouped horizontal bars: increase in Poisson deviance when a factor is shuffled."""
    table = table.sort_values("GLM")
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for offset, model in [(0.19, "GLM"), (-0.19, "LightGBM")]:
        ax.barh(y + offset, table[model], height=0.36, color=MODEL_COLORS[model], label=model)
    ax.set_yticks(y, table.index)
    ax.set_xlabel("Increase in test Poisson deviance when shuffled")
    ax.set_title("Permutation importance: claim frequency")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _save(fig, path)

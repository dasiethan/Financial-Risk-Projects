"""Figures saved as PNG in the project's figures/ folder."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
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


def curve_fits(snapshots: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
               error_by_maturity: pd.Series, path: Path) -> None:
    """ECB spot rates (dots) and fitted Nelson-Siegel curves on selected dates, plus the
    mean absolute fitting error by maturity."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (label, (m, y, grid, fitted, rmse)) in zip(axes.flat, snapshots.items()):
        ax.plot(grid, fitted, color=BLUE, label="Nelson-Siegel fit")
        ax.scatter(m, y, color=INK, s=14, zorder=3, label="ECB spot rates")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(f"{label}  (RMSE {rmse:.1f} bp)", fontsize=10)
        ax.set_xlabel("Maturity (years)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    axes.flat[0].legend(loc="lower right", fontsize=8)
    ax = axes.flat[-1]
    ax.bar(np.arange(len(error_by_maturity)), error_by_maturity.values, color=ORANGE, width=0.7)
    ax.set_xticks(np.arange(len(error_by_maturity)), error_by_maturity.index, rotation=90,
                  fontsize=8)
    ax.set_ylabel("Mean absolute error (bp)")
    ax.set_title("Fit error by maturity, all months", fontsize=10)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Euro area AAA zero-coupon curve: Nelson-Siegel fits", fontweight="bold",
                 color=INK)
    fig.tight_layout()
    _save(fig, path)


def factors(factors_df: pd.DataFrame, proxies: pd.DataFrame, corr: dict[str, float],
            path: Path) -> None:
    """Nelson-Siegel factors (fixed tau) against their usual empirical proxies."""
    specs = [("level", "beta0 (level)", "10Y rate"),
             ("slope", "-beta1 (slope)", "10Y - 3M spread"),
             ("curvature", "beta2 (curvature)", "2 x 2Y - 3M - 10Y")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    dates = factors_df.index
    for ax, (key, factor_label, proxy_label) in zip(axes, specs):
        ax.plot(dates, factors_df[key], color=BLUE, label=factor_label)
        ax.plot(dates, proxies[key], color=ORANGE, linewidth=1.3, label=proxy_label)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(f"{key.capitalize()} (correlation {corr[key]:.2f})", loc="left", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("Nelson-Siegel factors and empirical proxies (month-ends)", fontweight="bold",
                 color=INK)
    fig.tight_layout()
    _save(fig, path)


def surplus_shocks(impacts: pd.DataFrame, liabilities_pv: float, path: Path) -> None:
    """Surplus change (% of liability PV) for each asset strategy and scenario."""
    data = impacts / liabilities_pv
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for offset, (col, color) in zip([0.19, -0.19], zip(data.columns, [ORANGE, BLUE])):
        ax.barh(y + offset, data[col], height=0.36, color=color, label=col)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(y, data.index)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1%}"))
    ax.set_xlabel("Change in surplus (assets - liabilities), % of liability value")
    ax.set_title("ALM: surplus sensitivity to yield curve shocks")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, path)

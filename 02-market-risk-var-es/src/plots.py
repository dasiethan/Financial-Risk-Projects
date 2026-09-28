"""Figures saved as PNG in the project's figures/ folder."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from .backtest import GREEN_MAX, TRAFFIC_WINDOW, YELLOW_MAX  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ZONE_COLORS = {"green": "#0ca30c", "yellow": "#fab219", "red": "#d03b3b"}

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


def returns_distribution(r: pd.Series, path: Path) -> None:
    """Histogram on a log scale so that the tails are visible."""
    x = np.linspace(r.min(), r.max(), 500)
    nu, loc, scale = stats.t.fit(r)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(r, bins=120, density=True, color=AXIS, edgecolor="white", linewidth=0.3,
            label="observed daily returns")
    ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=BLUE, label="Normal fit")
    ax.plot(x, stats.t.pdf(x, nu, loc, scale), color=ORANGE, label=f"Student-t fit (ν = {nu:.1f})")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Daily portfolio return")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Fat tails: the Normal distribution underestimates extreme days")
    ax.legend(loc="upper left")
    _save(fig, path)


def garch_volatility(sigma: pd.Series, realized: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(realized.index, realized, color=AXIS, linewidth=1, label="realised volatility (21-day)")
    ax.plot(sigma.index, sigma, color=BLUE, linewidth=1.5, label="GARCH(1,1) forecast")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Annualised volatility")
    ax.set_title("Portfolio volatility: GARCH one-day-ahead forecast")
    ax.legend(loc="upper right")
    _save(fig, path)


def _var_panel(ax, returns: pd.Series, var: pd.DataFrame) -> None:
    ax.plot(returns.index, returns, color=AXIS, linewidth=0.6, label="daily return")
    for method, color in zip(var.columns, [BLUE, ORANGE]):
        ax.plot(var.index, -var[method], color=color, linewidth=1.2, label=f"-VaR 99% {method}")
        hits = returns[-returns > var[method]]
        ax.scatter(hits.index, hits, color=color, s=16, zorder=3, edgecolor="white",
                   linewidth=0.6, label=f"exception, {method}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))


def var_backtest(returns: pd.Series, var: pd.DataFrame, zoom: tuple[str, str], path: Path) -> None:
    """Daily returns against -VaR for two methods, full period and a zoom, exceptions marked."""
    fig, (full, detail) = plt.subplots(2, 1, figsize=(10, 7.5))
    _var_panel(full, returns, var)
    full.set_title("VaR 99% backtest: daily returns vs. one-day-ahead forecasts")
    _var_panel(detail, returns.loc[zoom[0]:zoom[1]], var.loc[zoom[0]:zoom[1]])
    detail.set_title(f"Zoom: {zoom[0][:7]} to {zoom[1][:7]} (Covid-19 crash)", fontsize=10)
    handles, labels = full.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    _save(fig, path)


def traffic_light(counts: pd.DataFrame, path: Path) -> None:
    """Rolling 250-day exception count per method, over the Basel zones."""
    top = max(counts.to_numpy().max() + 2, YELLOW_MAX + 4)
    fig, axes = plt.subplots(len(counts.columns), 1, figsize=(9, 1.7 * len(counts.columns)),
                             sharex=True, sharey=True)
    for ax, method in zip(axes, counts.columns):
        ax.axhspan(-0.5, GREEN_MAX + 0.5, color=ZONE_COLORS["green"], alpha=0.12, linewidth=0)
        ax.axhspan(GREEN_MAX + 0.5, YELLOW_MAX + 0.5, color=ZONE_COLORS["yellow"], alpha=0.18,
                   linewidth=0)
        ax.axhspan(YELLOW_MAX + 0.5, top, color=ZONE_COLORS["red"], alpha=0.12, linewidth=0)
        ax.plot(counts.index, counts[method], color=INK, linewidth=1.2)
        ax.set_ylim(-0.5, top)
        ax.set_title(method, loc="left", fontsize=10)
        ax.grid(False)
    axes[len(axes) // 2].set_ylabel(f"Exceptions over last {TRAFFIC_WINDOW} days")
    fig.suptitle("Basel traffic light: green 0–4, yellow 5–9, red 10+ exceptions",
                 fontweight="bold", color=INK, y=1.0)
    fig.tight_layout()
    _save(fig, path)


def snapshot(table: pd.DataFrame, path: Path) -> None:
    """Grouped bars of VaR 99% and ES 97.5% per method (in % of portfolio value)."""
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(y + 0.19, table["VaR 99%"], height=0.36, color=BLUE, label="VaR 99%")
    ax.barh(y - 0.19, table["ES 97.5%"], height=0.36, color=ORANGE, label="ES 97.5%")
    for yi, (v, e) in enumerate(zip(table["VaR 99%"], table["ES 97.5%"])):
        ax.text(v, yi + 0.19, f" {v:.2%}", va="center", fontsize=8, color=INK_SOFT)
        ax.text(e, yi - 0.19, f" {e:.2%}", va="center", fontsize=8, color=INK_SOFT)
    ax.set_yticks(y, table.index)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.set_xlim(0, table.to_numpy().max() * 1.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("One-day risk as of the last date (% of portfolio value)")
    ax.legend(loc="lower right")
    _save(fig, path)

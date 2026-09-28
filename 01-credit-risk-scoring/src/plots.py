"""Figures saved as PNG in the project's figures/ folder."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_SOFT, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK,
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


def iv_ranking(iv: pd.Series, selected: list[str], min_iv: float, path: Path) -> None:
    iv = iv.sort_values()
    colors = [BLUE if col in selected else "#c3c2b7" for col in iv.index]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(iv.index, iv.values, color=colors, height=0.7)
    ax.axvline(min_iv, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Information Value (train)")
    ax.set_title("Information Value by variable")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                       plt.Rectangle((0, 0), 1, 1, color="#c3c2b7")],
              labels=["kept in final scorecard", "not kept"], loc="lower right")
    _save(fig, path)


def roc_curves(y: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (name, pd_hat), color in zip(predictions.items(), [BLUE, ORANGE]):
        fpr, tpr, _ = roc_curve(y, pd_hat)
        ax.plot(fpr, tpr, color=color, label=name)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    _save(fig, path)


def ks_curve(y: np.ndarray, scores: np.ndarray, path: Path) -> None:
    """Cumulative share of defaulters and non-defaulters below each score."""
    grid = np.sort(np.unique(scores))
    cum_bad = np.searchsorted(np.sort(scores[y == 1]), grid, side="right") / (y == 1).sum()
    cum_good = np.searchsorted(np.sort(scores[y == 0]), grid, side="right") / (y == 0).sum()
    gap = cum_bad - cum_good
    i = int(np.argmax(gap))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(grid, cum_bad, color=ORANGE, label="defaulters")
    ax.plot(grid, cum_good, color=BLUE, label="non-defaulters")
    ax.vlines(grid[i], cum_good[i], cum_bad[i], color=INK, linewidth=1.2)
    ax.annotate(f"KS = {gap[i]:.3f}\nat {grid[i]:.0f} points", (grid[i], cum_bad[i]),
                xytext=(10, -30), textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("Scorecard points")
    ax.set_ylabel("Cumulative share")
    ax.set_title("KS curve - scorecard (test set)")
    ax.legend(loc="lower right")
    _save(fig, path)


def calibration(deciles: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    top = 0.0
    for (name, table), color in zip(deciles.items(), [BLUE, ORANGE]):
        ax.plot(table["mean_predicted_pd"], table["observed_default_rate"], color=color,
                marker="o", markersize=5, label=name)
        top = max(top, table[["mean_predicted_pd", "observed_default_rate"]].to_numpy().max())
    ax.plot([0, top], [0, top], color=MUTED, linestyle="--", linewidth=1, label="perfect calibration")
    ax.set_xlabel("Mean predicted PD (decile)")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration by decile (test set)")
    ax.legend(loc="upper left")
    _save(fig, path)


def score_distribution(y: np.ndarray, scores: np.ndarray, path: Path) -> None:
    bins = np.linspace(np.percentile(scores, 0.5), np.percentile(scores, 99.5), 40)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(scores[y == 0], bins=bins, density=True, color=BLUE, alpha=0.55,
            label="non-defaulters", edgecolor="white", linewidth=0.5)
    ax.hist(scores[y == 1], bins=bins, density=True, color=ORANGE, alpha=0.55,
            label="defaulters", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Scorecard points (higher = safer)")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution by outcome (test set)")
    ax.legend(loc="upper left")
    _save(fig, path)


def feature_importance(importance: pd.Series, path: Path, top: int = 15) -> None:
    data = importance.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(data.index, data.values / data.values.sum(), color=BLUE, height=0.7)
    ax.set_xlabel("Share of total gain")
    ax.set_title(f"LightGBM feature importance (top {top}, gain)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)

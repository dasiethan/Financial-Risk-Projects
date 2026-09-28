"""Weight of Evidence (WoE) binning and Information Value (IV).

Convention (Siddiqi): WoE = ln(%good / %bad). A positive WoE means a bin with
fewer defaults than average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SMOOTHING = 0.5  # added to counts so that an empty cell never gives log(0)


def woe_iv_table(bin_keys: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Per-bin counts, default rate, WoE and IV contribution."""
    counts = pd.crosstab(bin_keys, y).reindex(columns=[0, 1], fill_value=0)
    good = counts[0] + SMOOTHING
    bad = counts[1] + SMOOTHING
    dist_good = good / good.sum()
    dist_bad = bad / bad.sum()
    woe = np.log(dist_good / dist_bad)
    return pd.DataFrame({
        "n": counts.sum(axis=1),
        "share": counts.sum(axis=1) / counts.values.sum(),
        "default_rate": counts[1] / counts.sum(axis=1),
        "woe": woe,
        "iv": (dist_good - dist_bad) * woe,
    })


def _assign(edges: list[float], x: np.ndarray) -> np.ndarray:
    """Bin index for each value; bin i covers [edges[i-1], edges[i])."""
    return np.searchsorted(edges, x, side="right")


def _edge_to_drop(i: int, sizes: np.ndarray, n_edges: int) -> int:
    """Merging bin i with its smaller neighbour means removing one edge."""
    if i == 0:
        return 0
    if i == n_edges:
        return n_edges - 1
    return i - 1 if sizes[i - 1] < sizes[i + 1] else i


def _merge_small_bins(x: np.ndarray, edges: list[float], min_share: float) -> list[float]:
    while edges:
        shares = np.bincount(_assign(edges, x), minlength=len(edges) + 1) / len(x)
        smallest = int(np.argmin(shares))
        if shares[smallest] >= min_share:
            break
        edges.pop(_edge_to_drop(smallest, shares, len(edges)))
    return edges


def _merge_non_monotonic(x: np.ndarray, y: np.ndarray, edges: list[float]) -> list[float]:
    """Merge adjacent bins until the default rate is monotonic in x."""
    corr = pd.Series(x).corr(pd.Series(y), method="spearman")
    direction = -1.0 if corr < 0 else 1.0
    while edges:
        bins = _assign(edges, x)
        rates = np.bincount(bins, weights=y, minlength=len(edges) + 1) / np.bincount(
            bins, minlength=len(edges) + 1)
        steps = np.diff(rates) * direction
        violations = np.flatnonzero(steps < 0)
        if violations.size == 0:
            break
        worst = violations[np.argmin(np.abs(steps[violations]))]
        edges.pop(int(worst))
    return edges


def fit_numeric_edges(x: np.ndarray, y: np.ndarray, max_bins: int,
                      min_share: float) -> list[float]:
    quantiles = np.linspace(0, 1, max_bins + 1)[1:-1]
    edges = sorted(set(np.quantile(x, quantiles).tolist()))
    edges = _merge_small_bins(x, edges, min_share)
    return _merge_non_monotonic(x, y, edges)


def _interval_label(edges: list[float], i: int) -> str:
    low = "-inf" if i == 0 else f"{edges[i - 1]:,.4g}"
    high = "+inf" if i == len(edges) else f"{edges[i]:,.4g}"
    return f"[{low}, {high})"


class WoEEncoder:
    """Fit bins on training data, then replace each value by the WoE of its bin."""

    def __init__(self, categorical: list[str], max_bins: int = 10, min_share: float = 0.05):
        self.categorical = set(categorical)
        self.max_bins = max_bins
        self.min_share = min_share
        self.edges_: dict[str, list[float] | None] = {}
        self.tables_: dict[str, pd.DataFrame] = {}

    def bin_keys(self, col: str, x: pd.Series) -> pd.Series:
        edges = self.edges_[col]
        if edges is None:
            return x
        return pd.Series(_assign(edges, x.to_numpy()), index=x.index)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        for col in X.columns:
            if col in self.categorical:
                self.edges_[col] = None
            else:
                self.edges_[col] = fit_numeric_edges(
                    X[col].to_numpy(dtype=float), y.to_numpy(dtype=float),
                    self.max_bins, self.min_share)
            table = woe_iv_table(self.bin_keys(col, X[col]), y)
            table.insert(0, "bin", [self._label(col, key) for key in table.index])
            self.tables_[col] = table
        return self

    def _label(self, col: str, key) -> str:
        edges = self.edges_[col]
        return str(key) if edges is None else _interval_label(edges, int(key))

    def transform(self, X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        columns = columns or list(self.tables_)
        return pd.DataFrame({
            col: self.bin_keys(col, X[col]).map(self.tables_[col]["woe"]).fillna(0.0)
            for col in columns
        }, index=X.index)

    def information_values(self) -> pd.Series:
        iv = {col: table["iv"].sum() for col, table in self.tables_.items()}
        return pd.Series(iv, name="iv").sort_values(ascending=False)

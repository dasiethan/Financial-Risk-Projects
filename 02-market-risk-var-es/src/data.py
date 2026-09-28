"""Download daily prices from Yahoo Finance and build portfolio returns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = {
    "TTE.PA": "TotalEnergies",
    "BNP.PA": "BNP Paribas",
    "MC.PA": "LVMH",
    "AIR.PA": "Airbus",
    "SAN.PA": "Sanofi",
}
START, END = "2010-01-01", "2026-01-01"  # END is exclusive: data up to 31 Dec 2025
PRICES_FILE = "prices.csv"


def download_prices(data_dir: Path) -> pd.DataFrame:
    """Adjusted close prices (dividends and splits), cached in data/ after the first run."""
    path = data_dir / PRICES_FILE
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf

    raw = yf.download(list(TICKERS), start=START, end=END, auto_adjust=True, progress=False)
    prices = raw["Close"][list(TICKERS)].dropna()
    if prices.empty:
        raise RuntimeError("Yahoo Finance returned no data; retry later.")
    data_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
    return prices


def build_returns(prices: pd.DataFrame, weights: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    """Simple daily returns per asset, and of a portfolio rebalanced daily to constant weights."""
    asset_returns = prices.pct_change().dropna()
    portfolio = (asset_returns @ weights).rename("portfolio")
    return asset_returns, portfolio

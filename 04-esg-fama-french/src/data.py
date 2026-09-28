"""Fama-French factors (Kenneth French Data Library) and ETF monthly returns (Yahoo)."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
FACTOR_FILES = {"ff5": "F-F_Research_Data_5_Factors_2x3", "mom": "F-F_Momentum_Factor"}

# ESG ETF -> conventional ETF tracking the same universe
PAIRS = {
    "DSI": "IVV",    # iShares MSCI KLD 400 Social vs iShares Core S&P 500
    "SUSA": "SPY",   # iShares MSCI USA ESG Select vs SPDR S&P 500
    "ESGU": "IVV",   # iShares ESG Aware MSCI USA vs iShares Core S&P 500
    "ESGV": "VTI",   # Vanguard ESG U.S. Stock vs Vanguard Total Stock Market
}
START, END = "2000-01-01", "2026-01-01"  # END exclusive: last full month is Dec 2025
LAST_MONTH = "2025-12"


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _parse_french_monthly(text: str) -> pd.DataFrame:
    """Keep the first (monthly) block: rows whose first field is a YYYYMM date."""
    header, rows = None, []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if header is None and len(fields) > 1 and fields[0] == "" and fields[1]:
            header = fields[1:]
        elif header is not None and len(fields[0]) == 6 and fields[0].isdigit():
            rows.append(fields)
        elif rows:  # the monthly block ends at the first non-date line
            break
    frame = pd.DataFrame(rows).set_index(0).astype(float) / 100  # percent -> decimal
    frame.columns = header
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame


def french_factors(data_dir: Path) -> pd.DataFrame:
    """Monthly Mkt-RF, SMB, HML, RMW, CMA, RF and momentum (Mom), cached as CSV."""
    path = data_dir / "french_factors_monthly.csv"
    if path.exists():
        frame = pd.read_csv(path, index_col=0)
        frame.index = pd.PeriodIndex(frame.index, freq="M")
        return frame
    tables = []
    for name in FACTOR_FILES.values():
        archive = zipfile.ZipFile(io.BytesIO(_download(FRENCH_URL.format(name))))
        text = archive.read(archive.namelist()[0]).decode("latin-1")
        tables.append(_parse_french_monthly(text))
    factors = tables[0].join(tables[1], how="inner")
    factors.columns = [c.strip() for c in factors.columns]
    data_dir.mkdir(parents=True, exist_ok=True)
    factors.to_csv(path)
    return factors


def etf_monthly_returns(data_dir: Path) -> pd.DataFrame:
    """Month-end total returns (adjusted close) for all ETFs, cached as CSV."""
    path = data_dir / "etf_prices_daily.csv"
    if path.exists():
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        tickers = sorted(set(PAIRS) | set(PAIRS.values()))
        prices = yf.download(tickers, start=START, end=END, auto_adjust=True,
                             progress=False)["Close"]
        data_dir.mkdir(parents=True, exist_ok=True)
        prices.to_csv(path)
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change(fill_method=None)
    returns.index = returns.index.to_period("M")
    return returns.loc[:LAST_MONTH]

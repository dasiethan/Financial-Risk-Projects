# Data

## 1. Fama-French factors

| | |
|---|---|
| Source | Kenneth R. French Data Library, Dartmouth (Tuck School of Business) |
| Page | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html |
| Files | `F-F_Research_Data_5_Factors_2x3_CSV.zip` (Mkt-RF, SMB, HML, RMW, CMA, RF) and `F-F_Momentum_Factor_CSV.zip` (Mom) |
| Frequency | Monthly, in percent (converted to decimals) |
| Terms | Freely available for research and teaching, with citation of the library |

The factors are built by Fama and French from CRSP and Compustat data. They are updated
monthly and their history is occasionally revised. The analysis stops in December 2025,
even if the downloaded file goes further.

## 2. ETF prices

| | |
|---|---|
| Source | Yahoo Finance, accessed with the open-source `yfinance` library |
| Tickers | DSI, SUSA, ESGU, ESGV (ESG) and IVV, SPY, VTI (conventional) |
| Field | Daily close adjusted for dividends and splits, taken at each month-end to compute total returns |
| Period | From each ETF's inception to December 2025 |

Yahoo data are public but limited to personal, non-commercial use and cannot be
redistributed. That is why prices are **not stored in the repository**.

## Download

`python main.py` downloads both sources on the first run and caches them:

- `data/french_factors_monthly.csv`
- `data/etf_prices_daily.csv`

The `data/` folder is not versioned. Delete those files to force a fresh download.

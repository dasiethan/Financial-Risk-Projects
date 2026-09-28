# Data

| | |
|---|---|
| Source | Yahoo Finance, accessed with the open-source `yfinance` library |
| Tickers | `TTE.PA` (TotalEnergies), `BNP.PA` (BNP Paribas), `MC.PA` (LVMH), `AIR.PA` (Airbus), `SAN.PA` (Sanofi), all listed on Euronext Paris |
| Field | Daily close adjusted for dividends and splits (`auto_adjust=True`) |
| Period | 2010-01-01 to 2025-12-31 |
| Size | about 4,100 rows × 5 columns (< 1 MB) |

## Terms of use

The data are publicly accessible, but Yahoo's terms restrict them to personal, non-commercial
use and do not allow redistribution. That is why **the prices are not stored in this
repository**. The `yfinance` library is not affiliated with Yahoo. This project uses the data
for educational, non-commercial analysis only.

## Download

`python main.py` downloads the prices on the first run and caches them in `data/prices.csv`.
The `data/` folder is not versioned. Delete that file to force a fresh download.

Yahoo occasionally revises adjusted prices, for example after a new dividend. A new download
can therefore differ very slightly from the one used for the README figures. In the test
described in the README, the difference was at the 8th decimal.

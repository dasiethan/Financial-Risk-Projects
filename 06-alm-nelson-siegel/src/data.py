"""ECB euro area AAA government yield curve (spot rates and Svensson parameters)."""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{}"
MATURITIES = {"SR_3M": 0.25, "SR_6M": 0.5, "SR_9M": 0.75,
              **{f"SR_{y}Y": float(y) for y in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]}}
PARAMETERS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
START, END = "2004-09-06", "2025-12-31"
CACHE = "ecb_aaa_yield_curve.csv"


def _fetch(series: list[str]) -> pd.DataFrame:
    url = API.format("+".join(series)) + f"?startPeriod={START}&endPeriod={END}&format=csvdata"
    with urllib.request.urlopen(url, timeout=300) as response:
        raw = pd.read_csv(io.BytesIO(response.read()), usecols=["KEY", "TIME_PERIOD", "OBS_VALUE"])
    raw["series"] = raw["KEY"].str.split(".").str[-1]
    return raw.pivot(index="TIME_PERIOD", columns="series", values="OBS_VALUE")


def load_curves(data_dir: Path) -> pd.DataFrame:
    """Daily spot rates (in % , continuously compounded) and ECB Svensson parameters."""
    path = data_dir / CACHE
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    frame = _fetch(list(MATURITIES)).join(_fetch(PARAMETERS))
    frame.index = pd.to_datetime(frame.index)
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return frame


def month_ends(frame: pd.DataFrame) -> pd.DataFrame:
    """Last available business day of each month."""
    return frame.groupby(frame.index.to_period("M")).tail(1)

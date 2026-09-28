"""Download, clean and enrich the UCI 'Default of Credit Card Clients' dataset."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
RAW_FILE = "default_of_credit_card_clients.xls"
TARGET = "default"
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]


def download_raw(data_dir: Path) -> Path:
    """Download the zip archive from UCI once and keep only the .xls file."""
    data_dir.mkdir(parents=True, exist_ok=True)
    xls_path = data_dir / RAW_FILE
    if xls_path.exists():
        return xls_path
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    member = next(name for name in archive.namelist() if name.endswith(".xls"))
    xls_path.write_bytes(archive.read(member))
    return xls_path


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise column names and undocumented category codes."""
    df = df.rename(columns={"default payment next month": TARGET, "PAY_0": "PAY_1"})
    df = df.drop(columns="ID")
    # EDUCATION codes 0, 5, 6 and MARRIAGE code 0 are not documented: map to "other".
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].between(1, 3), 4)
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2]), 3)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few behavioural ratios commonly used in retail credit scoring."""
    df = df.copy()
    df["MAX_DELAY_6M"] = df[PAY_COLS].max(axis=1)
    df["N_MONTHS_LATE_6M"] = (df[PAY_COLS] > 0).sum(axis=1)
    df["UTILIZATION_1"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    df["AVG_UTILIZATION_6M"] = df[BILL_COLS].mean(axis=1) / df["LIMIT_BAL"]
    # Payment made in month t settles the bill of month t+1 (columns run backwards in time).
    df["PAY_RATIO_1"] = df["PAY_AMT1"] / df["BILL_AMT2"].clip(lower=1)
    return df


def load_dataset(data_dir: Path) -> pd.DataFrame:
    raw = pd.read_excel(download_raw(data_dir), header=1)
    return add_features(clean(raw))

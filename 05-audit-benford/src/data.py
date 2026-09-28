"""Download and clean French public procurement contracts (DECP, 2022 schema)."""
from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = ("https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/"
       "decp-2022-marches-valides/exports/csv")
FIELDS = ["id", "acheteur_id", "nature", "procedure", "codecpv", "montant", "dureemois",
          "datenotification", "datepublicationdonnees", "offresrecues", "formeprix",
          "titulaire_id_1", "objet"]
# Contracts notified in 2025, as published up to a fixed date so the extract is stable.
YEAR, PUBLISHED_BY = 2025, "2026-09-01"
RAW_FILE = f"decp_{YEAR}.csv"
# Framework agreements awarded to several suppliers repeat the same amount once per
# supplier: one record per buyer, contract id, amount and notification date is kept.
CONTRACT_KEY = ["acheteur_id", "id", "montant", "datenotification"]


def download(data_dir: Path) -> Path:
    path = data_dir / RAW_FILE
    if path.exists():
        return path
    params = {
        "select": ",".join(FIELDS),
        "where": f"year(datenotification)={YEAR} and "
                 f"datepublicationdonnees<=date'{PUBLISHED_BY}'",
        "delimiter": ";",
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    url = f"{API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=600) as response:
        path.write_bytes(response.read())
    return path


def load_contracts(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(download(data_dir), sep=";", dtype=str)
    df = raw.drop_duplicates(subset=CONTRACT_KEY).copy()
    df["montant"] = pd.to_numeric(df["montant"], errors="coerce")
    df["dureemois"] = pd.to_numeric(df["dureemois"], errors="coerce")
    df["offers"] = pd.to_numeric(df["offresrecues"], errors="coerce")  # "MQ NC" -> NaN
    for col in ["datenotification", "datepublicationdonnees"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["cpv_division"] = df["codecpv"].str[:2]
    df["log_amount"] = np.log10(df["montant"].clip(lower=1))
    info = {"raw_rows": len(raw), "contracts": len(df), "buyers": int(df["acheteur_id"].nunique()),
            "suppliers": int(df["titulaire_id_1"].nunique()),
            "total_amount_eur": float(df["montant"].sum())}
    return df.reset_index(drop=True), info

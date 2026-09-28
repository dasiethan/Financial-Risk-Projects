"""Load and reconcile the freMTPL2 frequency and severity tables (OpenML)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

FREQ_ID, SEV_ID = 41214, 41215
MAX_CLAIMS = 4           # more than 4 claims in a year is treated as a recording error
LARGE_CLAIM_CAP = 100_000  # EUR; the excess above it is priced through a loading


def fetch_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download once from OpenML; scikit-learn caches the files under data_dir."""
    freq = fetch_openml(data_id=FREQ_ID, as_frame=True, data_home=str(data_dir)).frame
    sev = fetch_openml(data_id=SEV_ID, as_frame=True, data_home=str(data_dir)).frame
    return freq, sev


def reconcile(freq: pd.DataFrame, sev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per policy with claim count and amounts that are consistent with each other.

    The number of claims is taken as the number of recorded payments, because the pure
    premium must be compared with observed amounts; claims declared without any
    payment record are counted as zero-cost claims (i.e. not counted).
    """
    freq = freq.copy()
    freq["IDpol"] = freq["IDpol"].astype(int)
    sev = sev.assign(IDpol=sev["IDpol"].astype(int))
    sev = sev[sev["IDpol"].isin(freq["IDpol"])]
    sev = sev.assign(ClaimAmountCapped=sev["ClaimAmount"].clip(upper=LARGE_CLAIM_CAP))
    per_policy = sev.groupby("IDpol").agg(
        ClaimNbPaid=("ClaimAmount", "size"),
        ClaimAmount=("ClaimAmount", "sum"),
        ClaimAmountCapped=("ClaimAmountCapped", "sum"),
    )
    df = freq.join(per_policy, on="IDpol")
    stats = {
        "n_policies": len(df),
        "n_severity_records": len(sev),
        "policies_with_declared_claim": int((df["ClaimNb"] > 0).sum()),
        "policies_with_declared_claim_but_no_amount": int(
            ((df["ClaimNb"] > 0) & df["ClaimNbPaid"].isna()).sum()),
        "policies_with_exposure_above_1": int((df["Exposure"] > 1).sum()),
        "claims_above_cap": int((sev["ClaimAmount"] > LARGE_CLAIM_CAP).sum()),
        "share_of_amount_above_cap": float(
            (sev["ClaimAmount"] - sev["ClaimAmountCapped"]).sum() / sev["ClaimAmount"].sum()),
    }
    df["ClaimNb"] = df["ClaimNbPaid"].fillna(0).clip(upper=MAX_CLAIMS).astype(int)
    df[["ClaimAmount", "ClaimAmountCapped"]] = df[["ClaimAmount", "ClaimAmountCapped"]].fillna(0.0)
    df["Exposure"] = df["Exposure"].clip(upper=1.0)
    df["VehGas"] = df["VehGas"].astype(str).str.strip("'")
    return df.drop(columns=["ClaimNbPaid"]), stats


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    return reconcile(*fetch_tables(data_dir))


def glm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Actuarial banding and transformations for the GLMs (inspired by Noll, Salzmann and
    Wüthrich, 2018): bands where the effect is non-linear, logs for skewed variables."""
    out = df.copy()
    out["VehPowerGrp"] = df["VehPower"].clip(upper=9).astype(str)
    out["VehAgeGrp"] = pd.cut(df["VehAge"], [-1, 0, 4, 9, 14, np.inf],
                              labels=["0", "1-4", "5-9", "10-14", "15+"]).astype(str)
    out["DrivAgeGrp"] = pd.cut(df["DrivAge"], [17, 20, 25, 30, 40, 50, 60, 70, np.inf],
                               labels=["18-20", "21-25", "26-30", "31-40", "41-50",
                                       "51-60", "61-70", "71+"]).astype(str)
    out["BonusMalusCap"] = df["BonusMalus"].clip(upper=150)
    out["LogDensity"] = np.log(df["Density"])
    return out

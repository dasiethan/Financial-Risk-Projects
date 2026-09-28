"""Rule-based audit controls and an Isolation Forest to rank unusual contracts.

A flag is a reason to look at a record, not evidence of wrongdoing: most flags in
open procurement data come from data-entry or reporting conventions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

COMPETITIVE = {"Appel d'offres ouvert", "Appel d'offres restreint", "Procédure adaptée",
               "Procédure avec négociation", "Dialogue compétitif"}
NO_COMPETITION = "Marché passé sans publicité ni mise en concurrence préalable"
SPLIT_LIMIT = 40_000


def possible_splits(df: pd.DataFrame) -> pd.Series:
    """Same buyer, supplier, CPV division and month: several contracts each below
    40,000 EUR whose total reaches 40,000 EUR (a classic 'split purchase' test)."""
    small = df[df["montant"] < SPLIT_LIMIT].copy()
    small["month"] = small["datenotification"].dt.to_period("M")
    group = ["acheteur_id", "titulaire_id_1", "cpv_division", "month"]
    stats = small.groupby(group)["montant"].agg(["size", "sum"])
    suspicious = stats[(stats["size"] >= 2) & (stats["sum"] >= SPLIT_LIMIT)].index
    flagged = small.set_index(group).index.isin(suspicious)
    return pd.Series(df.index.isin(small.index[flagged]), index=df.index)


def rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    delay = (df["datepublicationdonnees"] - df["datenotification"]).dt.days
    competitive = df["procedure"].isin(COMPETITIVE)
    exact_dup = df.duplicated(["acheteur_id", "titulaire_id_1", "montant", "datenotification"],
                              keep=False)
    return pd.DataFrame({
        "amount_below_10_eur": df["montant"] < 10,
        "amount_100m_eur_or_more": df["montant"] >= 1e8,
        "published_before_notification": delay < 0,
        "published_over_2_months_late": delay > 61,
        "offers_missing_or_invalid": df["offers"].isna(),
        "single_offer_competitive_procedure": competitive & (df["offers"] == 1),
        "no_competition_40k_or_more": (df["procedure"] == NO_COMPETITION)
                                      & (df["montant"] >= SPLIT_LIMIT),
        "same_buyer_supplier_amount_date": exact_dup,
        "possible_split_purchase": possible_splits(df),
    })


def isolation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric description of each contract. The amount is also compared with the
    typical amount of its CPV division (what is bought)."""
    division_median = df.groupby("cpv_division")["log_amount"].transform("median")
    return pd.DataFrame({
        "log_amount": df["log_amount"],
        "log_amount_vs_cpv_median": df["log_amount"] - division_median,
        "duration_months": df["dureemois"].fillna(df["dureemois"].median()),
        "offers": df["offers"].fillna(-1),
        "offers_missing": df["offers"].isna().astype(int),
        "procedure_code": df["procedure"].astype("category").cat.codes,
        "price_form_code": df["formeprix"].astype("category").cat.codes,
    })


def isolation_scores(df: pd.DataFrame, seed: int, contamination: float) -> pd.Series:
    """Higher score = more isolated (easier to separate from the rest)."""
    features = isolation_features(df)
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    model.fit(features)
    return pd.Series(-model.score_samples(features), index=df.index, name="anomaly_score")

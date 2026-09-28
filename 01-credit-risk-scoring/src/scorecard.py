"""Feature selection, logistic regression on WoE and conversion to scorecard points."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .woe import WoEEncoder

# Scaling: 600 points at good:bad odds of 50:1, and 20 more points double the odds.
BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def select_by_iv_and_correlation(iv: pd.Series, woe: pd.DataFrame, min_iv: float,
                                 max_corr: float) -> list[str]:
    """Keep variables with IV >= min_iv; among highly correlated ones keep the higher IV."""
    selected: list[str] = []
    for col in iv[iv >= min_iv].index:  # iv is sorted by decreasing IV
        if all(abs(woe[col].corr(woe[kept])) < max_corr for kept in selected):
            selected.append(col)
    return selected


def fit_logit(woe: pd.DataFrame, y: pd.Series):
    return sm.Logit(y, sm.add_constant(woe)).fit(disp=0)


def fit_with_sign_and_pvalue_checks(woe: pd.DataFrame, y: pd.Series,
                                    max_pvalue: float) -> tuple[object, list[str]]:
    """Refit, dropping one variable at a time, until every coefficient is negative
    (with WoE = ln(good/bad), a higher WoE must lower the default probability)
    and significant."""
    features = list(woe.columns)
    while True:
        result = fit_logit(woe[features], y)
        coefs = result.params.drop("const")
        pvalues = result.pvalues.drop("const")
        wrong_sign = coefs[coefs >= 0]
        if not wrong_sign.empty:
            features.remove(wrong_sign.idxmax())
            continue
        if pvalues.max() > max_pvalue:
            features.remove(pvalues.idxmax())
            continue
        return result, features


def predict_pd(result, woe: pd.DataFrame) -> np.ndarray:
    return np.asarray(result.predict(sm.add_constant(woe, has_constant="add")))


def score_points(result, woe: pd.DataFrame) -> np.ndarray:
    """score = OFFSET + FACTOR * ln(odds_good) = OFFSET - FACTOR * logit(PD)."""
    logit = np.asarray(sm.add_constant(woe, has_constant="add") @ result.params)
    return OFFSET - FACTOR * logit


def scorecard_table(result, encoder: WoEEncoder, features: list[str]) -> pd.DataFrame:
    """Points per attribute; the intercept is spread evenly across variables."""
    intercept_share = (OFFSET - FACTOR * result.params["const"]) / len(features)
    rows = []
    for col in features:
        table = encoder.tables_[col]
        points = -FACTOR * result.params[col] * table["woe"] + intercept_share
        for (_, row), pts in zip(table.iterrows(), points):
            rows.append({"variable": col, "bin": row["bin"], "share_train": row["share"],
                         "default_rate_train": row["default_rate"], "woe": row["woe"],
                         "points": int(round(pts))})
    return pd.DataFrame(rows)

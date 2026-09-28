"""Factor regressions with Newey-West (HAC) standard errors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTOR_MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5+Mom": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}
HAC_LAGS = 6          # six months of autocorrelation allowed in the residuals
Z_95, Z_POWER_80 = 1.96, 0.84


def fit_factor_model(y: pd.Series, factors: pd.DataFrame, model: str):
    """OLS of a monthly (excess or long-short) return on the model's factors."""
    X = sm.add_constant(factors.loc[y.index, FACTOR_MODELS[model]])
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def summarize(result, fund: str, kind: str, model: str) -> dict:
    """Alpha annualised as 12 x monthly alpha, plus the smallest annual alpha this sample
    could detect (5% two-sided test, 80% power)."""
    se_alpha = result.bse["const"]
    row = {
        "fund": fund, "kind": kind, "model": model, "n_months": int(result.nobs),
        "alpha_annual": 12 * result.params["const"],
        "alpha_ci_low": 12 * (result.params["const"] - Z_95 * se_alpha),
        "alpha_ci_high": 12 * (result.params["const"] + Z_95 * se_alpha),
        "alpha_t": result.tvalues["const"], "alpha_p": result.pvalues["const"],
        "min_detectable_alpha_annual": 12 * (Z_95 + Z_POWER_80) * se_alpha,
        "adj_r2": result.rsquared_adj,
    }
    for factor in FACTOR_MODELS[model]:
        row[f"beta_{factor}"] = result.params[factor]
        row[f"t_{factor}"] = result.tvalues[factor]
        row[f"se_{factor}"] = result.bse[factor]
    return row


def performance(returns: pd.Series, rf: pd.Series) -> dict:
    """Annualised statistics on monthly returns."""
    growth = (1 + returns).cumprod()
    excess = returns - rf.loc[returns.index]
    return {
        "annual_return": float(growth.iloc[-1] ** (12 / len(returns)) - 1),
        "annual_volatility": float(returns.std() * np.sqrt(12)),
        "sharpe": float(excess.mean() / excess.std() * np.sqrt(12)),
        "max_drawdown": float((growth / growth.cummax() - 1).min()),
    }

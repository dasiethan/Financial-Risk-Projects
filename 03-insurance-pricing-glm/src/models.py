"""Frequency and severity models: GLMs (statsmodels) and LightGBM challengers."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

RATING_FACTORS = ["C(VehPowerGrp)", "C(VehAgeGrp, Treatment('5-9'))",
                  "C(DrivAgeGrp, Treatment('41-50'))", "BonusMalusCap", "C(VehGas)",
                  "LogDensity", "C(Area)", "C(VehBrand, Treatment('B1'))",
                  "C(Region, Treatment('R24'))"]
MAX_PVALUE = 0.05
RAW_FEATURES = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand",
                "VehGas", "Density", "Region"]
CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]


# ---------- GLMs ----------

def backward_selection(fit, factors: list[str]) -> tuple[object, list[str], list[str]]:
    """Drop, one at a time, the rating factor with the highest joint Wald p-value until
    every remaining factor is significant. Uses the training data only."""
    factors, dropped = list(factors), []
    while True:
        result = fit(factors)
        pvalues = result.wald_test_terms(skip_single=False, scalar=True).table["pvalue"]
        pvalues = pvalues.drop("Intercept")
        if pvalues.max() <= MAX_PVALUE or len(factors) == 1:
            return result, factors, dropped
        worst = pvalues.idxmax()
        factors.remove(worst)
        dropped.append(worst)


def fit_frequency_glm(train: pd.DataFrame):
    """Poisson GLM, log link, log(exposure) offset: E[N] = exposure * exp(x'b)."""
    def fit(factors):
        return smf.glm(f"ClaimNb ~ {' + '.join(factors)}", data=train,
                       family=sm.families.Poisson(), offset=np.log(train["Exposure"])).fit()
    return backward_selection(fit, RATING_FACTORS)


def fit_severity_glm(claims: pd.DataFrame):
    """Gamma GLM, log link, on the average (capped) cost per claim, weighted by the
    number of claims."""
    def fit(factors):
        return smf.glm(f"AvgClaim ~ {' + '.join(factors)}", data=claims,
                       family=sm.families.Gamma(link=sm.families.links.Log()),
                       var_weights=claims["ClaimNb"]).fit()
    return backward_selection(fit, RATING_FACTORS)


def glm_frequency_rate(model, df: pd.DataFrame) -> np.ndarray:
    """Expected claims per policy-year (offset of zero = one full year)."""
    return np.asarray(model.predict(df, offset=np.zeros(len(df))))


def glm_severity(model, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(df))


# ---------- LightGBM ----------

LGBM_PARAMS = dict(learning_rate=0.03, num_leaves=31, min_child_samples=200,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   reg_lambda=1.0, verbose=-1)


def lgbm_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[RAW_FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype("category")
    return X


def _fit_lgbm(X, y, weight, objective: str, seed: int) -> lgb.LGBMRegressor:
    """Early stopping on a validation split of the training data, then refit on all of it."""
    X_fit, X_val, y_fit, y_val, w_fit, w_val = train_test_split(
        X, y, weight, test_size=0.2, random_state=seed)
    probe = lgb.LGBMRegressor(objective=objective, n_estimators=5000, random_state=seed,
                              **LGBM_PARAMS)
    probe.fit(X_fit, y_fit, sample_weight=w_fit, eval_X=X_val, eval_y=y_val,
              eval_sample_weight=[w_val], callbacks=[lgb.early_stopping(100, verbose=False)])
    model = lgb.LGBMRegressor(objective=objective, n_estimators=probe.best_iteration_,
                              random_state=seed, **LGBM_PARAMS)
    return model.fit(X, y, sample_weight=weight)


def fit_frequency_lgbm(train: pd.DataFrame, seed: int) -> lgb.LGBMRegressor:
    """Poisson objective on the claim rate N / exposure, weighted by exposure
    (equivalent to a Poisson likelihood on N with a log-exposure offset)."""
    return _fit_lgbm(lgbm_frame(train), train["ClaimNb"] / train["Exposure"],
                     train["Exposure"], "poisson", seed)


def fit_severity_lgbm(claims: pd.DataFrame, seed: int) -> lgb.LGBMRegressor:
    return _fit_lgbm(lgbm_frame(claims), claims["AvgClaim"], claims["ClaimNb"], "gamma", seed)

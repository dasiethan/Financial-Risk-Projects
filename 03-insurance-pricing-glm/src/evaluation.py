"""Deviance, Lorenz/Gini, balance and permutation importance for pricing models."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance


def poisson_deviance(df: pd.DataFrame, rate: np.ndarray) -> float:
    """Exposure-weighted mean Poisson deviance of the claim rate."""
    return mean_poisson_deviance(df["ClaimNb"] / df["Exposure"], rate,
                                 sample_weight=df["Exposure"])


def gamma_deviance(claims: pd.DataFrame, severity: np.ndarray) -> float:
    return mean_gamma_deviance(claims["AvgClaim"], severity, sample_weight=claims["ClaimNb"])


def lorenz_curve(amount: np.ndarray, exposure: np.ndarray, pure_premium: np.ndarray):
    """Policies sorted from lowest to highest predicted premium; cumulative share of
    exposure (x) vs cumulative share of observed claim cost (y)."""
    order = np.argsort(pure_premium)
    x = np.concatenate([[0], np.cumsum(exposure[order]) / exposure.sum()])
    y = np.concatenate([[0], np.cumsum(amount[order]) / amount.sum()])
    return x, y


def lorenz_gini(amount: np.ndarray, exposure: np.ndarray, pure_premium: np.ndarray) -> float:
    """1 - 2 x area under the Lorenz curve: 0 = no segmentation, higher = better ranking."""
    x, y = lorenz_curve(amount, exposure, pure_premium)
    return float(1 - 2 * np.trapezoid(y, x))


def premium_deciles(df: pd.DataFrame, pure_premium: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted cost per policy-year, by decile of predicted pure premium
    (deciles contain equal exposure)."""
    frame = pd.DataFrame({"exposure": df["Exposure"].to_numpy(),
                          "observed": df["ClaimAmountCapped"].to_numpy(),
                          "predicted": pure_premium * df["Exposure"].to_numpy(),
                          "premium": pure_premium}).sort_values("premium")
    frame["decile"] = np.minimum(
        (frame["exposure"].cumsum() / frame["exposure"].sum() * n_bins).astype(int), n_bins - 1) + 1
    grouped = frame.groupby("decile")[["exposure", "observed", "predicted"]].sum()
    return pd.DataFrame({
        "exposure": grouped["exposure"],
        "observed_pure_premium": grouped["observed"] / grouped["exposure"],
        "predicted_pure_premium": grouped["predicted"] / grouped["exposure"],
    })


def permutation_importance(df: pd.DataFrame, predict_rate: Callable[[pd.DataFrame], np.ndarray],
                           features: list[str], rng: np.random.Generator,
                           n_repeats: int = 3) -> pd.Series:
    """Increase in Poisson deviance when one raw rating factor is shuffled."""
    baseline = poisson_deviance(df, predict_rate(df))
    increases = {}
    for col in features:
        scores = []
        for _ in range(n_repeats):
            shuffled = df.copy()
            shuffled[col] = pd.Series(rng.permutation(df[col].to_numpy()), index=df.index,
                                      dtype=df[col].dtype)
            scores.append(poisson_deviance(df, predict_rate(shuffled)) - baseline)
        increases[col] = float(np.mean(scores))
    return pd.Series(increases).sort_values(ascending=False)

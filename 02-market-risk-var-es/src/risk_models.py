"""One-day Value-at-Risk and Expected Shortfall estimators.

Sign convention: r is a daily return, the loss is L = -r, and VaR / ES are reported
as positive fractions of portfolio value.
    VaR_a = a-quantile of L          ES_a = E[L | L >= VaR_a]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

GARCH_SCALE = 100.0  # arch is numerically happier with returns in percent


@dataclass
class RiskEstimate:
    var: float
    es: float


def historical(r: np.ndarray, alpha: float) -> RiskEstimate:
    losses = -np.asarray(r)
    var = np.quantile(losses, alpha)
    return RiskEstimate(var, losses[losses >= var].mean())


def normal(r: np.ndarray, alpha: float) -> RiskEstimate:
    mu, sigma = np.mean(r), np.std(r, ddof=1)
    z = stats.norm.ppf(alpha)
    return RiskEstimate(-mu + sigma * z, -mu + sigma * stats.norm.pdf(z) / (1 - alpha))


def t_tail(alpha: float, nu: float) -> RiskEstimate:
    """VaR and ES of the loss for a standard Student-t (location 0, scale 1)."""
    q = stats.t.ppf(alpha, nu)
    es = stats.t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1)
    return RiskEstimate(q, es)


def student_t(r: np.ndarray, alpha: float) -> RiskEstimate:
    nu, loc, scale = stats.t.fit(r)
    tail = t_tail(alpha, nu)
    return RiskEstimate(-loc + scale * tail.var, -loc + scale * tail.es)


def monte_carlo_t(asset_returns: np.ndarray, weights: np.ndarray, alpha: float, nu: float,
                  n_sims: int, rng: np.random.Generator) -> RiskEstimate:
    """Simulate joint asset returns from a multivariate Student-t with the sample mean and
    covariance, then read VaR/ES on the simulated portfolio returns."""
    mean = asset_returns.mean(axis=0)
    chol = np.linalg.cholesky(np.cov(asset_returns, rowvar=False))
    gaussian = rng.standard_normal((n_sims, len(mean))) @ chol.T
    # A t variable is a Gaussian divided by sqrt(chi2/nu); rescale so the covariance matches.
    mixing = np.sqrt(rng.chisquare(nu, size=(n_sims, 1)) / nu)
    simulated = mean + gaussian / mixing * np.sqrt((nu - 2) / nu)
    return historical(simulated @ weights, alpha)


@dataclass
class GarchForecast:
    """One-step-ahead GARCH(1,1) forecast (in return units, not percent)."""
    mu: float
    sigma: float
    nu: float
    std_resid: np.ndarray
    params: pd.Series


def fit_garch(r: pd.Series, params: pd.Series | None = None) -> GarchForecast:
    """Estimate a constant-mean GARCH(1,1) with Student-t innovations, or reuse given
    parameters (only the volatility filter is rerun)."""
    model = arch_model(r * GARCH_SCALE, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    result = model.fit(disp="off") if params is None else model.fix(params)
    forecast = result.forecast(horizon=1, reindex=False)
    return GarchForecast(
        mu=float(forecast.mean.iloc[-1, 0]) / GARCH_SCALE,
        sigma=float(np.sqrt(forecast.variance.iloc[-1, 0])) / GARCH_SCALE,
        nu=float(result.params["nu"]),
        std_resid=np.asarray(result.std_resid.dropna()),
        params=result.params,
    )


def garch_t(f: GarchForecast, alpha: float) -> RiskEstimate:
    """Parametric: innovations follow a unit-variance Student-t."""
    tail = t_tail(alpha, f.nu)
    unit_variance = np.sqrt((f.nu - 2) / f.nu)
    return RiskEstimate(-f.mu + f.sigma * unit_variance * tail.var,
                        -f.mu + f.sigma * unit_variance * tail.es)


def filtered_historical(f: GarchForecast, alpha: float) -> RiskEstimate:
    """Filtered historical simulation: empirical tail of standardised residuals x GARCH vol."""
    z_tail = historical(f.std_resid, alpha)
    return RiskEstimate(-f.mu + f.sigma * z_tail.var, -f.mu + f.sigma * z_tail.es)

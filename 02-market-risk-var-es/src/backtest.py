"""Rolling one-day VaR forecasts and regulatory backtests (Kupiec, Christoffersen, Basel)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from . import risk_models as rm

# Basel Committee (1996) traffic light for 250 observations at 99% VaR
GREEN_MAX, YELLOW_MAX = 4, 9
TRAFFIC_WINDOW = 250


def rolling_var(r: pd.Series, window: int, alpha: float, refit_every: int) -> pd.DataFrame:
    """VaR for day t estimated only with the `window` returns before t.
    GARCH parameters are re-estimated every `refit_every` days; in between, the
    volatility is re-filtered with the last parameters (standard practice, much faster)."""
    rows, params = [], None
    for i, t in enumerate(range(window, len(r))):
        sample = r.iloc[t - window:t]
        values = sample.to_numpy()
        garch = rm.fit_garch(sample, None if i % refit_every == 0 else params)
        params = garch.params
        rows.append({
            "Historical": rm.historical(values, alpha).var,
            "Normal": rm.normal(values, alpha).var,
            "Student-t": rm.student_t(values, alpha).var,
            "GARCH-t": rm.garch_t(garch, alpha).var,
            "FHS (GARCH)": rm.filtered_historical(garch, alpha).var,
            "garch_sigma": garch.sigma,
        })
    return pd.DataFrame(rows, index=r.index[window:])


def kupiec_pof(exceptions: np.ndarray, alpha: float) -> tuple[float, float]:
    """Proportion-of-failures likelihood ratio: is the exception rate equal to 1 - alpha?"""
    n, x = len(exceptions), int(exceptions.sum())
    p, p_hat = 1 - alpha, x / n
    log_l0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_l1 = xlogy(n - x, 1 - p_hat) + xlogy(x, p_hat)
    lr = -2 * (log_l0 - log_l1)
    return lr, stats.chi2.sf(lr, df=1)


def christoffersen_independence(exceptions: np.ndarray) -> tuple[float, float]:
    """Likelihood ratio: does an exception today change the probability of one tomorrow?"""
    prev, curr = exceptions[:-1], exceptions[1:]
    n00 = np.sum(~prev & ~curr)
    n01 = np.sum(~prev & curr)
    n10 = np.sum(prev & ~curr)
    n11 = np.sum(prev & curr)
    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l1 = (xlogy(n00, 1 - pi0) + xlogy(n01, pi0)
              + xlogy(n10, 1 - pi1) + xlogy(n11, pi1))
    lr = -2 * (log_l0 - log_l1)
    return lr, stats.chi2.sf(lr, df=1)


def basel_zone(n_exceptions: int) -> str:
    if n_exceptions <= GREEN_MAX:
        return "green"
    return "yellow" if n_exceptions <= YELLOW_MAX else "red"


def rolling_exception_counts(exceptions: pd.DataFrame) -> pd.DataFrame:
    return exceptions.astype(int).rolling(TRAFFIC_WINDOW).sum().dropna()


def backtest_summary(losses: pd.Series, var: pd.DataFrame, alpha: float) -> pd.DataFrame:
    exceptions = var.lt(losses, axis=0)
    counts = rolling_exception_counts(exceptions)
    rows = {}
    for method in var.columns:
        hits = exceptions[method].to_numpy()
        lr_pof, p_pof = kupiec_pof(hits, alpha)
        lr_ind, p_ind = christoffersen_independence(hits)
        zones = counts[method].map(basel_zone)
        last_count = int(counts[method].iloc[-1])
        rows[method] = {
            "n_days": len(hits),
            "exceptions": int(hits.sum()),
            "expected": round(len(hits) * (1 - alpha), 1),
            "exception_rate": hits.mean(),
            "kupiec_lr": lr_pof,
            "kupiec_pvalue": p_pof,
            "christoffersen_ind_pvalue": p_ind,
            "conditional_coverage_pvalue": stats.chi2.sf(lr_pof + lr_ind, df=2),
            "last_250d_exceptions": last_count,
            "last_250d_zone": basel_zone(last_count),
            "share_windows_green": (zones == "green").mean(),
            "share_windows_yellow": (zones == "yellow").mean(),
            "share_windows_red": (zones == "red").mean(),
        }
    return pd.DataFrame(rows).T

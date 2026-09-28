"""Nelson-Siegel and Svensson zero-coupon curves, and Nelson-Siegel estimation.

    y(m) = b0 + b1 * f1(m/tau) + b2 * f2(m/tau)
    f1(x) = (1 - exp(-x)) / x          slope loading (1 at m = 0, -> 0 at long maturities)
    f2(x) = f1(x) - exp(-x)            curvature loading (hump at medium maturities)

b0 is the long-term level, b0 + b1 the instantaneous short rate, b2 the hump.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

TAU_GRID = np.linspace(0.2, 10.0, 197)  # years


def ns_loadings(m: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(m, dtype=float) / tau
    f1 = (1 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(x), f1, f1 - np.exp(-x)])


def ns_yield(m: np.ndarray, betas: np.ndarray, tau: float) -> np.ndarray:
    return ns_loadings(m, tau) @ betas


def svensson_yield(m: np.ndarray, b0: float, b1: float, b2: float, b3: float,
                   tau1: float, tau2: float) -> np.ndarray:
    base = ns_yield(m, np.array([b0, b1, b2]), tau1)
    return base + b3 * ns_loadings(m, tau2)[:, 2]


@dataclass
class NSFit:
    betas: np.ndarray
    tau: float
    rmse: float


def _ols(m: np.ndarray, y: np.ndarray, tau: float) -> tuple[np.ndarray, float]:
    X = ns_loadings(m, tau)
    betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    return betas, float(np.sum((y - X @ betas) ** 2))


def fit_fixed_tau(m: np.ndarray, y: np.ndarray, tau: float) -> NSFit:
    """Diebold-Li approach: with tau fixed the model is linear, so OLS gives the betas."""
    betas, sse = _ols(m, y, tau)
    return NSFit(betas, tau, float(np.sqrt(sse / len(y))))


def fit_nelson_siegel(m: np.ndarray, y: np.ndarray) -> NSFit:
    """Profile least squares: for each tau the betas are OLS, so only tau is searched.
    A grid search avoids local minima, then a bounded 1-D search refines it."""
    sse_grid = [_ols(m, y, tau)[1] for tau in TAU_GRID]
    i = int(np.argmin(sse_grid))
    low, high = TAU_GRID[max(i - 1, 0)], TAU_GRID[min(i + 1, len(TAU_GRID) - 1)]
    tau = minimize_scalar(lambda t: _ols(m, y, t)[1], bounds=(low, high), method="bounded").x
    return fit_fixed_tau(m, y, float(tau))

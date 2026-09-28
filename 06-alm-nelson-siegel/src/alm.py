"""Stylised ALM: present values, durations and surplus sensitivity to curve shocks.

Rates are continuously compounded decimals: discount factor = exp(-y(t) * t).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Curve = Callable[[np.ndarray], np.ndarray]  # maturity (years) -> zero rate (decimal)
BP = 1e-4


def liability_cash_flows(total: float, years: int) -> pd.Series:
    """Run-off of an annuity-like book: payments decrease linearly over `years`."""
    t = np.arange(1, years + 1, dtype=float)
    weights = years + 1 - t
    return pd.Series(total * weights / weights.sum(), index=t)


def bond_cash_flows(face: float, coupon: float, maturity: int) -> pd.Series:
    t = np.arange(1, maturity + 1, dtype=float)
    flows = np.full(maturity, face * coupon)
    flows[-1] += face
    return pd.Series(flows, index=t)


def present_value(flows: pd.Series, curve: Curve) -> float:
    t = flows.index.to_numpy(dtype=float)
    return float(np.sum(flows.to_numpy() * np.exp(-curve(t) * t)))


def shifted(curve: Curve, shift: Curve) -> Curve:
    return lambda t: curve(t) + shift(t)


def effective_duration_convexity(flows: pd.Series, curve: Curve, bump: float = BP) -> tuple[float, float]:
    """Finite differences with a parallel bump of the zero curve."""
    pv0 = present_value(flows, curve)
    up = present_value(flows, shifted(curve, lambda t: np.full_like(t, bump)))
    down = present_value(flows, shifted(curve, lambda t: np.full_like(t, -bump)))
    return (down - up) / (2 * pv0 * bump), (up + down - 2 * pv0) / (pv0 * bump**2)


def scale_to(flows: pd.Series, target_pv: float, curve: Curve) -> pd.Series:
    return flows * target_pv / present_value(flows, curve)


def duration_matched_mix(short: pd.Series, long: pd.Series, target_pv: float,
                         target_duration: float, curve: Curve) -> pd.Series:
    """Weights of two bonds so that the portfolio has the target PV and duration
    (classical Redington immunisation against parallel shifts)."""
    d_short = effective_duration_convexity(short, curve)[0]
    d_long = effective_duration_convexity(long, curve)[0]
    w_long = (target_duration - d_short) / (d_long - d_short)
    mix = scale_to(short, target_pv * (1 - w_long), curve).add(
        scale_to(long, target_pv * w_long, curve), fill_value=0.0)
    return mix


def surplus_impacts(assets: dict[str, pd.Series], liabilities: pd.Series, curve: Curve,
                    shocks: dict[str, Curve]) -> pd.DataFrame:
    """Change in surplus (assets - liabilities) for each shock, in currency units."""
    rows = {}
    for name, shock in shocks.items():
        stressed = shifted(curve, shock)
        d_liab = present_value(liabilities, stressed) - present_value(liabilities, curve)
        rows[name] = {portfolio: present_value(flows, stressed) - present_value(flows, curve)
                      - d_liab for portfolio, flows in assets.items()}
    return pd.DataFrame(rows).T

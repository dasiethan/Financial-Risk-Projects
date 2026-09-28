"""Benford's law tests (first digit, first two digits) with Nigrini's MAD thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Mean absolute deviation cut-offs from M. Nigrini, "Benford's Law" (2012)
MAD_THRESHOLDS = {
    1: [(0.006, "close conformity"), (0.012, "acceptable conformity"),
        (0.015, "marginally acceptable"), (np.inf, "nonconformity")],
    2: [(0.0012, "close conformity"), (0.0018, "acceptable conformity"),
        (0.0022, "marginally acceptable"), (np.inf, "nonconformity")],
}


def expected(n_digits: int) -> pd.Series:
    """P(leading digits = d) = log10(1 + 1/d)."""
    digits = np.arange(10 ** (n_digits - 1), 10 ** n_digits)
    return pd.Series(np.log10(1 + 1 / digits), index=digits)


def leading_digits(amounts: pd.Series, n_digits: int) -> pd.Series:
    """First n significant digits of positive amounts >= 10 (scale-free)."""
    x = amounts[amounts >= 10].to_numpy(dtype=float)
    # Scientific notation ("3.4500...e+04") gives the significant digits without the
    # floating-point pitfalls of log10 near powers of ten.
    return pd.Series([int(f"{v:.12e}".replace(".", "")[:n_digits]) for v in x])


def nigrini_label(mad: float, n_digits: int) -> str:
    return next(label for limit, label in MAD_THRESHOLDS[n_digits] if mad <= limit)


def test(amounts: pd.Series, n_digits: int) -> tuple[dict, pd.DataFrame]:
    digits = leading_digits(amounts, n_digits)
    exp = expected(n_digits)
    counts = digits.value_counts().reindex(exp.index, fill_value=0)
    observed = counts / counts.sum()
    chi2, p = stats.chisquare(counts, exp * counts.sum())
    mad = float(np.mean(np.abs(observed - exp)))
    # z-statistic per digit, with Nigrini's continuity correction
    n = counts.sum()
    z = (np.abs(observed - exp) - 1 / (2 * n)) / np.sqrt(exp * (1 - exp) / n)
    table = pd.DataFrame({"count": counts, "observed": observed, "expected": exp,
                          "z": z.clip(lower=0)})
    summary = {"n": int(n), "chi2": float(chi2), "chi2_pvalue": float(p), "mad": mad,
               "conformity": nigrini_label(mad, n_digits)}
    return summary, table

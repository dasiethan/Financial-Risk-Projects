"""Bunching of contract amounts just below regulatory thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Thresholds in EUR excluding VAT (the DECP "montant" is HT), applicable in 2025
THRESHOLDS = {
    40_000: "no advertising / competition required below; DECP publication required above",
    90_000: "adapted procedures: formal notice (BOAMP or legal gazette) required above",
    143_000: "EU threshold, central government supplies & services (2024-2025)",
    221_000: "EU threshold, local authorities supplies & services (2024-2025)",
}
BIN_WIDTH = 0.01            # bins of 1% of the threshold
FIT_RANGE = (0.80, 1.20)    # window used to estimate the counterfactual density
EXCLUDED = (0.95, 1.05)     # bins around the threshold left out of the fit


def binned(amounts: pd.Series, threshold: float) -> pd.DataFrame:
    edges = np.round(np.arange(FIT_RANGE[0], FIT_RANGE[1] + BIN_WIDTH / 2, BIN_WIDTH), 4)
    ratio = amounts / threshold
    counts, _ = np.histogram(ratio, bins=edges)
    return pd.DataFrame({"left": edges[:-1], "count": counts})


def counterfactual(bins: pd.DataFrame) -> np.ndarray:
    """Log-linear density fitted on bins away from the threshold (a simple bunching
    estimator): what the counts would look like without any threshold effect."""
    outside = (bins["left"] < EXCLUDED[0] - 1e-9) | (bins["left"] >= EXCLUDED[1] - 1e-9)
    fit = bins[outside & (bins["count"] > 0)]
    slope, intercept = np.polyfit(fit["left"], np.log(fit["count"]), deg=1)
    return np.exp(intercept + slope * bins["left"].to_numpy())


def bunching(amounts: pd.Series, threshold: float) -> tuple[dict, pd.DataFrame]:
    bins = binned(amounts, threshold)
    bins["expected"] = counterfactual(bins)
    below = (bins["left"] >= EXCLUDED[0] - 1e-9) & (bins["left"] < 1 - 1e-9)
    above = (bins["left"] >= 1 - 1e-9) & (bins["left"] < EXCLUDED[1] - 1e-9)
    just_below = ((amounts >= threshold * 0.99) & (amounts < threshold)).sum()
    just_above = ((amounts >= threshold) & (amounts < threshold * 1.01)).sum()
    observed, expected = bins.loc[below, "count"].sum(), bins.loc[below, "expected"].sum()
    summary = {
        "observed_5pct_below": int(observed),
        "expected_5pct_below": float(expected),
        "excess_ratio_below": float(observed / expected),
        # One-sided Poisson test of excess mass below the threshold. Approximate: it
        # ignores the estimation error of the counterfactual itself.
        "excess_below_pvalue": float(stats.poisson.sf(observed - 1, expected)),
        "observed_5pct_above": int(bins.loc[above, "count"].sum()),
        "expected_5pct_above": float(bins.loc[above, "expected"].sum()),
        "count_1pct_below": int(just_below),
        "count_1pct_above": int(just_above),
        # Just below vs just above: with a smooth density the two should be close
        "binomial_pvalue_1pct_below_vs_above": float(stats.binomtest(
            int(just_below), int(just_below + just_above), 0.5,
            alternative="greater").pvalue) if just_below + just_above else float("nan"),
    }
    return summary, bins

"""ESG vs conventional US equity ETFs: CAPM, Fama-French 3 and 5 factors + momentum.

Run from this folder:  python main.py
Outputs: results/ (metrics.json, regression tables) and figures/ (PNG).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import plots
from src.data import LAST_MONTH, PAIRS, etf_monthly_returns, french_factors
from src.regression import FACTOR_MODELS, fit_factor_model, performance, summarize

ROOT = Path(__file__).resolve().parent
DATA_DIR, RESULTS_DIR, FIGURES_DIR = ROOT / "data", ROOT / "results", ROOT / "figures"


def analyse_pair(esg: str, conv: str, returns: pd.DataFrame, factors: pd.DataFrame):
    """Regress the ESG fund, its conventional twin (same months) and the long-short
    difference on each factor model. The difference is self-financing, so it is not
    reduced by the risk-free rate."""
    sample = returns[[esg, conv]].dropna().loc[:LAST_MONTH]
    rf = factors.loc[sample.index, "RF"]
    series = {"ESG": sample[esg] - rf, "conventional": sample[conv] - rf,
              "ESG minus conventional": sample[esg] - sample[conv]}
    rows = []
    for kind, y in series.items():
        fund = {"ESG": esg, "conventional": conv}.get(kind, f"{esg}-{conv}")
        for model in FACTOR_MODELS:
            row = summarize(fit_factor_model(y, factors, model), fund, kind, model)
            rows.append({"pair": f"{esg} vs {conv}", **row})
    diff = sample[esg] - sample[conv]
    stats = {
        "period": [str(sample.index[0]), str(sample.index[-1])], "n_months": len(sample),
        esg: performance(sample[esg], factors["RF"]),
        conv: performance(sample[conv], factors["RF"]),
        "tracking_error_annual": float(diff.std() * 12 ** 0.5),
        "correlation": float(sample[esg].corr(sample[conv])),
    }
    return rows, stats


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    factors = french_factors(DATA_DIR)
    returns = etf_monthly_returns(DATA_DIR)

    rows, pair_stats = [], {}
    for esg, conv in PAIRS.items():
        pair_rows, pair_stats[f"{esg} vs {conv}"] = analyse_pair(esg, conv, returns, factors)
        rows += pair_rows
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "factor_regressions.csv", index=False, float_format="%.5f")

    full = table[table["model"] == "FF5+Mom"]
    key_cols = ["pair", "kind", "fund", "n_months", "alpha_annual", "alpha_t", "alpha_p",
                "min_detectable_alpha_annual", "adj_r2"] + \
               [f"beta_{f}" for f in FACTOR_MODELS["FF5+Mom"]] + \
               [f"t_{f}" for f in FACTOR_MODELS["FF5+Mom"]]
    metrics = {
        "last_month": LAST_MONTH,
        "pairs": pair_stats,
        "ff5_mom": full[key_cols].to_dict(orient="records"),
        "capm_alpha": table.loc[table["model"] == "CAPM",
                                ["fund", "kind", "alpha_annual", "alpha_t", "beta_Mkt-RF"]]
                           .to_dict(orient="records"),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    plots.cumulative_growth(returns.loc[:LAST_MONTH], PAIRS, FIGURES_DIR / "cumulative_growth.png")
    plots.alphas(full, FIGURES_DIR / "alphas_ff5_mom.png")
    plots.factor_tilts(full[full["kind"] == "ESG minus conventional"],
                       FACTOR_MODELS["FF5+Mom"], FIGURES_DIR / "factor_tilts.png")

    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()

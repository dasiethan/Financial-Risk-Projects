"""Market risk of an equally weighted portfolio of five French large caps:
VaR / Expected Shortfall (historical, parametric, Monte Carlo, GARCH) and backtesting.

Run from this folder:  python main.py
Outputs: results/ (metrics.json, CSV tables) and figures/ (PNG).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import plots
from src import risk_models as rm
from src.backtest import backtest_summary, rolling_exception_counts, rolling_var
from src.data import TICKERS, build_returns, download_prices

SEED = 42
ALPHA_VAR = 0.99       # Basel backtesting level
ALPHA_ES = 0.975       # FRTB Expected Shortfall level
WINDOW = 500           # about two years of daily data
REFIT_EVERY = 20       # GARCH re-estimated roughly monthly
N_SIMS = 100_000
NOTIONAL_EUR = 1_000_000
TRADING_DAYS = 252
METHODS = ["Historical", "Normal", "Student-t", "GARCH-t", "FHS (GARCH)"]

ROOT = Path(__file__).resolve().parent
DATA_DIR, RESULTS_DIR, FIGURES_DIR = ROOT / "data", ROOT / "results", ROOT / "figures"


def describe(r: pd.Series) -> dict:
    jb_stat, jb_p = stats.jarque_bera(r)
    return {
        "start": str(r.index[0].date()), "end": str(r.index[-1].date()), "n_days": len(r),
        "annualised_return": float(r.mean() * TRADING_DAYS),
        "annualised_volatility": float(r.std() * np.sqrt(TRADING_DAYS)),
        "skewness": float(stats.skew(r)), "excess_kurtosis": float(stats.kurtosis(r)),
        "jarque_bera_pvalue": float(jb_p),
        "worst_day": str(r.idxmin().date()), "worst_day_return": float(r.min()),
    }


def snapshot_table(asset_returns: pd.DataFrame, r: pd.Series, weights: np.ndarray) -> pd.DataFrame:
    """VaR 99% and ES 97.5% for the day after the last observation, using the last WINDOW days."""
    rng = np.random.default_rng(SEED)
    sample, assets = r.iloc[-WINDOW:], asset_returns.iloc[-WINDOW:].to_numpy()
    nu = stats.t.fit(sample)[0]
    garch = rm.fit_garch(sample)
    estimators = {
        "Historical": lambda a: rm.historical(sample.to_numpy(), a),
        "Normal": lambda a: rm.normal(sample.to_numpy(), a),
        "Student-t": lambda a: rm.student_t(sample.to_numpy(), a),
        "Monte Carlo (multivariate t)": lambda a: rm.monte_carlo_t(
            assets, weights, a, nu, N_SIMS, rng),
        "GARCH-t": lambda a: rm.garch_t(garch, a),
        "FHS (GARCH)": lambda a: rm.filtered_historical(garch, a),
    }
    rows = {name: {"VaR 99%": f(ALPHA_VAR).var, "ES 97.5%": f(ALPHA_ES).es}
            for name, f in estimators.items()}
    return pd.DataFrame(rows).T


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    weights = np.full(len(TICKERS), 1 / len(TICKERS))
    prices = download_prices(DATA_DIR)
    asset_returns, r = build_returns(prices, weights)

    snapshot = snapshot_table(asset_returns, r, weights)
    var = rolling_var(r, WINDOW, ALPHA_VAR, REFIT_EVERY)
    losses = -r.loc[var.index]
    summary = backtest_summary(losses, var[METHODS], ALPHA_VAR)
    counts = rolling_exception_counts(var[METHODS].lt(losses, axis=0))

    metrics = {
        "portfolio": {"tickers": TICKERS, "weights": "equal, rebalanced daily",
                      **describe(r)},
        "settings": {"alpha_var": ALPHA_VAR, "alpha_es": ALPHA_ES, "window_days": WINDOW,
                     "garch_refit_every_days": REFIT_EVERY, "mc_simulations": N_SIMS},
        "snapshot_as_of": str(r.index[-1].date()),
        "snapshot_pct": snapshot.to_dict(orient="index"),
        "snapshot_eur_on_1m": (snapshot * NOTIONAL_EUR).round(0).to_dict(orient="index"),
        "backtest_period": [str(var.index[0].date()), str(var.index[-1].date())],
        "backtest": summary.to_dict(orient="index"),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    snapshot.to_csv(RESULTS_DIR / "snapshot_var_es.csv", float_format="%.5f")
    summary.to_csv(RESULTS_DIR / "backtest_summary.csv", float_format="%.4f")
    var.to_csv(RESULTS_DIR / "rolling_var.csv", float_format="%.6f")

    realized = r.rolling(21).std() * np.sqrt(TRADING_DAYS)
    plots.returns_distribution(r, FIGURES_DIR / "returns_distribution.png")
    plots.garch_volatility(var["garch_sigma"] * np.sqrt(TRADING_DAYS), realized.loc[var.index],
                           FIGURES_DIR / "garch_volatility.png")
    plots.var_backtest(r.loc[var.index], var[["Historical", "GARCH-t"]],
                       ("2020-01-01", "2020-12-31"), FIGURES_DIR / "var_backtest.png")
    plots.traffic_light(counts, FIGURES_DIR / "traffic_light.png")
    plots.snapshot(snapshot, FIGURES_DIR / "snapshot_var_es.png")

    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()

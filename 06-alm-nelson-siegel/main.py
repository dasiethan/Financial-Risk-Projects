"""Nelson-Siegel fitting of the ECB euro area AAA yield curve and an ALM application.

Run from this folder:  python main.py
Outputs: results/ (metrics.json, CSV tables) and figures/ (PNG).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import alm, plots
from src.curves import fit_fixed_tau, fit_nelson_siegel, ns_loadings, ns_yield, svensson_yield
from src.data import MATURITIES, load_curves, month_ends

ROOT = Path(__file__).resolve().parent
DATA_DIR, RESULTS_DIR, FIGURES_DIR = ROOT / "data", ROOT / "results", ROOT / "figures"
SNAPSHOT_DATES = ["2008-06", "2012-07", "2016-07", "2023-10"]  # + the last month
LIABILITY_NOMINAL, LIABILITY_YEARS = 100e6, 30
COUPON = 0.03
SHOCK = 0.01  # 100 bp


def ecb_parameter_check(frame: pd.DataFrame, m: np.ndarray) -> dict:
    """Gap (bp) between ECB spot rates and the curve rebuilt from the ECB Svensson
    parameters: validates the formula and flags days where the two series disagree."""
    params = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
    gaps = pd.Series([np.max(np.abs(svensson_yield(m, *row[params].to_numpy())
                                    - row[list(MATURITIES)].to_numpy())) * 100
                      for _, row in frame.iterrows()], index=frame.index)
    return {"median_max_gap_bp": float(gaps.median()),
            "days_with_gap_above_0_1bp": int((gaps > 0.1).sum()),
            "days_checked": len(gaps),
            "largest_gap_bp": float(gaps.max()), "largest_gap_date": str(gaps.idxmax().date())}


def monthly_fits(monthly: pd.DataFrame, m: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, residuals = [], []
    for date, row in monthly.iterrows():
        y = row[list(MATURITIES)].to_numpy(dtype=float)
        fit = fit_nelson_siegel(m, y)
        rows.append({"date": date, "beta0": fit.betas[0], "beta1": fit.betas[1],
                     "beta2": fit.betas[2], "tau": fit.tau, "rmse_bp": fit.rmse * 100})
        residuals.append((y - ns_yield(m, fit.betas, fit.tau)) * 100)
    fits = pd.DataFrame(rows).set_index("date")
    return fits, pd.DataFrame(residuals, index=fits.index, columns=list(MATURITIES))


def factor_analysis(monthly: pd.DataFrame, m: np.ndarray, tau: float):
    """Diebold-Li factors with a fixed tau, compared with empirical proxies."""
    betas = np.array([fit_fixed_tau(m, row[list(MATURITIES)].to_numpy(dtype=float), tau).betas
                      for _, row in monthly.iterrows()])
    factors_df = pd.DataFrame({"level": betas[:, 0], "slope": -betas[:, 1],
                               "curvature": betas[:, 2]}, index=monthly.index)
    proxies = pd.DataFrame({
        "level": monthly["SR_10Y"],
        "slope": monthly["SR_10Y"] - monthly["SR_3M"],
        "curvature": 2 * monthly["SR_2Y"] - monthly["SR_3M"] - monthly["SR_10Y"],
    })
    corr = {k: float(factors_df[k].corr(proxies[k])) for k in factors_df}
    return factors_df, proxies, corr


def alm_analysis(fit_row: pd.Series) -> tuple[dict, pd.DataFrame]:
    betas = fit_row[["beta0", "beta1", "beta2"]].to_numpy(dtype=float) / 100
    tau = float(fit_row["tau"])

    def curve(t):
        return ns_yield(t, betas, tau)

    liabilities = alm.liability_cash_flows(LIABILITY_NOMINAL, LIABILITY_YEARS)
    pv_l = alm.present_value(liabilities, curve)
    dur_l, conv_l = alm.effective_duration_convexity(liabilities, curve)
    short = alm.scale_to(alm.bond_cash_flows(1.0, COUPON, 5), pv_l, curve)
    matched = alm.duration_matched_mix(alm.bond_cash_flows(1.0, COUPON, 5),
                                       alm.bond_cash_flows(1.0, COUPON, 20), pv_l, dur_l, curve)
    assets = {"5Y bond only": short, "Duration-matched 5Y + 20Y": matched}

    def loading(k):  # shift of the curve when NS factor k moves by SHOCK
        return lambda t: SHOCK * ns_loadings(t, tau)[:, k]

    shocks = {
        "Parallel +100 bp": lambda t: np.full_like(t, SHOCK),
        "Parallel -100 bp": lambda t: np.full_like(t, -SHOCK),
        "Short end up (beta1 +100 bp): flattening": loading(1),
        "Short end down (beta1 -100 bp): steepening": lambda t: -loading(1)(t),
        "Hump up (beta2 +100 bp)": loading(2),
        "Hump down (beta2 -100 bp)": lambda t: -loading(2)(t),
    }
    impacts = alm.surplus_impacts(assets, liabilities, curve, shocks)

    profile = {"liabilities": {"pv_eur": pv_l, "duration": dur_l, "convexity": conv_l}}
    for name, flows in assets.items():
        d, c = alm.effective_duration_convexity(flows, curve)
        profile[name] = {"pv_eur": alm.present_value(flows, curve), "duration": d, "convexity": c}
    factor_durations = {}
    for name, flows in {"liabilities": liabilities, **assets}.items():
        pv = alm.present_value(flows, curve)
        factor_durations[name] = {
            f"beta{k}": -(alm.present_value(flows, alm.shifted(curve, loading(k))) - pv) / pv
            for k in range(3)}
    summary = {"curve_date": str(fit_row.name.date()), "ns_betas_pct": (betas * 100).tolist(),
               "tau": tau, "portfolios": profile,
               "factor_duration_pct_change_per_100bp": factor_durations,
               "surplus_impact_eur": impacts.to_dict(orient="index"),
               "surplus_impact_pct_of_liabilities": (impacts / pv_l).to_dict(orient="index")}
    return summary, impacts


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    frame = load_curves(DATA_DIR).dropna(subset=list(MATURITIES))
    m = np.array(list(MATURITIES.values()))
    monthly = month_ends(frame)

    fits, residuals = monthly_fits(monthly, m)
    tau_fixed = float(fits["tau"].median())
    factors_df, proxies, corr = factor_analysis(monthly, m, tau_fixed)
    alm_summary, impacts = alm_analysis(fits.iloc[-1])

    abs_err = residuals.abs()
    metrics = {
        "data": {"first_day": str(frame.index[0].date()), "last_day": str(frame.index[-1].date()),
                 "n_days": len(frame), "n_month_ends": len(monthly),
                 "maturities_years": m.tolist()},
        "ecb_svensson_reconstruction": ecb_parameter_check(frame, m),
        "nelson_siegel_fit": {
            "rmse_bp_mean": float(fits["rmse_bp"].mean()),
            "rmse_bp_median": float(fits["rmse_bp"].median()),
            "rmse_bp_max": float(fits["rmse_bp"].max()),
            "worst_month": str(fits["rmse_bp"].idxmax().date()),
            "share_months_rmse_below_5bp": float((fits["rmse_bp"] < 5).mean()),
            "mean_abs_error_bp_by_maturity": abs_err.mean().round(2).to_dict(),
            "tau_median_years": tau_fixed,
            "tau_min_max": [float(fits["tau"].min()), float(fits["tau"].max())],
            "share_months_tau_at_search_bound": float(
                ((fits["tau"] < 0.21) | (fits["tau"] > 9.99)).mean()),
        },
        "factors_fixed_tau": {
            "correlation_with_proxies": corr,
            "std_monthly_change_beta1_free_tau": float(fits["beta1"].diff().std()),
            "std_monthly_change_beta1_fixed_tau": float(factors_df["slope"].diff().std()),
            "level_min_max": [float(factors_df["level"].min()), float(factors_df["level"].max())],
        },
        "alm": alm_summary,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    fits.to_csv(RESULTS_DIR / "nelson_siegel_monthly_fits.csv", float_format="%.5f")
    factors_df.to_csv(RESULTS_DIR / "nelson_siegel_factors_fixed_tau.csv", float_format="%.5f")
    impacts.to_csv(RESULTS_DIR / "alm_surplus_impacts_eur.csv", float_format="%.0f")

    grid = np.linspace(0.25, 30, 200)
    snapshots = {}
    for label in SNAPSHOT_DATES + [str(fits.index[-1].to_period("M"))]:
        date = fits.index[fits.index.to_period("M") == pd.Period(label, "M")][0]
        row = fits.loc[date]
        betas = row[["beta0", "beta1", "beta2"]].to_numpy(dtype=float)
        snapshots[str(date.date())] = (m, monthly.loc[date, list(MATURITIES)].to_numpy(dtype=float),
                                       grid, ns_yield(grid, betas, row["tau"]), row["rmse_bp"])
    error_by_maturity = abs_err.mean().rename(lambda s: s.replace("SR_", ""))
    plots.curve_fits(snapshots, error_by_maturity,
                     FIGURES_DIR / "nelson_siegel_fits.png")
    plots.factors(factors_df, proxies, corr, FIGURES_DIR / "nelson_siegel_factors.png")
    plots.surplus_shocks(impacts, alm_summary["portfolios"]["liabilities"]["pv_eur"],
                         FIGURES_DIR / "alm_surplus_shocks.png")

    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()

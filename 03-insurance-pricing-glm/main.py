"""Motor insurance pricing on freMTPL2: frequency x severity GLMs vs LightGBM.

Run from this folder:  python main.py
Outputs: results/ (metrics.json, CSV tables, GLM summaries) and figures/ (PNG).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import evaluation as ev
from src import models, plots
from src.data import LARGE_CLAIM_CAP, glm_features, load_dataset

SEED = 42
TEST_SIZE = 0.20
ROOT = Path(__file__).resolve().parent
DATA_DIR, RESULTS_DIR, FIGURES_DIR = ROOT / "data", ROOT / "results", ROOT / "figures"


def claims_only(df: pd.DataFrame) -> pd.DataFrame:
    claims = df[df["ClaimNb"] > 0].copy()
    claims["AvgClaim"] = claims["ClaimAmountCapped"] / claims["ClaimNb"]
    return claims


def one_way_table(df: pd.DataFrame, band: pd.Series, rates: dict[str, np.ndarray]) -> pd.DataFrame:
    """Exposure-weighted observed and predicted frequency per band."""
    frame = pd.DataFrame({"band": band.to_numpy(), "exposure": df["Exposure"].to_numpy(),
                          "observed": df["ClaimNb"].to_numpy()})
    for name, rate in rates.items():
        frame[name] = rate * frame["exposure"]
    grouped = frame.groupby("band", sort=False, observed=True).sum()
    grouped = grouped.loc[sorted(grouped.index, key=_band_order)]
    for col in ["observed", *rates]:
        grouped[col] = grouped[col] / grouped["exposure"]
    return grouped


def _band_order(label: str) -> float:
    return float(label.split("-")[0].rstrip("+"))


def glm_relativities(model) -> pd.DataFrame:
    table = pd.DataFrame({"coef": model.params, "relativity": np.exp(model.params),
                          "p_value": model.pvalues})
    return table


def main() -> None:
    rng = np.random.default_rng(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    df, data_stats = load_dataset(DATA_DIR)
    train, test = train_test_split(glm_features(df), test_size=TEST_SIZE, random_state=SEED)
    claims_train, claims_test = claims_only(train), claims_only(test)

    # Frequency and severity models
    freq_glm, freq_factors, freq_dropped = models.fit_frequency_glm(train)
    sev_glm, sev_factors, sev_dropped = models.fit_severity_glm(claims_train)
    freq_gbm = models.fit_frequency_lgbm(train, SEED)
    sev_gbm = models.fit_severity_lgbm(claims_train, SEED)

    rate = {"GLM": lambda d: models.glm_frequency_rate(freq_glm, glm_features(d)),
            "LightGBM": lambda d: freq_gbm.predict(models.lgbm_frame(d))}
    severity = {"GLM": lambda d: models.glm_severity(sev_glm, glm_features(d)),
                "LightGBM": lambda d: sev_gbm.predict(models.lgbm_frame(d))}

    # Benchmarks: a single rate and a single average cost, estimated on train
    null_rate = train["ClaimNb"].sum() / train["Exposure"].sum()
    null_severity = claims_train["ClaimAmountCapped"].sum() / claims_train["ClaimNb"].sum()
    # Large-claim loading: the cost above the cap, spread proportionally over all premiums
    loading = (train["ClaimAmount"].sum() - train["ClaimAmountCapped"].sum()) \
        / train["ClaimAmountCapped"].sum()

    freq_dev_null = ev.poisson_deviance(test, np.full(len(test), null_rate))
    sev_dev_null = ev.gamma_deviance(claims_test, np.full(len(claims_test), null_severity))
    exposure = test["Exposure"].to_numpy()
    observed_capped = test["ClaimAmountCapped"].to_numpy()

    results, deciles, lorenz = {}, {}, {}
    test_rates = {}
    for name in ["GLM", "LightGBM"]:
        # Balance correction: rescale so that total predicted = total observed on train
        train_cost = (rate[name](train) * severity[name](train) * train["Exposure"]).sum()
        balance = train["ClaimAmountCapped"].sum() / train_cost
        test_rates[name] = rate[name](test)
        pure_premium = balance * test_rates[name] * severity[name](test)  # € per policy-year
        freq_dev = ev.poisson_deviance(test, test_rates[name])
        sev_dev = ev.gamma_deviance(claims_test, severity[name](claims_test))
        gini = ev.lorenz_gini(observed_capped, exposure, pure_premium)
        x, y = ev.lorenz_curve(observed_capped, exposure, pure_premium)
        lorenz[name] = (x, y, gini)
        deciles[name] = ev.premium_deciles(test, pure_premium)
        results[name] = {
            "frequency_poisson_deviance": freq_dev,
            "frequency_deviance_explained": 1 - freq_dev / freq_dev_null,
            "severity_gamma_deviance": sev_dev,
            "severity_deviance_explained": 1 - sev_dev / sev_dev_null,
            "pure_premium_lorenz_gini": gini,
            "balance_factor_from_train": float(balance),
            "predicted_over_observed_capped_cost": float(
                (pure_premium * exposure).sum() / observed_capped.sum()),
            "predicted_loaded_over_observed_total_cost": float(
                (pure_premium * (1 + loading) * exposure).sum() / test["ClaimAmount"].sum()),
            "mean_pure_premium_loaded_eur": float(
                np.average(pure_premium * (1 + loading), weights=exposure)),
            "top_decile_over_bottom_decile": float(
                deciles[name]["predicted_pure_premium"].iloc[-1]
                / deciles[name]["predicted_pure_premium"].iloc[0]),
        }

    importance = pd.DataFrame({
        name: ev.permutation_importance(test, rate[name], models.RAW_FEATURES, rng)
        for name in ["GLM", "LightGBM"]})

    bm_band = pd.cut(test["BonusMalus"], [0, 50, 60, 80, 100, np.inf],
                     labels=["50", "51-60", "61-80", "81-100", "101+"]).astype(str)
    one_ways = {"driver age": one_way_table(test, test["DrivAgeGrp"], test_rates),
                "bonus-malus level": one_way_table(test, bm_band, test_rates)}

    metrics = {
        "data": {**data_stats, "large_claim_cap_eur": LARGE_CLAIM_CAP,
                 "n_train": len(train), "n_test": len(test),
                 "claims_train": int(claims_train["ClaimNb"].sum()),
                 "claims_test": int(claims_test["ClaimNb"].sum()),
                 "observed_frequency_train": float(null_rate),
                 "observed_frequency_test": float(
                     test["ClaimNb"].sum() / test["Exposure"].sum()),
                 "observed_mean_capped_severity_train": float(null_severity),
                 "observed_mean_capped_severity_test": float(
                     claims_test["ClaimAmountCapped"].sum() / claims_test["ClaimNb"].sum()),
                 "large_claim_loading": float(loading)},
        "glm_selection": {"frequency_kept": freq_factors, "frequency_dropped": freq_dropped,
                          "severity_kept": sev_factors, "severity_dropped": sev_dropped},
        "benchmark_constant": {"frequency_poisson_deviance": freq_dev_null,
                               "severity_gamma_deviance": sev_dev_null},
        "models": results,
        "lightgbm_trees": {"frequency": int(freq_gbm.n_estimators),
                           "severity": int(sev_gbm.n_estimators)},
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (RESULTS_DIR / "glm_frequency_summary.txt").write_text(str(freq_glm.summary()))
    (RESULTS_DIR / "glm_severity_summary.txt").write_text(str(sev_glm.summary()))
    glm_relativities(freq_glm).to_csv(RESULTS_DIR / "glm_frequency_relativities.csv",
                                      float_format="%.4f")
    glm_relativities(sev_glm).to_csv(RESULTS_DIR / "glm_severity_relativities.csv",
                                     float_format="%.4f")
    importance.to_csv(RESULTS_DIR / "permutation_importance.csv", float_format="%.6f")
    pd.concat(deciles, names=["model"]).to_csv(RESULTS_DIR / "premium_deciles.csv",
                                               float_format="%.2f")
    pd.concat(one_ways, names=["factor"]).to_csv(RESULTS_DIR / "one_way_frequency.csv",
                                                 float_format="%.4f")

    sev_records = df.loc[df["ClaimNb"] > 0, "ClaimAmount"] / df.loc[df["ClaimNb"] > 0, "ClaimNb"]
    plots.severity_distribution(sev_records, LARGE_CLAIM_CAP,
                                FIGURES_DIR / "severity_distribution.png")
    plots.one_way(one_ways, FIGURES_DIR / "frequency_one_way.png")
    plots.lorenz(lorenz, FIGURES_DIR / "lorenz_curves.png")
    plots.premium_deciles(deciles, FIGURES_DIR / "premium_deciles.png")
    plots.importance(importance, FIGURES_DIR / "permutation_importance.png")

    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()

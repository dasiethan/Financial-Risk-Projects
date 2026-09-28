"""End-to-end credit scoring pipeline: WoE/IV scorecard vs LightGBM.

Run from this folder:  python main.py
Outputs: results/ (metrics.json, CSV tables) and figures/ (PNG).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import plots
from src.boosting import fit_lgbm
from src.data import CATEGORICAL, TARGET, load_dataset
from src.evaluation import (characteristic_stability, decile_table,
                            discrimination_metrics, score_psi)
from src.scorecard import (BASE_ODDS, BASE_SCORE, PDO, fit_with_sign_and_pvalue_checks,
                           predict_pd, score_points, scorecard_table,
                           select_by_iv_and_correlation)
from src.woe import WoEEncoder

SEED = 42
TEST_SIZE = 0.30
MIN_IV = 0.02        # below 0.02 a variable is usually considered not predictive
MAX_CORR = 0.70      # WoE correlation above which two variables are redundant
MAX_PVALUE = 0.05

ROOT = Path(__file__).resolve().parent
DATA_DIR, RESULTS_DIR, FIGURES_DIR = ROOT / "data", ROOT / "results", ROOT / "figures"


def build_scorecard(X_train, y_train):
    encoder = WoEEncoder(CATEGORICAL).fit(X_train, y_train)
    iv = encoder.information_values()
    woe_train = encoder.transform(X_train)
    candidates = select_by_iv_and_correlation(iv, woe_train, MIN_IV, MAX_CORR)
    logit, kept = fit_with_sign_and_pvalue_checks(woe_train[candidates], y_train, MAX_PVALUE)
    return encoder, iv, candidates, logit, kept


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_dataset(DATA_DIR)
    features = [c for c in df.columns if c != TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[TARGET], test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED)

    # 1. Scorecard: WoE binning -> IV/correlation selection -> logistic regression
    encoder, iv, candidates, logit, kept = build_scorecard(X_train, y_train)
    woe_train, woe_test = encoder.transform(X_train, kept), encoder.transform(X_test, kept)
    sc_pd_train, sc_pd_test = predict_pd(logit, woe_train), predict_pd(logit, woe_test)
    sc_points_train, sc_points_test = score_points(logit, woe_train), score_points(logit, woe_test)

    # 2. Challenger: LightGBM on the raw (non-binned) variables
    lgbm = fit_lgbm(X_train, y_train, SEED)
    gb_pd_train = lgbm.predict_proba(X_train)[:, 1]
    gb_pd_test = lgbm.predict_proba(X_test)[:, 1]

    # 3. Stability: train vs test, plus an illustrative shifted sub-population
    young_test = (X_test["AGE"] < 30).to_numpy()
    csi = characteristic_stability(encoder, X_train, X_test, kept)
    stability = {
        "score_psi_train_vs_test_scorecard": score_psi(sc_points_train, sc_points_test),
        "score_psi_train_vs_test_lgbm": score_psi(gb_pd_train, gb_pd_test),
        "max_variable_psi_train_vs_test": float(csi.max()),
        "illustrative_score_psi_train_vs_test_age_under_30": score_psi(
            sc_points_train, sc_points_test[young_test]),
        "illustrative_share_age_under_30_in_test": float(young_test.mean()),
    }

    y_tr, y_te = y_train.to_numpy(), y_test.to_numpy()
    deciles = {"Scorecard (logistic, WoE)": decile_table(y_te, sc_pd_test),
               "LightGBM": decile_table(y_te, gb_pd_test)}
    metrics = {
        "dataset": {"n_rows": len(df), "n_train": len(X_train), "n_test": len(X_test),
                    "default_rate": float(df[TARGET].mean()),
                    "n_candidate_variables": len(features)},
        "scorecard": {
            "scaling": {"base_score": BASE_SCORE, "base_odds_good_bad": BASE_ODDS, "pdo": PDO},
            "n_variables_iv_above_threshold": int((iv >= MIN_IV).sum()),
            "n_variables_after_correlation_filter": len(candidates),
            "final_variables": kept,
            "train": discrimination_metrics(y_tr, sc_pd_train),
            "test": discrimination_metrics(y_te, sc_pd_test),
        },
        "lightgbm": {"n_trees": int(lgbm.n_estimators),
                     "train": discrimination_metrics(y_tr, gb_pd_train),
                     "test": discrimination_metrics(y_te, gb_pd_test)},
        "stability": stability,
    }

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (RESULTS_DIR / "logit_summary.txt").write_text(str(logit.summary()))
    iv.to_frame().assign(kept=lambda t: t.index.isin(kept)).to_csv(
        RESULTS_DIR / "information_value.csv", float_format="%.4f")
    scorecard_table(logit, encoder, kept).to_csv(
        RESULTS_DIR / "scorecard.csv", index=False, float_format="%.4f")
    csi.to_csv(RESULTS_DIR / "variable_psi.csv", float_format="%.4f")
    pd.concat(deciles, names=["model"]).to_csv(
        RESULTS_DIR / "calibration_deciles.csv", float_format="%.4f")

    importance = pd.Series(lgbm.booster_.feature_importance("gain"), index=features)
    plots.iv_ranking(iv, kept, MIN_IV, FIGURES_DIR / "iv_ranking.png")
    plots.roc_curves(y_te, {"Scorecard (logistic, WoE)": sc_pd_test, "LightGBM": gb_pd_test},
                     FIGURES_DIR / "roc_curves.png")
    plots.ks_curve(y_te, sc_points_test, FIGURES_DIR / "ks_curve.png")
    plots.calibration(deciles, FIGURES_DIR / "calibration.png")
    plots.score_distribution(y_te, sc_points_test, FIGURES_DIR / "score_distribution.png")
    plots.feature_importance(importance, FIGURES_DIR / "lgbm_feature_importance.png")

    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()

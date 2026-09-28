"""Discrimination, calibration and stability metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

PSI_FLOOR = 1e-6  # avoids log(0) when a bin is empty in one sample


def ks_statistic(y: np.ndarray, pd_hat: np.ndarray) -> float:
    """Maximum distance between the cumulative distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, pd_hat)
    return float(np.max(tpr - fpr))


def discrimination_metrics(y: np.ndarray, pd_hat: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y, pd_hat)
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, pd_hat),
        "brier": brier_score_loss(y, pd_hat),
        "log_loss": log_loss(y, pd_hat),
        "mean_predicted_pd": float(np.mean(pd_hat)),
        "observed_default_rate": float(np.mean(y)),
    }


def decile_table(y: np.ndarray, pd_hat: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted default rate by decile of predicted PD (1 = safest)."""
    frame = pd.DataFrame({"y": y, "pd": pd_hat})
    frame["decile"] = pd.qcut(frame["pd"].rank(method="first"), n_bins, labels=False) + 1
    return frame.groupby("decile").agg(n=("y", "size"), mean_predicted_pd=("pd", "mean"),
                                       observed_default_rate=("y", "mean"))


def psi_from_shares(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, PSI_FLOOR, None)
    a = np.clip(actual, PSI_FLOOR, None)
    return float(np.sum((a - e) * np.log(a / e)))


def score_psi(expected_scores: np.ndarray, actual_scores: np.ndarray, n_bins: int = 10) -> float:
    """PSI with bins set on the quantiles of the reference (training) scores."""
    edges = np.unique(np.quantile(expected_scores, np.linspace(0, 1, n_bins + 1)[1:-1]))
    shares = [np.bincount(np.searchsorted(edges, s, side="right"),
                          minlength=len(edges) + 1) / len(s)
              for s in (expected_scores, actual_scores)]
    return psi_from_shares(*shares)


def characteristic_stability(encoder, X_ref: pd.DataFrame, X_new: pd.DataFrame,
                             features: list[str]) -> pd.Series:
    """PSI of each variable computed on its own WoE bins (a.k.a. CSI)."""
    values = {}
    for col in features:
        ref = encoder.bin_keys(col, X_ref[col]).value_counts(normalize=True)
        new = encoder.bin_keys(col, X_new[col]).value_counts(normalize=True)
        ref, new = ref.align(new, fill_value=0.0)
        values[col] = psi_from_shares(ref.to_numpy(), new.to_numpy())
    return pd.Series(values, name="psi").sort_values(ascending=False)

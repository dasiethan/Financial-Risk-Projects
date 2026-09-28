"""Audit analytics on French public procurement contracts notified in 2025 (DECP):
Benford's law, round numbers, bunching below thresholds, rule-based controls and
Isolation Forest.

Run from this folder:  python main.py
Outputs: results/ (metrics.json, CSV tables) and figures/ (PNG).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import benford, plots
from src.anomalies import isolation_scores, rule_flags
from src.data import PUBLISHED_BY, YEAR, load_contracts
from src.thresholds import THRESHOLDS, bunching

SEED = 42
CONTAMINATION = 0.01     # share of contracts ranked as anomalies by the Isolation Forest
MIN_GROUP = 1_000        # minimum contracts per procedure for a separate Benford test
ROOT = Path(__file__).resolve().parent
DATA_DIR, RESULTS_DIR, FIGURES_DIR = ROOT / "data", ROOT / "results", ROOT / "figures"


def round_number_stats(amounts: pd.Series) -> dict:
    return {
        "share_multiple_of_1000": float((amounts % 1_000 == 0).mean()),
        "share_multiple_of_10000": float((amounts % 10_000 == 0).mean()),
        "top_exact_amounts": {f"{k:,.0f}": int(v)
                              for k, v in amounts.value_counts().head(10).items()},
    }


def benford_section(df: pd.DataFrame) -> tuple[dict, dict]:
    amounts = df["montant"]
    not_round = amounts[amounts % 1_000 != 0]
    first_all, table_all = benford.test(amounts, 1)
    first_nr, table_nr = benford.test(not_round, 1)
    two_all, table_two = benford.test(amounts, 2)
    by_procedure = {}
    for procedure, group in df.groupby("procedure"):
        if len(group) >= MIN_GROUP:
            by_procedure[procedure] = benford.test(group["montant"], 1)[0]
    results = {"first_digit_all": first_all, "first_digit_excluding_multiples_of_1000": first_nr,
               "first_two_digits_all": two_all, "first_digit_by_procedure": by_procedure,
               "largest_two_digit_deviations": table_two["z"].nlargest(8).round(1).to_dict()}
    tables = {"first_all": table_all, "first_not_round": table_nr, "two_all": table_two}
    return results, tables


def anomaly_section(df: pd.DataFrame, flags: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    scores = isolation_scores(df, SEED, CONTAMINATION)
    top = scores >= scores.quantile(1 - CONTAMINATION)
    profile = {
        "n_ranked_anomalies": int(top.sum()),
        "share_also_flagged_by_a_rule": float(flags[top].any(axis=1).mean()),
        "share_flagged_by_a_rule_overall": float(flags.any(axis=1).mean()),
        "median_amount_anomalies": float(df.loc[top, "montant"].median()),
        "median_amount_all": float(df["montant"].median()),
        "share_amount_below_10_or_100m_plus": float(
            ((df.loc[top, "montant"] < 10) | (df.loc[top, "montant"] >= 1e8)).mean()),
        "share_offers_missing": float(df.loc[top, "offers"].isna().mean()),
        "share_offers_20_plus": float((df.loc[top, "offers"] >= 20).mean()),
        "top_procedures": df.loc[top, "procedure"].value_counts().head(3).to_dict(),
    }
    columns = ["acheteur_id", "id", "procedure", "codecpv", "montant", "dureemois", "offers",
               "datenotification", "objet"]
    listing = df.loc[top, columns].assign(anomaly_score=scores[top],
                                          objet=df.loc[top, "objet"].str[:120])
    return profile, listing.sort_values("anomaly_score", ascending=False)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    df, info = load_contracts(DATA_DIR)

    benford_results, tables = benford_section(df)
    # Round amounts (e.g. a 220,000 EUR framework maximum) create spikes of their own,
    # so bunching is also measured on amounts that are not multiples of 1,000 EUR.
    not_round = df.loc[df["montant"] % 1_000 != 0, "montant"]
    bunching_results, bunching_bins = {}, {}
    for threshold, meaning in THRESHOLDS.items():
        summary, _ = bunching(df["montant"], threshold)
        summary_nr, bins = bunching(not_round, threshold)
        bunching_results[f"{threshold:,}"] = {
            "meaning": meaning, "all_amounts": summary,
            "excluding_multiples_of_1000": summary_nr,
            "exact_threshold_minus_1000": int((df["montant"] == threshold - 1_000).sum()),
            "exact_threshold": int((df["montant"] == threshold).sum()),
        }
        bunching_bins[threshold] = bins

    flags = rule_flags(df)
    flag_counts = flags.sum().astype(int)
    anomaly_profile, anomaly_list = anomaly_section(df, flags)

    metrics = {
        "data": {"notification_year": YEAR, "published_by": PUBLISHED_BY, **info},
        "round_numbers": round_number_stats(df["montant"]),
        "benford": benford_results,
        "threshold_bunching": bunching_results,
        "rule_flags": flag_counts.to_dict(),
        "rule_flags_any": int(flags.any(axis=1).sum()),
        "isolation_forest": anomaly_profile,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float,
                                                         ensure_ascii=False), encoding="utf-8")
    for name, table in tables.items():
        table.to_csv(RESULTS_DIR / f"benford_{name}.csv", float_format="%.5f")
    anomaly_list.head(200).to_csv(RESULTS_DIR / "top_anomalies.csv", index=False,
                                  float_format="%.4f", encoding="utf-8")

    plots.benford_first_digit(
        {"all contracts": tables["first_all"],
         "excluding multiples of 1,000 €": tables["first_not_round"]},
        {"all contracts": benford_results["first_digit_all"]["mad"],
         "excluding multiples of 1,000 €":
             benford_results["first_digit_excluding_multiples_of_1000"]["mad"]},
        FIGURES_DIR / "benford_first_digit.png")
    plots.benford_two_digits(tables["two_all"], tables["two_all"]["z"].nlargest(8),
                             FIGURES_DIR / "benford_first_two_digits.png")
    plots.thresholds(bunching_bins, FIGURES_DIR / "threshold_bunching.png")
    plots.flags(flag_counts, len(df), FIGURES_DIR / "audit_flags.png")

    print(json.dumps(metrics, indent=2, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()

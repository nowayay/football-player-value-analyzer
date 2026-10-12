"""Football Player Value Analyzer - Streamlit app.

Runs from committed files only: data/processed/player_values.csv, data/processed/player_stats.csv,
reports/*.csv and reports/figures/*.png. Predictions are precomputed by `python -m src.predict`,
so the app needs no raw data, no training and no model files.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `streamlit run app/...` puts app/ on the path, not the repo root

from src.plot_style import SERIES, TEXT_MUTED, apply_style  # noqa: E402

VALUES_PATH = ROOT / "data" / "processed" / "player_values.csv"
STATS_PATH = ROOT / "data" / "processed" / "player_stats.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

DEFAULT_AGE = (21, 32)
DEFAULT_MIN_VALUE_M = 1.0
DEFAULT_MIN_MINUTES = 900
VETERAN_AGE = 33
POSITIONS = ["GK", "DEF", "MID", "ATT"]
FILTER_NOTE = ("Veterans (33+) and very low values are hidden by default: the model doesn't know contract "
               "length or retirement horizon. Widen the filters to see everyone.")
STAT_LABELS = {
    "minutes": "League minutes", "appearances": "League appearances", "minutes_share": "Share of available minutes",
    "goals_p90": "Goals per 90", "assists_p90": "Assists per 90", "ga_p90": "Goals + assists per 90",
    "yellow_p90": "Yellow cards per 90", "prev_minutes": "Previous-season minutes",
    "prev_ga_p90": "Previous-season G+A per 90", "form_ga_p90": "2-season form, G+A per 90",
    "club_ppg": "Club points per game", "height_in_cm": "Height (cm)",
}


@st.cache_data
def load_values() -> pd.DataFrame:
    """Player value table from src.predict, with EUR columns also in millions for display."""
    values = pd.read_csv(VALUES_PATH)
    for col in ["market_value_eur", "predicted_value_eur", "centred_value_eur", "centred_low_eur", "centred_high_eur"]:
        values[col.replace("_eur", "_m")] = values[col] / 1e6
    return values


@st.cache_data
def load_stats() -> pd.DataFrame:
    """Key per-90 stats per player for the lookup tab."""
    return pd.read_csv(STATS_PATH)


@st.cache_data
def load_report(name: str, **kwargs) -> pd.DataFrame:
    """Read one of the CSV reports written by the pipeline."""
    return pd.read_csv(REPORTS / name, **kwargs)


def sidebar_filters(values: pd.DataFrame) -> dict:
    """Render the sidebar and return the chosen filter values."""
    st.sidebar.header("Filters")
    st.sidebar.caption("Apply to the Undervalued and Overvalued tabs.")
    positions = st.sidebar.multiselect("Position group", POSITIONS, default=POSITIONS)
    age_min, age_max = int(values["age"].min()), int(values["age"].max()) + 1
    age = st.sidebar.slider("Age", age_min, age_max, DEFAULT_AGE)
    clubs = st.sidebar.multiselect("Club (empty = all)", sorted(values["club"].unique()))
    top_value = float(values["market_value_m"].max())
    value = st.sidebar.slider("Market value (EUR m)", 0.0, top_value, (DEFAULT_MIN_VALUE_M, top_value), step=0.5)
    minutes = st.sidebar.slider("Minimum league minutes", 450, int(values["minutes"].max()), DEFAULT_MIN_MINUTES,
                                step=90, help="Players need 900+ minutes to receive an over/undervalued flag.")
    return {"positions": positions, "age": age, "clubs": clubs, "value": value, "minutes": minutes}


def apply_filters(values: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Rows matching the sidebar filters."""
    keep = (values["position"].isin(f["positions"]) & values["age"].between(*f["age"])
            & values["market_value_m"].between(*f["value"]) & (values["minutes"] >= f["minutes"]))
    if f["clubs"]:
        keep &= values["club"].isin(f["clubs"])
    return values[keep]


def ranking_table(df: pd.DataFrame, undervalued: bool, top_n: int) -> None:
    """Top-N players by adjusted gap (largest first for undervalued, smallest first for overvalued)."""
    ranked = df.sort_values("adjusted_gap_pct", ascending=not undervalued, kind="mergesort").head(top_n)
    ranked = ranked.assign(gap_m=ranked["centred_value_m"] - ranked["market_value_m"])  # centred estimate - market
    columns = ["player", "club", "position", "age", "market_value_m", "centred_value_m", "adjusted_gap_pct", "gap_m",
               "flag", "minutes"]
    if (ranked["age"] >= VETERAN_AGE).any():
        ranked = ranked.assign(caution=ranked["age"].ge(VETERAN_AGE).map({True: "⚠ age 33+", False: ""}))
        columns.append("caution")
    st.dataframe(
        ranked[columns], hide_index=True, width="stretch",
        column_config={
            "player": "Player", "club": "Club", "position": "Pos",
            "age": st.column_config.NumberColumn("Age", format="%.1f"),
            "market_value_m": st.column_config.NumberColumn("Market value (EUR m)", format="%.1f"),
            "centred_value_m": st.column_config.NumberColumn(
                "Predicted (EUR m)*", format="%.1f",
                help="Model estimate, centred on the season (see note below the table)"),
            "adjusted_gap_pct": st.column_config.NumberColumn("Adjusted gap", format="%+.0f%%"),
            "gap_m": st.column_config.NumberColumn("Gap (EUR m)", format="%+.1f",
                                                   help="Season-centred estimate minus market value"),
            "flag": "80% interval flag", "minutes": st.column_config.NumberColumn("Minutes", format="%d"),
            "caution": "Caution",
        },
    )
    st.caption("*Predicted values are shifted by the season's median gap, because the model under-predicts the "
               "whole 2025/26 season. Adjusted gap = predicted / market value - 1. A flag needs the market value "
               "outside the (shifted) 80% interval and 900+ minutes.")


def _format_stat(column: str, value: float) -> str:
    """Readable value for the lookup stats table."""
    if pd.isna(value):
        return "-"
    if column == "minutes_share":
        return f"{100 * value:.0f}%"
    if column in {"minutes", "appearances", "prev_minutes", "height_in_cm"}:
        return f"{value:.0f}"
    return f"{value:.2f}"


def value_chart(row: pd.Series):
    """Horizontal bars: market value vs season-centred estimate with its 80% interval."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 1.9))
    labels = ["Market value", "Model estimate"]
    ax.barh(labels, [row["market_value_m"], row["centred_value_m"]], color=[TEXT_MUTED, SERIES[0]], height=0.5)
    ax.errorbar(row["centred_value_m"], 1, xerr=[[row["centred_value_m"] - row["centred_low_m"]],
                                                [row["centred_high_m"] - row["centred_value_m"]]],
                fmt="none", ecolor=SERIES[1], elinewidth=2, capsize=6)
    ax.set(xlabel="EUR million", title="Market value vs model estimate (orange line: 80% interval)")
    ax.set_xlim(0, max(row["market_value_m"], row["centred_high_m"]) * 1.05)
    ax.invert_yaxis()
    return fig


def tab_ranking(filtered: pd.DataFrame, undervalued: bool) -> None:
    """Undervalued or overvalued tab."""
    word = "above" if undervalued else "below"
    st.info(FILTER_NOTE)
    top_n = st.slider("Number of players", 10, 100, 25, step=5, key=f"top_n_{undervalued}")
    st.write(f"{len(filtered):,} players match the filters. Sorted by how far the model's estimate is {word} "
             "the market value, relative to the rest of the season.")
    ranking_table(filtered, undervalued, top_n)


def tab_lookup(values: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Search one player: values, gaps, interval chart, key stats and cautions."""
    query = st.text_input("Search player by name", placeholder="e.g. Saka")
    if not query:
        st.caption("Type part of a name. All 2025/26 top-5 league players with 450+ league minutes are included, "
                   "whatever the sidebar filters say.")
        return
    matches = values[values["player"].str.contains(query, case=False, regex=False)]
    matches = matches.sort_values("market_value_eur", ascending=False)  # best-known players first
    if matches.empty:
        st.warning("No player found. Only players with 450+ top-5 league minutes in 2025/26 are included.")
        return
    options = {f"{r.player} ({r.club}, {r.position}, {r.age:.0f})": i for i, r in matches.iterrows()}
    row = values.loc[options[st.selectbox("Player", list(options))]]

    cols = st.columns(3)
    cols[0].metric("Market value", f"EUR {row['market_value_m']:.1f}m")
    cols[1].metric("Model estimate (centred)", f"EUR {row['centred_value_m']:.1f}m")
    cols[2].metric("Adjusted gap", f"{row['adjusted_gap_pct']:+.0f}%")
    st.write(f"**Flag:** {row['flag']}  ·  raw model estimate EUR {row['predicted_value_m']:.1f}m, "
             f"raw gap {row['raw_gap_pct']:+.0f}%  ·  {row['club']}, {row['league']}, {row['sub_position']}")
    if row["age"] >= VETERAN_AGE:
        st.warning("Age 33+: the model doesn't know contract length or retirement horizon, so it likely overestimates.")
    elif row["age"] <= 21:
        st.warning("Young players: potential isn't captured, so the model likely underestimates.")
    if row["flag"] == "not rated":
        st.caption("Under 900 league minutes: too little evidence for an over/undervalued flag.")

    fig = value_chart(row)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    player_stats = stats[stats["player_id"] == row["player_id"]]
    if not player_stats.empty:
        s = player_stats.iloc[0]
        table = pd.DataFrame({"Stat": list(STAT_LABELS.values()),
                              "Value": [_format_stat(c, s[c]) for c in STAT_LABELS]})
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Previous-season stats show '-' when the player has no domestic-league season in the dataset "
                   "the year before.")


def _pct_or_dash(share: float) -> str:
    """Format a share as a whole percentage; '-' for models without a prediction interval."""
    return "-" if pd.isna(share) else f"{100 * share:.0f}%"


def tab_performance() -> None:
    """Model metrics, plots, SHAP and the backtest with its caveats."""
    test = load_report("test_metrics.csv", index_col="model")
    cv = load_report("cv_summary.csv", index_col="model")
    main_bias = test.loc["xgb", "mean_bias"]

    st.subheader("Accuracy on the held-out 2025/26 season (evaluated once)")
    shown = test[["label", "rmse_log", "r2_log", "mae_eur", "median_ape", "within_25", "within_50", "mean_bias",
                  "interval_coverage_80"]].copy()
    for col in ["median_ape", "within_25", "within_50"]:
        shown[col] *= 100
    shown["interval_coverage_80"] = shown["interval_coverage_80"].map(_pct_or_dash)
    shown["mae_eur"] /= 1e6
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        "label": "Model", "rmse_log": st.column_config.NumberColumn("RMSE (log)", format="%.3f"),
        "r2_log": st.column_config.NumberColumn("R² (log)", format="%.2f"),
        "mae_eur": st.column_config.NumberColumn("MAE (EUR m)", format="%.1f"),
        "median_ape": st.column_config.NumberColumn("Median % error", format="%.0f%%"),
        "within_25": st.column_config.NumberColumn("Within 25%", format="%.0f%%"),
        "within_50": st.column_config.NumberColumn("Within 50%", format="%.0f%%"),
        "mean_bias": st.column_config.NumberColumn("Mean bias (log)", format="%+.3f"),
        "interval_coverage_80": st.column_config.TextColumn("80% coverage"),
    })
    st.caption(f"The main model (XGBoost) under-predicts the whole 2025/26 season (mean bias {main_bias:+.3f} log, "
               f"about {100 * (np.exp(main_bias) - 1):.0f}%): "
               "market prices rose faster than the lagged league market level. Gaps and flags in this app are "
               "centred on each season's median to remove that shared shift.")

    st.subheader("Cross-validation (validation seasons 2021/22-2024/25)")
    cv_shown = cv[["label", "rmse_log", "r2_log", "median_ape", "within_50", "mean_bias", "coverage_conformal"]].copy()
    for col in ["median_ape", "within_50"]:
        cv_shown[col] *= 100
    cv_shown["coverage_conformal"] = cv_shown["coverage_conformal"].map(_pct_or_dash)
    st.dataframe(cv_shown, hide_index=True, width="stretch", column_config={
        "label": "Model", "rmse_log": st.column_config.NumberColumn("RMSE (log)", format="%.3f"),
        "r2_log": st.column_config.NumberColumn("R² (log)", format="%.2f"),
        "median_ape": st.column_config.NumberColumn("Median % error", format="%.0f%%"),
        "within_50": st.column_config.NumberColumn("Within 50%", format="%.0f%%"),
        "mean_bias": st.column_config.NumberColumn("Mean bias (log)", format="%+.3f"),
        "coverage_conformal": st.column_config.TextColumn("80% coverage"),
    })

    st.subheader("Where the model is right and wrong")
    left, right = st.columns(2)
    left.image(str(FIGURES / "pred_vs_actual.png"), width="stretch")
    right.image(str(FIGURES / "residuals_vs_pred.png"), width="stretch")
    failures = load_report("test_failure_segments.csv", index_col=0)
    failures = failures.assign(median_ape=100 * failures["median_ape"])
    st.dataframe(failures[["n", "rmse_log", "median_ape", "mean_bias"]], width="stretch", column_config={
        "n": st.column_config.NumberColumn("Players", format="%d"),
        "rmse_log": st.column_config.NumberColumn("RMSE (log)", format="%.3f"),
        "median_ape": st.column_config.NumberColumn("Median % error", format="%.0f%%"),
        "mean_bias": st.column_config.NumberColumn("Mean bias (log)", format="%+.3f"),
    })
    st.caption("Young players (potential) and EUR 50m+ stars are under-predicted by about a third; "
               "players with few minutes are noisy.")

    st.subheader("What drives the model (SHAP)")
    left, right = st.columns(2)
    left.image(str(FIGURES / "shap_importance_grouped.png"), width="stretch")
    right.image(str(FIGURES / "shap_beeswarm.png"), width="stretch")
    st.write("SHAP values show how each feature moved this model's prediction relative to its average output; they "
             "describe the model, not football. They show association, not causation: a high value for club points "
             "per game does not mean joining a better club would raise a player's value by that amount. Club "
             "strength, age and the league market level dominate, so the estimate is mostly context-implied rather "
             "than performance-implied.")

    st.subheader("Backtest: do flagged players' values move afterwards?")
    st.image(str(FIGURES / "backtest_value_change.png"), width="stretch")
    robust = load_report("backtest_robustness.csv")
    st.dataframe(robust[["spec", "group", "n_flagged", "matched_excess", "matched_ci_low", "matched_ci_high",
                         "reg_coef", "reg_ci_low", "reg_ci_high"]], hide_index=True, width="stretch",
                 column_config={
                     "spec": "Specification", "group": "Flag", "n_flagged": "Flagged",
                     **{c: st.column_config.NumberColumn(c.replace("_", " "), format="%+.3f")
                        for c in ["matched_excess", "matched_ci_low", "matched_ci_high", "reg_coef", "reg_ci_low",
                                  "reg_ci_high"]},
                 })
    main_under = robust[(robust["spec"] == "main") & (robust["group"] == "undervalued")].iloc[0]
    bound_under = robust[(robust["spec"].str.startswith("(i)")) & (robust["group"] == "undervalued")].iloc[0]
    st.write(
        f"**Conclusion.** Over 2021/22-2024/25, players flagged as undervalued lost about "
        f"{100 * bound_under['matched_excess']:.0f}-{100 * main_under['matched_excess']:.0f} log points less market "
        "value over the next 12 months than comparable players (same season, position and age band). The effect "
        "holds without veterans and very cheap players, and under pessimistic bounds for missing outcomes; the "
        "overvalued effect is about half as large. This shows that Transfermarkt values tend to move toward the "
        "model's estimate (mean reversion in a noisy valuation process), not that flagged players are proven "
        "bargains in real transfer fees.")
    st.markdown(
        "- **Mean reversion:** the model learned Transfermarkt's own pricing, so a flag partly marks a noisy value "
        "that later moves back.\n"
        "- **Flags come from models trained only on earlier seasons**; the hyperparameters were chosen on all four "
        "CV folds.\n"
        "- **The 2024/25 outcome uses 2025/26 valuations** (outcome only, never a feature or tuning input).\n"
        "- **2025/26 season-wide under-prediction** is handled by centring gaps on each season's median.")


def main() -> None:
    st.set_page_config(page_title="Player Value Analyzer", page_icon="⚽", layout="wide")
    values, stats = load_values(), load_stats()
    season = values["season"].iloc[0]

    st.title("Football Player Value Analyzer")
    st.write(
        f"**{season}, top-5 European leagues.** The model estimates the market value implied by a player's age, "
        "role, playing time, club and league context, trained on Transfermarkt values from 2012/13-2024/25. "
        "Market value is not \"true value\": the gap between market value and the estimate shows where the market "
        "pays a premium or a discount (hype, brand, potential, injuries, contract), not proven bargains.")

    filters = sidebar_filters(values)
    filtered = apply_filters(values, filters)
    under, over, lookup, performance = st.tabs(["Undervalued", "Overvalued", "Player lookup", "Model performance"])
    with under:
        tab_ranking(filtered, undervalued=True)
    with over:
        tab_ranking(filtered, undervalued=False)
    with lookup:
        tab_lookup(values, stats)
    with performance:
        tab_performance()

    st.divider()
    st.caption("Data: Transfermarkt, via the CC0 dataset "
               "[dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets). "
               "Market values are Transfermarkt's estimates, not transfer fees. Not affiliated with Transfermarkt.")


main()

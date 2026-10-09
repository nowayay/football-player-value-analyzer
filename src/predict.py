"""Value estimates and over/undervalued flags for one season, saved for the app and the AI Scout project.

The saved models were trained on 2012/13-2024/25, so only seasons from 2025/26 on are out of
sample; earlier seasons are refused (use src.backtest for out-of-fold flags of past seasons).
Flags use src.evaluate.add_gaps_and_flags: the gap is centred on the season's median gap,
because the model under-predicts the whole 2025/26 season by about 14% (see README).

Outputs:
    data/processed/player_values.csv   one row per player-season with value estimate and flag
    data/processed/player_stats.csv    key per-90 stats for the app's player lookup

Usage:
    python -m src.predict                # 2025/26
    python -m src.predict --season 2025
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.data import MODELS_DIR, PROCESSED_DIR, TEST_SEASON, TOP5_LEAGUES, season_label
from src.evaluate import add_gaps_and_flags
from src.features import FEATURES_PATH
from src.train import CONFORMAL_PATH, apply_correction

PLAYER_VALUES_PATH = PROCESSED_DIR / "player_values.csv"
PLAYER_STATS_PATH = PROCESSED_DIR / "player_stats.csv"
STATS_COLUMNS = ["minutes", "appearances", "minutes_share", "goals_p90", "assists_p90", "ga_p90", "yellow_p90",
                 "prev_minutes", "prev_ga_p90", "form_ga_p90", "club_ppg", "height_in_cm"]


def predict_season(features: pd.DataFrame, season: int) -> pd.DataFrame:
    """Return the value table for one season: estimates, 80% interval, raw/adjusted gaps and flags."""
    if season < TEST_SEASON:
        raise ValueError(f"{season_label(season)} was in the training data; use src.backtest for past seasons")
    rows = features[features["season"] == season].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"no feature rows for season {season_label(season)}")

    main_name = json.loads((MODELS_DIR / "model_config.json").read_text())["main_model"]
    correction = json.loads(CONFORMAL_PATH.read_text())[main_name]["correction_log"]
    pred = joblib.load(MODELS_DIR / f"{main_name}.joblib").predict(rows)
    low, high = apply_correction(joblib.load(MODELS_DIR / f"{main_name}_q10.joblib").predict(rows),
                                 joblib.load(MODELS_DIR / f"{main_name}_q90.joblib").predict(rows), correction)
    scored = add_gaps_and_flags(rows.assign(pred=pred, pred_low=low, pred_high=high))

    to_pct = lambda gap: (100 * (np.exp(gap) - 1)).round(1)  # noqa: E731 - log gap -> % above market
    return pd.DataFrame({
        "player_id": scored["player_id"],
        "season": scored["season"].map(season_label),
        "player": scored["name"],
        "club": scored["club_name"],  # main club by league minutes in that season (appearances table)
        "league": scored["competition_id"].map(TOP5_LEAGUES),
        "position": scored["position_group"],
        "sub_position": scored["sub_position"],
        "age": scored["age"].round(1),
        "minutes": scored["minutes"].astype(int),
        "market_value_eur": scored["market_value_eur"].astype(int),
        "predicted_value_eur": np.exp(scored["pred"]).round(-4).astype(int),
        "interval_low_eur": np.exp(scored["pred_low"]).round(-4).astype(int),
        "interval_high_eur": np.exp(scored["pred_high"]).round(-4).astype(int),
        "raw_gap_pct": to_pct(scored["raw_gap"]),
        "adjusted_gap_pct": to_pct(scored["adjusted_gap"]),
        "flag": scored["flag"],
        "flag_raw": scored["flag_raw"],
    }).sort_values("adjusted_gap_pct", ascending=False, kind="mergesort")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=TEST_SEASON, help="Transfermarkt season id, e.g. 2025")
    args = parser.parse_args()

    features = pd.read_parquet(FEATURES_PATH)
    values = predict_season(features, args.season)
    stats = features.loc[features["season"] == args.season, ["player_id", *STATS_COLUMNS]].round(3)
    values.to_csv(PLAYER_VALUES_PATH, index=False)
    stats.assign(season=season_label(args.season)).to_csv(PLAYER_STATS_PATH, index=False)

    pd.set_option("display.width", 220)
    cols = ["player", "club", "position", "age", "minutes", "market_value_eur", "predicted_value_eur",
            "adjusted_gap_pct", "flag"]
    fmt = {"market_value_eur": "{:,}".format, "predicted_value_eur": "{:,}".format,
           "adjusted_gap_pct": "{:+.0f}%".format}
    pl = values[values["league"] == "Premier League"]
    print(f"{season_label(args.season)}: {len(values):,} players, flags: {values['flag'].value_counts().to_dict()}")
    print("\nPremier League, top 10 adjusted gap (model above market, 900+ min):")
    print(pl[pl["minutes"] >= 900].head(10)[cols].to_string(index=False, formatters=fmt))
    print("\nPremier League, bottom 10 adjusted gap (market above model, 900+ min):")
    print(pl[pl["minutes"] >= 900].tail(10).iloc[::-1][cols].to_string(index=False, formatters=fmt))
    print(f"\nSaved {PLAYER_VALUES_PATH.relative_to(PROCESSED_DIR.parents[1])} and "
          f"{PLAYER_STATS_PATH.relative_to(PROCESSED_DIR.parents[1])}")


if __name__ == "__main__":
    main()

"""Feature engineering for the player-season table.

Leakage rule: every feature of a row is computed only from games dated <= that row's
valuation_date. Season stats, previous-season stats, club strength and the market level
are all computed "as of" the valuation date, so data from after it can never leak in.

Not used as features, on purpose:
- the player's own market value (current or lagged): it would make the gap meaningless;
- contract expiry and current club: they exist only as of today's snapshot (time mismatch).

Usage:
    python -m src.features   # build data/processed/features.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import (
    PLAYER_SEASON_PATH,
    PROCESSED_DIR,
    TEST_SEASON,
    TOP5_LEAGUES,
    aggregate_player_seasons,
    league_appearances,
    load_tables,
)

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
FIRST_SEASON = 2012
FORM_WEIGHTS = (2 / 3, 1 / 3)  # (current season, previous season)
PER90_MIN_MINUTES = 90  # per-90 denominator floor: a 10-minute cameo with a goal is not "9 goals per 90"

POSITION_LEVELS = ["GK", "DEF", "MID", "ATT"]
SUB_POSITION_LEVELS = [
    "Goalkeeper", "Centre-Back", "Left-Back", "Right-Back", "Defensive Midfield", "Central Midfield",
    "Attacking Midfield", "Left Midfield", "Right Midfield", "Left Winger", "Right Winger",
    "Second Striker", "Centre-Forward",
]
FOOT_LEVELS = ["right", "left", "both"]
LEAGUE_LEVELS = list(TOP5_LEAGUES)

NUMERIC_FEATURES = [
    # player profile
    "age", "age_sq", "height_in_cm",
    # season S (as of valuation_date)
    "minutes", "appearances", "minutes_share",
    "goals_p90", "assists_p90", "ga_p90", "yellow_p90", "red_p90",
    # season S-1, any domestic league in the dataset
    "has_prev_season", "prev_league_is_top5", "prev_minutes",
    "prev_goals_p90", "prev_assists_p90", "prev_ga_p90",
    # 2-season weighted form
    "form_goals_p90", "form_assists_p90", "form_ga_p90",
    # context
    "club_ppg", "club_rank_pct",
    # league-level market level (median of the league's S-1 targets, never the player's own value)
    "league_median_log_value_prev",
]
ONE_HOT_FEATURES = (
    [f"pos_{p}" for p in POSITION_LEVELS]
    + [f"subpos_{s}" for s in SUB_POSITION_LEVELS]
    + [f"foot_{f}" for f in FOOT_LEVELS]
    + [f"league_{lg}" for lg in LEAGUE_LEVELS]
)
FEATURES = NUMERIC_FEATURES + ONE_HOT_FEATURES
RIDGE_EXTRA = ["season_index"]  # seasons since 2012/13: a linear time trend, only for a Ridge variant

TARGET = "log_value"
# Kept next to the features for display and evaluation only; never fed to a model.
INFO_COLUMNS = ["player_id", "season", "name", "club_name", "competition_id", "position_group", "sub_position",
                "valuation_date", "market_value_eur", TARGET]


def per90(count: pd.Series | np.ndarray, minutes: pd.Series | np.ndarray) -> np.ndarray:
    """Events per 90 minutes. Minutes are floored at 90, so 0 minutes gives 0 (never inf/NaN)."""
    count = np.asarray(count, dtype=float)
    return 90 * count / np.maximum(np.asarray(minutes, dtype=float), PER90_MIN_MINUTES)


def domestic_league_appearances(
    appearances: pd.DataFrame, games: pd.DataFrame, competitions: pd.DataFrame
) -> pd.DataFrame:
    """Appearances in any domestic league in the dataset, with the game's season attached."""
    leagues = competitions.loc[competitions["type"] == "domestic_league", "competition_id"]
    apps = appearances[appearances["competition_id"].isin(leagues)]
    return apps.merge(games[["game_id", "season"]], on="game_id", how="inner", validate="many_to_one")


def _stats_as_of(rows: pd.DataFrame, apps: pd.DataFrame, season_offset: int) -> pd.DataFrame:
    """Aggregate a player's season (S + season_offset) stats from apps dated <= the row's valuation_date."""
    keys = rows[["player_id", "season", "valuation_date"]].assign(stat_season=rows["season"] + season_offset)
    merged = apps.merge(keys, left_on=["player_id", "season"], right_on=["player_id", "stat_season"],
                        suffixes=("_app", ""))
    merged = merged[merged["date"] <= merged["valuation_date"]]
    # aggregate per (player, row season) so the result joins back onto the rows
    return aggregate_player_seasons(merged.drop(columns="season_app"))


def _league_results(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (club, league game) with points and goals, for top-5 league games."""
    lg = games[games["competition_id"].isin(TOP5_LEAGUES)]
    cols = ["competition_id", "season", "date"]
    home = lg[cols].assign(club_id=lg["home_club_id"], gf=lg["home_club_goals"], ga=lg["away_club_goals"])
    away = lg[cols].assign(club_id=lg["away_club_id"], gf=lg["away_club_goals"], ga=lg["home_club_goals"])
    res = pd.concat([home, away], ignore_index=True)
    res["points"] = np.select([res["gf"] > res["ga"], res["gf"] == res["ga"]], [3, 1], 0)
    return res


def club_strength_as_of(rows: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Club points per game, league rank and games played, using only league games dated <= valuation_date.

    Rank uses points, then goal difference, then goals scored. club_rank_pct = rank / number of teams,
    so 0.05 is top of a 20-team league and 1.0 is bottom. The number of teams is fixed before the season.
    """
    res = _league_results(games)
    n_teams = res.groupby(["competition_id", "season"])["club_id"].nunique().rename("n_teams")
    out = []
    for (comp, season, as_of), group in rows.groupby(["competition_id", "season", "valuation_date"]):
        played = res[(res["competition_id"] == comp) & (res["season"] == season) & (res["date"] <= as_of)]
        table = played.groupby("club_id").agg(points=("points", "sum"), gf=("gf", "sum"), ga=("ga", "sum"),
                                              club_games=("points", "size"))
        table["gd"] = table["gf"] - table["ga"]
        table = table.sort_values(["points", "gd", "gf"], ascending=False)
        table["rank"] = np.arange(1, len(table) + 1)
        found = group[["player_id", "season", "club_id"]].merge(table, left_on="club_id", right_index=True, how="left")
        found["club_rank_pct"] = found["rank"] / n_teams.loc[(comp, season)]
        out.append(found)
    strength = pd.concat(out, ignore_index=True)
    strength["club_ppg"] = strength["points"] / strength["club_games"]
    return strength[["player_id", "season", "club_ppg", "club_rank_pct", "club_games"]]


def market_level_prev(rows: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """Median log value of the row's league in season S-1, using only S-1 targets dated <= valuation_date.

    A league-level aggregate of other players' previous-season values, never the player's own value.
    In practice nearly all S-1 targets are dated months earlier; the as-of rule just makes it strict.
    """
    t = targets[["competition_id", "season", "valuation_date", "log_value"]].sort_values("valuation_date")
    t = t.assign(level=t.groupby(["competition_id", "season"])["log_value"].transform(lambda s: s.expanding().median()))
    t = t.drop_duplicates(["competition_id", "season", "valuation_date"], keep="last")  # include all same-day rows
    t = t.assign(season=t["season"] + 1)  # season S-1 targets describe the market level for season S rows
    lookup = pd.merge_asof(
        rows[["competition_id", "season", "valuation_date"]].reset_index().sort_values("valuation_date"),
        t[["competition_id", "season", "valuation_date", "level"]],
        on="valuation_date", by=["competition_id", "season"], direction="backward",
    )
    return lookup.set_index("index")["level"].reindex(rows.index)


def build_features(
    player_season: pd.DataFrame,
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    competitions: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per player-season with INFO_COLUMNS + FEATURES + RIDGE_EXTRA.

    player_season provides the rows, targets and valuation dates (from src.data). All
    performance and club features are recomputed here from raw tables as of valuation_date.
    """
    base = player_season.reset_index(drop=True)
    rows = base[["player_id", "season", "valuation_date"]]

    # Season S stats (main club/league = most minutes up to valuation_date).
    current = _stats_as_of(rows, league_appearances(appearances, games), season_offset=0)
    # Drop the full-season aggregates from src.data; they are replaced by the as-of versions.
    df = base.drop(columns=["minutes", "appearances", "goals", "assists", "yellow_cards", "red_cards", "club_id",
                            "competition_id", "first_game_date", "last_game_date", "n_clubs"], errors="ignore")
    df = df.merge(current, on=["player_id", "season"], how="left", validate="one_to_one")
    df["goals_p90"] = per90(df["goals"], df["minutes"])
    df["assists_p90"] = per90(df["assists"], df["minutes"])
    df["ga_p90"] = per90(df["goals"] + df["assists"], df["minutes"])
    df["yellow_p90"] = per90(df["yellow_cards"], df["minutes"])
    df["red_p90"] = per90(df["red_cards"], df["minutes"])

    # Previous season (S-1), any domestic league. Missing -> NaN + has_prev_season = 0.
    prev = _stats_as_of(rows, domestic_league_appearances(appearances, games, competitions), season_offset=-1)
    prev = prev.assign(
        prev_minutes=prev["minutes"],
        prev_goals_p90=per90(prev["goals"], prev["minutes"]),
        prev_assists_p90=per90(prev["assists"], prev["minutes"]),
        prev_ga_p90=per90(prev["goals"] + prev["assists"], prev["minutes"]),
        prev_league_is_top5=prev["competition_id"].isin(TOP5_LEAGUES).astype(float),
    )
    prev_cols = ["prev_minutes", "prev_goals_p90", "prev_assists_p90", "prev_ga_p90", "prev_league_is_top5"]
    df = df.merge(prev[["player_id", "season"] + prev_cols], on=["player_id", "season"], how="left")
    df["has_prev_season"] = df["prev_minutes"].notna().astype(float)
    df["prev_league_is_top5"] = df["prev_league_is_top5"].fillna(0.0)

    w_cur, w_prev = FORM_WEIGHTS
    for stat in ["goals_p90", "assists_p90", "ga_p90"]:
        df[f"form_{stat}"] = w_cur * df[stat] + w_prev * df[f"prev_{stat}"]  # NaN when no previous season

    # Club strength and minutes share as of valuation_date.
    df = df.merge(club_strength_as_of(df, games), on=["player_id", "season"], how="left", validate="one_to_one")
    # Share of available minutes = league minutes / (90 x main club's league games so far), capped at 1
    # (a mid-season transfer can play more than the main club's games).
    df["minutes_share"] = np.clip(df["minutes"] / (90 * df["club_games"]), 0, 1)

    df["league_median_log_value_prev"] = market_level_prev(df, player_season)

    df["age"] = (df["valuation_date"] - df["date_of_birth"]).dt.days / 365.25
    df["age_sq"] = df["age"] ** 2
    df["season_index"] = df["season"] - FIRST_SEASON

    # One-hot encodings with fixed levels, so every season gets identical columns.
    for p in POSITION_LEVELS:
        df[f"pos_{p}"] = (df["position_group"] == p).astype(float)
    for s in SUB_POSITION_LEVELS:
        df[f"subpos_{s}"] = (df["sub_position"] == s).astype(float)
    for f in FOOT_LEVELS:
        df[f"foot_{f}"] = (df["foot"] == f).astype(float)
    for lg in LEAGUE_LEVELS:
        df[f"league_{lg}"] = (df["competition_id"] == lg).astype(float)

    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    return df[INFO_COLUMNS + FEATURES + RIDGE_EXTRA]


def main() -> None:
    player_season = pd.read_parquet(PLAYER_SEASON_PATH)
    tables = load_tables(["appearances", "games", "competitions"])
    features = build_features(player_season, tables["appearances"], tables["games"], tables["competitions"])
    features.to_parquet(FEATURES_PATH, index=False)

    dev = features[features["season"] < TEST_SEASON]
    print(f"Saved {FEATURES_PATH.name}: {len(features):,} rows ({len(dev):,} development, "
          f"{len(features) - len(dev):,} test season - not inspected)")
    print(f"\n{len(FEATURES)} features ({len(NUMERIC_FEATURES)} numeric + {len(ONE_HOT_FEATURES)} one-hot), "
          f"RIDGE_EXTRA = {RIDGE_EXTRA}")
    nan_share = dev[FEATURES + RIDGE_EXTRA].isna().mean().mul(100).round(2)
    print("\nNaN share per feature (development seasons, %):")
    print(nan_share.to_string())


if __name__ == "__main__":
    main()

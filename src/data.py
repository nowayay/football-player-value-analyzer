"""Download, load, clean and merge the Transfermarkt open dataset into a player-season table.

Source: https://github.com/dcaribou/transfermarkt-datasets (CC0-1.0), data © Transfermarkt.

One row = one player in one top-5 league season (Transfermarkt's `season` 2025 means 2025/26).

Target rule: the player's FIRST market valuation dated on/after their last top-5 league
appearance of season S, and no later than 31 October of the following calendar year.
Taking the first valuation after the last game means the market had seen every game
our stats use. Transfermarkt mostly revalues in late May/June, so a window anchored on
30 June would miss almost all recent seasons (see stage 1 inspection).

Usage:
    python -m src.data --download   # fetch raw tables into data/raw/
    python -m src.data --inspect    # print schema, row counts, missing values, coverage
    python -m src.data              # build data/processed/player_season.parquet
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
PLAYER_SEASON_PATH = PROCESSED_DIR / "player_season.parquet"

# Public mirror published by the dataset maintainer (same files as the Kaggle dataset).
BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = ["players", "player_valuations", "appearances", "games", "club_games", "clubs", "competitions"]

# Transfermarkt competition ids of the top-5 European leagues.
TOP5_LEAGUES = {"GB1": "Premier League", "ES1": "LaLiga", "IT1": "Serie A", "L1": "Bundesliga", "FR1": "Ligue 1"}
POSITION_GROUPS = {"Goalkeeper": "GK", "Defender": "DEF", "Midfield": "MID", "Attack": "ATT"}

TEST_SEASON = 2025  # 2025/26: latest complete season, held out until the final evaluation
MIN_MINUTES = 450  # about 5 full league games; below this, per-90 stats are mostly noise
VALUATION_DEADLINE = (10, 31)  # (month, day) in year S+1: latest accepted valuation date

# Columns parsed as dates, per table (only those that exist are parsed).
DATE_COLUMNS = {
    "players": ["date_of_birth", "contract_expiration_date"],
    "player_valuations": ["date"],
    "appearances": ["date"],
    "games": ["date"],
}


def season_label(season: int) -> str:
    """Format a Transfermarkt season id, e.g. 2025 -> '2025/26'."""
    return f"{season}/{str(season + 1)[-2:]}"


def download_raw(tables: list[str] = TABLES, overwrite: bool = False) -> None:
    """Download the raw .csv.gz tables into data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for table in tables:
        dest = RAW_DIR / f"{table}.csv.gz"
        if dest.exists() and not overwrite:
            print(f"  {dest.name} already exists, skipping")
            continue
        print(f"  downloading {table} ...")
        # The mirror rejects Python's default User-Agent with 403.
        request = urllib.request.Request(f"{BASE_URL}/{table}.csv.gz", headers={"User-Agent": "curl/8.0"})
        tmp = dest.with_suffix(".part")  # write to a temp file so an interrupted download leaves no corrupt file
        with urllib.request.urlopen(request) as response, open(tmp, "wb") as f:
            f.write(response.read())
        tmp.rename(dest)


def load_table(name: str) -> pd.DataFrame:
    """Load one raw table (.csv.gz from the mirror or .csv from Kaggle) and parse its date columns."""
    path = next((p for p in [RAW_DIR / f"{name}.csv.gz", RAW_DIR / f"{name}.csv"] if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"{name}.csv(.gz) not found in {RAW_DIR}. Run `python -m src.data --download` first.")
    df = pd.read_csv(path, low_memory=False)
    for col in DATE_COLUMNS.get(name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_tables(tables: list[str] = TABLES) -> dict[str, pd.DataFrame]:
    """Load raw tables into a dict keyed by table name."""
    return {name: load_table(name) for name in tables}


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return count and % of missing values per column (only columns with any missing)."""
    n_missing = df.isna().sum()
    report = pd.DataFrame({"n_missing": n_missing, "pct_missing": (100 * n_missing / len(df)).round(1)})
    return report[report["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def league_appearances(appearances: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Top-5 league appearances with the game's season attached."""
    apps = appearances[appearances["competition_id"].isin(TOP5_LEAGUES)]
    return apps.merge(games[["game_id", "season"]], on="game_id", how="inner", validate="many_to_one")


def aggregate_player_seasons(apps: pd.DataFrame) -> pd.DataFrame:
    """Sum league stats per player-season; main club/league = where the player played most minutes."""
    keys = ["player_id", "season"]
    stats = apps.groupby(keys).agg(
        minutes=("minutes_played", "sum"),
        appearances=("game_id", "size"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
        first_game_date=("date", "min"),
        last_game_date=("date", "max"),
    )
    # Players who moved clubs mid-season: keep the club (and its league) with the most minutes.
    by_club = apps.groupby(keys + ["player_club_id", "competition_id"], as_index=False)["minutes_played"].sum()
    main_club = (
        by_club.sort_values("minutes_played", ascending=False, kind="stable")
        .drop_duplicates(keys)
        .set_index(keys)[["player_club_id", "competition_id"]]
        .rename(columns={"player_club_id": "club_id"})
    )
    n_clubs = by_club.groupby(keys)["player_club_id"].nunique().rename("n_clubs")
    return stats.join(main_club).join(n_clubs).reset_index()


def attach_target(player_seasons: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    """Attach the first valuation on/after the player's last league game, if before the deadline.

    Adds valuation_date, market_value_eur, log_value and days_since_last_game. Rows without
    a valuation in the window get NaN (they are counted and dropped by the caller).
    """
    vals = valuations.loc[valuations["market_value_in_eur"] > 0, ["player_id", "date", "market_value_in_eur"]]
    vals = vals.rename(columns={"date": "valuation_date", "market_value_in_eur": "market_value_eur"})
    # merge_asof(direction="forward") finds, per player, the first valuation with date >= last_game_date.
    merged = pd.merge_asof(
        player_seasons.sort_values("last_game_date"),
        vals.sort_values("valuation_date"),
        left_on="last_game_date",
        right_on="valuation_date",
        by="player_id",
        direction="forward",
    )
    month, day = VALUATION_DEADLINE
    deadline = pd.to_datetime(dict(year=merged["season"] + 1, month=month, day=day))
    too_late = merged["valuation_date"] > deadline
    merged.loc[too_late, ["valuation_date", "market_value_eur"]] = np.nan
    merged["log_value"] = np.log(merged["market_value_eur"])
    merged["days_since_last_game"] = (merged["valuation_date"] - merged["last_game_date"]).dt.days
    return merged.sort_values(["season", "player_id"]).reset_index(drop=True)


def build_player_season(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the modelling table and a per-season report of row counts after each filter.

    Returns (player_season, filter_counts).
    """
    games, players = tables["games"], tables["players"]
    seasons = aggregate_player_seasons(league_appearances(tables["appearances"], games))

    # Context columns: the league's final matchday (for the sensitivity check) and names.
    league_games = games[games["competition_id"].isin(TOP5_LEAGUES)]
    final_matchday = league_games.groupby(["competition_id", "season"])["date"].max().rename("league_final_matchday")
    seasons = seasons.merge(final_matchday.reset_index(), on=["competition_id", "season"], how="left")
    club_names = pd.concat([
        games[["home_club_id", "home_club_name"]].set_axis(["club_id", "club_name"], axis=1),
        games[["away_club_id", "away_club_name"]].set_axis(["club_id", "club_name"], axis=1),
    ]).dropna().drop_duplicates("club_id")
    seasons = seasons.merge(club_names, on="club_id", how="left")

    # Static player attributes. Position, foot and height are the snapshot values; they rarely
    # change, but a player's listed position can differ from the one played years ago.
    player_cols = ["player_id", "name", "date_of_birth", "position", "sub_position", "foot", "height_in_cm",
                   "country_of_citizenship"]
    seasons = seasons.merge(players[player_cols], on="player_id", how="left", validate="many_to_one")
    seasons["position_group"] = seasons["position"].map(POSITION_GROUPS)  # "Missing" -> NaN

    seasons = attach_target(seasons, tables["player_valuations"])
    seasons["age"] = (seasons["valuation_date"] - seasons["date_of_birth"]).dt.days / 365.25

    # Apply filters one at a time and record how many rows survive each step.
    steps = [
        ("all top-5 league player-seasons", pd.Series(True, index=seasons.index)),
        (f"minutes >= {MIN_MINUTES}", seasons["minutes"] >= MIN_MINUTES),
        ("known position and birth date", seasons["position_group"].notna() & seasons["date_of_birth"].notna()),
        ("valuation in target window", seasons["valuation_date"].notna()),
    ]
    keep = pd.Series(True, index=seasons.index)
    counts = {}
    for label, mask in steps:
        keep &= mask
        counts[label] = seasons[keep].groupby("season").size()
    filter_counts = pd.DataFrame(counts).fillna(0).astype(int)
    filter_counts.loc["total"] = filter_counts.sum()

    player_season = seasons[keep].reset_index(drop=True)
    return player_season, filter_counts


def build(save: bool = True) -> pd.DataFrame:
    """Build, report on and save the player-season table."""
    tables = load_tables(["players", "player_valuations", "appearances", "games"])
    player_season, filter_counts = build_player_season(tables)

    pd.set_option("display.width", 160, "display.max_columns", 30)
    print("Rows remaining after each filter, per season:")
    print(filter_counts.to_string())
    print(f"\nFinal table: {len(player_season):,} rows x {player_season.shape[1]} cols")
    print("\nMissing values in final table:")
    print(missing_report(player_season).to_string())
    print("\nDays from last league game to valuation (percentiles):")
    print(player_season["days_since_last_game"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
    before_final = player_season["valuation_date"] < player_season["league_final_matchday"]
    print(f"\nValuations dated before the league's final matchday: {before_final.mean():.1%} "
          "(players whose season ended early, e.g. injury or a winter move)")

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        player_season.to_parquet(PLAYER_SEASON_PATH, index=False)
        filter_counts.to_csv(REPORTS_DIR / "filter_counts.csv")
        print(f"\nSaved {PLAYER_SEASON_PATH.relative_to(ROOT)} and reports/filter_counts.csv")
    return player_season


def inspect(tables: dict[str, pd.DataFrame]) -> None:
    """Print schema, row counts, missing values and data coverage relevant to the project."""
    pd.set_option("display.width", 160, "display.max_columns", 30)

    for name, df in tables.items():
        print(f"\n=== {name}: {len(df):,} rows x {df.shape[1]} cols ===")
        print(", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items()))
        miss = missing_report(df)
        if not miss.empty:
            print("missing:", "; ".join(f"{c} {p}%" for c, p in miss["pct_missing"].items()))

    games, apps, vals = tables["games"], tables["appearances"], tables["player_valuations"]

    print("\n=== Date coverage ===")
    for name, df in [("games", games), ("appearances", apps), ("player_valuations", vals)]:
        print(f"{name:18s} {df['date'].min().date()} -> {df['date'].max().date()}")

    print("\n=== Top-5 league coverage (league games per season) ===")
    league = games[games["competition_id"].isin(TOP5_LEAGUES)]
    print(league.pivot_table(index="season", columns="competition_id", values="game_id", aggfunc="count").fillna(0).astype(int))

    print("\n=== Appearances per season in top-5 league games ===")
    apps_top5 = league_appearances(apps, games)
    print(apps_top5.groupby("season").agg(appearances=("game_id", "size"), players=("player_id", "nunique")))

    print("\n=== Valuation snapshots per year and month ===")
    v = vals.assign(year=vals["date"].dt.year, month=vals["date"].dt.month)
    print(v[v["year"] >= 2012].pivot_table(index="year", columns="month", values="player_id", aggfunc="count").fillna(0).astype(int))

    print("\n=== Valuations per player (how often is a player re-valued?) ===")
    per_player = vals.groupby("player_id").size()
    print(per_player.describe().round(1).to_string())
    print(f"\nmarket_value_in_eur: min {vals['market_value_in_eur'].min():,.0f}, median "
          f"{vals['market_value_in_eur'].median():,.0f}, max {vals['market_value_in_eur'].max():,.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="download raw tables into data/raw/")
    parser.add_argument("--inspect", action="store_true", help="print schema and coverage report")
    args = parser.parse_args()
    if args.download:
        download_raw()
    if args.inspect:
        inspect(load_tables())
    if not (args.download or args.inspect):
        build()


if __name__ == "__main__":
    main()

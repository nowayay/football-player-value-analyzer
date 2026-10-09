"""Download, load and inspect the Transfermarkt open dataset.

Source: https://github.com/dcaribou/transfermarkt-datasets (CC0-1.0), data © Transfermarkt.

Stage 1 scope: download the raw tables and report schema + coverage.
The player-season table is built in stage 2.

Usage:
    python -m src.data --download   # fetch raw tables into data/raw/
    python -m src.data --inspect    # print schema, row counts, missing values, coverage
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Public mirror published by the dataset maintainer (same files as the Kaggle dataset).
BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = ["players", "player_valuations", "appearances", "games", "club_games", "clubs", "competitions"]

# Transfermarkt competition ids of the top-5 European leagues.
TOP5_LEAGUES = {"GB1": "Premier League", "ES1": "LaLiga", "IT1": "Serie A", "L1": "Bundesliga", "FR1": "Ligue 1"}

# Columns parsed as dates, per table (only those that exist are parsed).
DATE_COLUMNS = {
    "players": ["date_of_birth", "contract_expiration_date"],
    "player_valuations": ["date"],
    "appearances": ["date"],
    "games": ["date"],
}


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
    """Load one raw table and parse its date columns."""
    path = RAW_DIR / f"{name}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data --download` first.")
    df = pd.read_csv(path, low_memory=False)
    for col in DATE_COLUMNS.get(name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_tables(tables: list[str] = TABLES) -> dict[str, pd.DataFrame]:
    """Load all raw tables into a dict keyed by table name."""
    return {name: load_table(name) for name in tables}


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return count and % of missing values per column (only columns with any missing)."""
    n_missing = df.isna().sum()
    report = pd.DataFrame({"n_missing": n_missing, "pct_missing": (100 * n_missing / len(df)).round(1)})
    return report[report["n_missing"] > 0].sort_values("pct_missing", ascending=False)


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
    apps_top5 = apps[apps["competition_id"].isin(TOP5_LEAGUES)].merge(games[["game_id", "season"]], on="game_id")
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
        parser.print_help()


if __name__ == "__main__":
    main()

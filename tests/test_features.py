"""Leakage and correctness tests for src.data + src.features on a small synthetic dataset.

Fixture: a 4-club league (GB1) over seasons 2012/13 and 2013/14, one player per club.
Player 10 gets injured in March 2014: their season-2013 valuation (15 March 2014) comes
before the club's remaining games, which is exactly where full-season features would leak.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from src.data import build_player_season
from src.features import FEATURES, FEATURES_PATH, RIDGE_EXTRA, TARGET, build_features, per90

INJURED_PLAYER, INJURED_SEASON = 10, 2013
INJURY_DATE = pd.Timestamp("2014-03-01")
INJURED_VALUATION = pd.Timestamp("2014-03-15")
PAIRINGS = [((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3))]


def make_tables() -> dict[str, pd.DataFrame]:
    """Build a tiny, deterministic version of the raw Transfermarkt tables."""
    rng = np.random.default_rng(0)
    games, apps = [], []
    game_id = 0
    for season in [2012, 2013]:
        for rnd in range(38):
            date = pd.Timestamp(f"{season}-08-10") + pd.Timedelta(weeks=rnd)
            for home, away in PAIRINGS[rnd % 3]:
                game_id += 1
                games.append({"game_id": game_id, "competition_id": "GB1", "season": season, "date": date,
                              "home_club_id": home, "away_club_id": away,
                              "home_club_goals": int(rng.integers(0, 4)), "away_club_goals": int(rng.integers(0, 4)),
                              "home_club_name": f"Club {home}", "away_club_name": f"Club {away}"})
                for club in (home, away):
                    player = 9 + club  # players 10-13 play for clubs 1-4
                    if player == INJURED_PLAYER and season == INJURED_SEASON and date > INJURY_DATE:
                        continue
                    apps.append({"game_id": game_id, "player_id": player, "player_club_id": club, "date": date,
                                 "competition_id": "GB1", "goals": int(rng.integers(0, 2)),
                                 "assists": int(rng.integers(0, 2)), "yellow_cards": 0, "red_cards": 0,
                                 "minutes_played": 90})
    # Player 14: unused sub (0 minutes) in another league in 2012, then plays for club 4 in 2013.
    games.append({"game_id": 999, "competition_id": "NL1", "season": 2012, "date": pd.Timestamp("2012-09-01"),
                  "home_club_id": 50, "away_club_id": 51, "home_club_goals": 1, "away_club_goals": 1,
                  "home_club_name": "Club 50", "away_club_name": "Club 51"})
    apps.append({"game_id": 999, "player_id": 14, "player_club_id": 50, "date": pd.Timestamp("2012-09-01"),
                 "competition_id": "NL1", "goals": 0, "assists": 0, "yellow_cards": 0, "red_cards": 0,
                 "minutes_played": 0})
    for app in [a for a in apps if a["player_id"] == 13 and a["date"] >= pd.Timestamp("2013-08-01")]:
        apps.append({**app, "player_id": 14, "goals": 0})

    players = pd.DataFrame({
        "player_id": [10, 11, 12, 13, 14],
        "name": [f"Player {i}" for i in range(10, 15)],
        "date_of_birth": pd.to_datetime(["1990-01-01", "1992-05-05", "1995-03-03", "1988-07-07", "1996-02-02"]),
        "position": ["Attack", "Midfield", "Defender", "Goalkeeper", "Attack"],
        "sub_position": ["Centre-Forward", "Central Midfield", "Centre-Back", "Goalkeeper", "Left Winger"],
        "foot": ["right", "left", "right", None, "both"],
        "height_in_cm": [180.0, 175.0, 190.0, 195.0, np.nan],
        "country_of_citizenship": ["England"] * 5,
    })
    valuations = [{"player_id": p, "date": pd.Timestamp(d), "market_value_in_eur": v}
                  for p in range(10, 15)
                  for d, v in [("2013-06-10", 1_000_000 * p), ("2014-06-10", 2_000_000 * p)]]
    valuations.append({"player_id": INJURED_PLAYER, "date": INJURED_VALUATION, "market_value_in_eur": 15_000_000})
    competitions = pd.DataFrame({"competition_id": ["GB1", "NL1"], "type": ["domestic_league", "domestic_league"]})
    return {"games": pd.DataFrame(games), "appearances": pd.DataFrame(apps), "players": players,
            "player_valuations": pd.DataFrame(valuations), "competitions": competitions}


def features_for(player_season: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run feature building and index the result by (player_id, season)."""
    out = build_features(player_season, tables["appearances"], tables["games"], tables["competitions"])
    return out.set_index(["player_id", "season"]).sort_index()


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return make_tables()


@pytest.fixture(scope="module")
def player_season(tables) -> pd.DataFrame:
    return build_player_season(tables)[0]


def test_injured_player_row_uses_early_valuation(player_season):
    """Sanity check of the fixture: the injured player's target predates the club's last games."""
    row = player_season.set_index(["player_id", "season"]).loc[(INJURED_PLAYER, INJURED_SEASON)]
    assert row["valuation_date"] == INJURED_VALUATION
    assert row["valuation_date"] < row["league_final_matchday"]


def test_features_ignore_data_after_valuation_date(tables, player_season):
    """Test 1: altering appearances, games and valuations dated after valuation_date changes nothing."""
    cutoff = INJURED_VALUATION
    key = (INJURED_PLAYER, INJURED_SEASON)
    before = features_for(player_season, tables).loc[key, FEATURES]

    altered = {name: df.copy() for name, df in tables.items()}
    games, apps = altered["games"], altered["appearances"]
    late_games = games["date"] > cutoff
    games.loc[late_games, ["home_club_goals", "away_club_goals"]] = [[9, 0]]  # change every later result
    late_apps = apps["date"] > cutoff
    apps.loc[late_apps, ["goals", "assists", "minutes_played"]] = [[5, 5, 45]]
    # the injured player "returns" after the valuation date: must not affect this row
    comeback = apps[(apps["player_id"] == 13) & late_apps & (apps["date"].dt.year == 2014)].assign(
        player_id=INJURED_PLAYER, player_club_id=1)
    altered["appearances"] = pd.concat([apps, comeback], ignore_index=True)
    # valuations after the cutoff: these are the other rows' targets; scale them all
    late_targets = player_season["valuation_date"] > cutoff
    altered_ps = player_season.copy()
    altered_ps.loc[late_targets, "market_value_eur"] *= 3
    altered_ps.loc[late_targets, TARGET] = np.log(altered_ps.loc[late_targets, "market_value_eur"])

    after = features_for(altered_ps, altered).loc[key, FEATURES]
    pd.testing.assert_series_equal(before, after)


def test_club_strength_is_computed_as_of_valuation_date(tables, player_season):
    """Club points per game must equal the value from club 1's games dated <= valuation_date."""
    games = tables["games"]
    played = games[(games["season"] == INJURED_SEASON) & (games["competition_id"] == "GB1")
                   & (games["date"] <= INJURED_VALUATION)]
    home, away = played[played["home_club_id"] == 1], played[played["away_club_id"] == 1]
    gf = pd.concat([home["home_club_goals"], away["away_club_goals"]])
    ga = pd.concat([home["away_club_goals"], away["home_club_goals"]])
    expected_ppg = (3 * (gf.to_numpy() > ga.to_numpy()) + (gf.to_numpy() == ga.to_numpy())).mean()
    feats = features_for(player_season, tables)
    assert feats.loc[(INJURED_PLAYER, INJURED_SEASON), "club_ppg"] == pytest.approx(expected_ppg)


def test_market_level_ignores_previous_season_targets_dated_later(tables, player_season):
    """The S-1 league median only uses S-1 targets dated <= the row's valuation date."""
    ps = player_season.copy()
    prev = ps["season"] == INJURED_SEASON - 1
    ps.loc[prev & (ps["player_id"] == 11), "valuation_date"] = INJURED_VALUATION + pd.Timedelta(days=1)
    ps.loc[prev & (ps["player_id"] == 11), TARGET] = 99.0  # would shift the median if it leaked in
    feats = features_for(ps, tables)
    visible = ps[prev & (ps["valuation_date"] <= INJURED_VALUATION)][TARGET]
    assert feats.loc[(INJURED_PLAYER, INJURED_SEASON), "league_median_log_value_prev"] == pytest.approx(visible.median())


def test_season_targets_do_not_change_same_season_features(tables, player_season):
    """Test 2: changing season-S market values leaves season-S features unchanged."""
    before = features_for(player_season, tables)
    altered = player_season.copy()
    in_season = altered["season"] == INJURED_SEASON
    altered.loc[in_season, "market_value_eur"] *= 10
    altered.loc[in_season, TARGET] = np.log(altered.loc[in_season, "market_value_eur"])
    after = features_for(altered, tables)
    rows = before.index.get_level_values("season") == INJURED_SEASON
    pd.testing.assert_frame_equal(before.loc[rows, FEATURES], after.loc[rows, FEATURES])


def test_feature_names_contain_no_target_or_snapshot_columns():
    """Test 3a: no own market value, valuation, contract or current-club columns among model inputs."""
    forbidden = re.compile(r"market_value|valuation|contract|current_club|log_value|target|highest", re.I)
    # The one allowed exception: a league-level median of the PREVIOUS season's targets (never the
    # player's own value), tested for leakage above.
    allowed = {"league_median_log_value_prev"}
    offenders = [c for c in FEATURES + RIDGE_EXTRA if forbidden.search(c) and c not in allowed]
    assert offenders == []
    assert TARGET not in FEATURES + RIDGE_EXTRA


def test_target_is_log_of_market_value(tables, player_season):
    """Test 3b: the target is the natural log of the market value in EUR, in both tables."""
    np.testing.assert_allclose(player_season[TARGET], np.log(player_season["market_value_eur"]))
    feats = features_for(player_season, tables)
    np.testing.assert_allclose(feats[TARGET], np.log(feats["market_value_eur"]))


def test_per90_handles_zero_minutes():
    """Test 4: per-90 with 0 minutes gives finite values and no division warnings."""
    with np.errstate(all="raise"):
        result = per90(np.array([0, 1, 3, 2]), np.array([0, 0, 45, 180]))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, [0.0, 1.0, 3.0, 1.0])  # minutes floored at 90


def test_zero_minute_previous_season_gives_finite_features(tables, player_season):
    """Test 4b: a previous season with only 0-minute appearances still yields finite per-90 features."""
    row = features_for(player_season, tables).loc[(14, 2013)]
    assert row["has_prev_season"] == 1 and row["prev_minutes"] == 0
    assert np.isfinite(row[["prev_goals_p90", "prev_assists_p90", "prev_ga_p90", "form_ga_p90"]].astype(float)).all()


def test_first_season_has_no_previous_season(tables, player_season):
    """Test 5: 2012/13 rows have has_prev_season == 0 and NaN previous-season stats (fixture)."""
    first = features_for(player_season, tables).xs(2012, level="season")
    assert (first["has_prev_season"] == 0).all()
    assert first[["prev_minutes", "prev_ga_p90", "league_median_log_value_prev"]].isna().all().all()


@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="run `python -m src.features` to build the real table")
def test_first_season_has_no_previous_season_real_data():
    """Test 5b: same check on the real feature table, when it has been built."""
    feats = pd.read_parquet(FEATURES_PATH, columns=["season", "has_prev_season"])
    assert (feats.loc[feats["season"] == 2012, "has_prev_season"] == 0).all()

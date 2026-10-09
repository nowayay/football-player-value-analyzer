"""Tests for the shared flag rule, the backtest outcome and out-of-fold flags, and SHAP grouping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import next_year_outcome, out_of_fold_predictions
from src.data import TEST_SEASON
from src.evaluate import add_gaps_and_flags
from src.explain import grouped_shap
from src.features import TARGET
from src.train import CV_VALID_SEASONS


def make_scored(n_per_season: int = 201, seed: int = 0) -> pd.DataFrame:
    """Synthetic predictions with a different overall bias in each season."""
    rng = np.random.default_rng(seed)
    frames = []
    for season, bias in [(2023, -0.2), (2024, 0.1)]:
        y = rng.normal(15, 1, n_per_season)
        pred = y + bias + rng.normal(0, 0.4, n_per_season)
        frames.append(pd.DataFrame({"season": season, "log_value": y, "pred": pred, "pred_low": pred - 0.5,
                                    "pred_high": pred + 0.5, "minutes": rng.integers(450, 3400, n_per_season)}))
    return pd.concat(frames, ignore_index=True)


def test_adjusted_gap_has_zero_median_per_season():
    """(a) after centring, the median adjusted gap is 0 within every season."""
    scored = add_gaps_and_flags(make_scored())
    medians = scored.groupby("season")["adjusted_gap"].median()
    np.testing.assert_allclose(medians.to_numpy(), 0.0, atol=1e-12)


def test_interval_is_shifted_by_the_season_constant_and_ordered():
    """(b) the interval moves by exactly the season shift, keeps its width and upper >= lower."""
    scored = add_gaps_and_flags(make_scored())
    np.testing.assert_allclose(scored["adjusted_low"], scored["pred_low"] - scored["season_shift"])
    np.testing.assert_allclose(scored["adjusted_high"], scored["pred_high"] - scored["season_shift"])
    assert (scored["adjusted_high"] >= scored["adjusted_low"]).all()
    assert scored["season_shift"].nunique() == 2  # one constant per season


def test_shift_changes_flags_as_expected():
    """A season-wide under-prediction no longer marks everyone overvalued once centred."""
    df = pd.DataFrame({"season": 2025, "log_value": 15.0, "pred": [14.4, 14.5, 14.6],
                       "pred_low": [14.2, 14.3, 14.4], "pred_high": [14.6, 14.7, 14.8], "minutes": 2000})
    scored = add_gaps_and_flags(df)
    assert scored["flag_raw"].tolist() == ["overvalued"] * 3
    assert scored["flag"].tolist() == ["in range"] * 3


def test_flags_respect_900_minute_threshold():
    """(e) a player far outside the interval is only flagged with >= 900 minutes."""
    df = pd.DataFrame({"season": 2025, "log_value": [10.0, 10.0, 15.0, 15.0, 15.0],
                       "pred": [15.0, 15.0, 15.0, 15.0, 15.0], "pred_low": 14.5, "pred_high": 15.5,
                       "minutes": [899, 900, 2000, 2000, 2000]})
    flags = add_gaps_and_flags(df)["flag"].tolist()
    assert flags[:2] == ["not rated", "undervalued"]


def test_outcome_uses_only_later_valuations_in_window():
    """(c) outcome = valuation closest to +365 days within 330-400 days; never an earlier one."""
    rows = pd.DataFrame({"player_id": [1, 2], "season": [2023, 2023],
                         "valuation_date": pd.to_datetime(["2024-06-01", "2024-06-01"]),
                         TARGET: np.log([10e6, 10e6])})
    base = pd.Timestamp("2024-06-01")
    valuations = pd.DataFrame({
        "player_id": [1, 1, 1, 1, 1, 1, 2, 2],
        "date": [base - pd.Timedelta(days=d) for d in (365, 1)] + [base + pd.Timedelta(days=d) for d in (200, 340, 360, 420)]
        + [base - pd.Timedelta(days=365), base + pd.Timedelta(days=100)],
        "market_value_in_eur": [99e6, 99e6, 99e6, 12e6, 20e6, 99e6, 99e6, 99e6],
    })
    out = next_year_outcome(rows, valuations).set_index("player_id")
    assert out.loc[1, "outcome_date"] == base + pd.Timedelta(days=360)
    assert out.loc[1, "delta_log_value"] == np.log(20e6) - np.log(10e6)
    assert np.isnan(out.loc[2, "delta_log_value"])  # no valuation 330-400 days later
    assert (out["outcome_date"].dropna() > rows.set_index("player_id")["valuation_date"].reindex(out.dropna().index)).all()


class SpyModel:
    """Records the latest training season and predicts a constant."""

    calls: list[tuple[int, int]] = []

    def __init__(self, value: float):
        self.value = value

    def fit(self, df: pd.DataFrame, y: pd.Series) -> SpyModel:
        self.trained_through = int(df["season"].max())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        SpyModel.calls.append((self.trained_through, int(df["season"].iloc[0])))
        return np.full(len(df), self.value)


def test_out_of_fold_flags_come_from_models_trained_on_earlier_seasons():
    """(d) predictions for season S come from models whose training data ends before S."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame([{"season": s, "valuation_date": pd.Timestamp(f"{s + 1}-06-01") + pd.Timedelta(days=i),
                        TARGET: rng.normal(15, 1)} for s in range(2012, TEST_SEASON + 1) for i in range(20)])
    SpyModel.calls = []
    oof = out_of_fold_predictions(df, lambda: SpyModel(15.0), lambda: SpyModel(14.0), lambda: SpyModel(16.0))
    assert sorted(oof["season"].unique()) == CV_VALID_SEASONS
    assert (oof["trained_through"] < oof["season"]).all()
    assert all(trained < predicted for trained, predicted in SpyModel.calls)
    assert TEST_SEASON not in set(oof["season"])
    assert (oof["pred_high"] >= oof["pred_low"]).all()
    assert (oof.loc[oof["season"] == CV_VALID_SEASONS[0], "interval_correction"] == 0).all()  # no earlier fold


def test_grouped_shap_keeps_market_level_separate_and_preserves_sums():
    columns = ["league_GB1", "league_ES1", "league_median_log_value_prev", "pos_GK", "age"]
    values = np.arange(10, dtype=float).reshape(2, 5)
    grouped = grouped_shap(values, columns)
    assert set(grouped.columns) == {"league", "league_median_log_value_prev", "position group", "age"}
    np.testing.assert_allclose(grouped.sum(axis=1), values.sum(axis=1))
    np.testing.assert_allclose(grouped["league"], values[:, 0] + values[:, 1])

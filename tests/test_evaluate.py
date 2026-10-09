"""Tests for the time split, metric functions, interval flags and residual ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import TEST_SEASON
from src.evaluate import interval_flag, rank_by_residual, regression_metrics
from src.features import TARGET
from src.train import CV_VALID_SEASONS, expanding_window_folds, make_xgb


def make_split_frame() -> pd.DataFrame:
    """Two rows per season 2012-2025; one 2020 row is dated after the first 2021 valuation."""
    rows = []
    for season in range(2012, TEST_SEASON + 1):
        rows.append({"season": season, "valuation_date": pd.Timestamp(f"{season + 1}-06-01")})
        rows.append({"season": season, "valuation_date": pd.Timestamp(f"{season + 1}-06-15")})
    rows.append({"season": 2021, "valuation_date": pd.Timestamp("2021-12-01")})  # early-ended 2021/22 season
    rows.append({"season": 2020, "valuation_date": pd.Timestamp("2021-12-20")})  # late 2020/21 target
    return pd.DataFrame(rows)


def test_folds_train_only_on_earlier_seasons_and_dates():
    df = make_split_frame()
    folds = list(expanding_window_folds(df))
    assert [season for season, _, _ in folds] == CV_VALID_SEASONS
    for season, train, valid in folds:
        assert train.any() and valid.any()
        assert not (train & valid).any()
        assert df.loc[train, "season"].max() < season
        assert (df.loc[valid, "season"] == season).all()
        # no date overlap: every training target is older than every validation target
        assert df.loc[train, "valuation_date"].max() < df.loc[valid, "valuation_date"].min()
        assert TEST_SEASON not in set(df.loc[train | valid, "season"])


def test_late_target_is_purged_from_training():
    df = make_split_frame()
    _, train, _ = next(f for f in expanding_window_folds(df) if f[0] == 2021)
    late_row = (df["season"] == 2020) & (df["valuation_date"] == pd.Timestamp("2021-12-20"))
    assert not train[late_row.to_numpy()].any()


def test_test_season_cannot_be_a_validation_fold():
    with pytest.raises(ValueError):
        list(expanding_window_folds(make_split_frame(), valid_seasons=[TEST_SEASON]))


def test_metrics_on_known_values():
    true = np.log([100.0, 200.0])
    pred = np.log([100.0, 400.0])  # exact, then 2x too high
    m = regression_metrics(true, pred)
    assert m["rmse_log"] == pytest.approx(np.log(2) / np.sqrt(2))
    assert m["mae_eur"] == pytest.approx(100.0)
    assert m["median_ape"] == pytest.approx(0.5)  # absolute % errors are 0 and 1.0
    assert m["within_25"] == pytest.approx(0.5)
    assert m["within_50"] == pytest.approx(0.5)
    expected_r2 = 1 - np.log(2) ** 2 / (2 * (np.log(2) / 2) ** 2)
    assert m["r2_log"] == pytest.approx(expected_r2)
    assert m["n"] == 2


def test_perfect_predictions():
    y = np.log([1e6, 5e6, 2e7])
    m = regression_metrics(y, y)
    assert m["rmse_log"] == 0 and m["r2_log"] == 1 and m["mae_eur"] == pytest.approx(0)
    assert m["within_25"] == 1


def test_metrics_skip_nan_and_zero_values_without_warnings():
    with np.errstate(divide="ignore"):
        true = np.array([np.log(1e6), np.nan, np.log(0.0), np.log(2e6)])  # NaN and log(0) = -inf
    pred = np.array([np.log(1e6), np.log(1e6), np.log(1e6), np.nan])
    with np.errstate(all="raise"):
        m = regression_metrics(true, pred)
        empty = regression_metrics([np.nan], [1.0])
        constant = regression_metrics([1.0, 1.0], [1.0, 2.0])
    assert m["n"] == 1 and m["rmse_log"] == 0
    assert empty["n"] == 0 and np.isnan(empty["rmse_log"])
    assert np.isnan(constant["r2_log"])  # undefined, not a division error


def test_interval_flag():
    df = pd.DataFrame({
        "log_value": [10.0, 10.0, 10.0, 10.0],
        "pred_low": [10.5, 9.0, 9.0, 10.5],
        "pred_high": [11.0, 9.5, 11.0, 11.0],
        "minutes": [2000, 2000, 2000, 500],
    })
    assert interval_flag(df, min_minutes=900).tolist() == ["undervalued", "overvalued", "in range", "not rated"]


def test_residual_ranking_is_reproducible_with_fixed_seed():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"player_id": np.arange(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    df[TARGET] = 15 + df["a"] - 0.5 * df["b"] + rng.normal(scale=0.3, size=n)
    params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50, "min_child_weight": 1, "subsample": 0.7}

    def ranking() -> list[int]:
        model = make_xgb(params, ["a", "b"]).fit(df, df[TARGET])
        scored = df.assign(residual=model.predict(df) - df[TARGET])
        return rank_by_residual(scored)["player_id"].tolist()

    assert ranking() == ranking()


def test_residual_ties_are_broken_by_player_id():
    df = pd.DataFrame({"player_id": [3, 1, 2], "residual": [0.5, 0.5, 0.9]})
    assert rank_by_residual(df)["player_id"].tolist() == [2, 1, 3]

"""Time-based splits and models.

Split design (never random - a random split would let the model learn from the future):
- Tuning: expanding-window CV. Each fold trains on all seasons before the validation
  season: validate 2021/22, 2022/23, 2023/24, 2024/25.
- Final model: refit on 2012/13-2024/25 with the chosen hyperparameters.
- Test: 2025/26, evaluated once at the end, never used for tuning.

Stage 2 scope: the split and the baseline model. Ridge and XGBoost are added in stage 4.

Usage:
    python -m src.train
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from src.data import PLAYER_SEASON_PATH, season_label
from src.evaluate import regression_metrics

TARGET = "log_value"
CV_VALID_SEASONS = [2021, 2022, 2023, 2024]
TEST_SEASON = 2025

AGE_BINS = [0, 21, 24, 27, 30, 33, 100]
AGE_LABELS = ["<21", "21-23", "24-26", "27-29", "30-32", "33+"]


def age_bucket(age: pd.Series) -> pd.Series:
    """Bucket ages into the bands used by the baseline and the error breakdown."""
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False)


def expanding_window_folds(
    seasons: pd.Series, valid_seasons: list[int] = CV_VALID_SEASONS
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (valid_season, train_mask, valid_mask): train on seasons < S, validate on S.

    The test season is never part of any fold.
    """
    for season in valid_seasons:
        if season >= TEST_SEASON:
            raise ValueError(f"validation season {season} overlaps the test season {TEST_SEASON}")
        yield season, (seasons < season).to_numpy(), (seasons == season).to_numpy()


class BaselineModel:
    """Predict the median log value of players with the same position group and age band.

    Falls back to the position median, then the overall median, for combinations unseen in training.
    """

    def fit(self, df: pd.DataFrame, y: pd.Series) -> BaselineModel:
        """Learn medians from the training rows."""
        data = pd.DataFrame({"pos": df["position_group"].to_numpy(), "age": age_bucket(df["age"]).to_numpy(),
                             "y": np.asarray(y)})
        self.cell_medians_ = data.groupby(["pos", "age"], observed=True)["y"].median()
        self.pos_medians_ = data.groupby("pos")["y"].median()
        self.global_median_ = float(data["y"].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Look up the median for each row's (position, age band)."""
        keys = pd.MultiIndex.from_arrays([df["position_group"], age_bucket(df["age"])])
        pred = self.cell_medians_.reindex(keys).to_numpy()
        pos_fallback = self.pos_medians_.reindex(df["position_group"]).to_numpy()
        pred = np.where(np.isnan(pred), pos_fallback, pred)
        return np.where(np.isnan(pred), self.global_median_, pred)


def cross_validate(model_factory, df: pd.DataFrame) -> pd.DataFrame:
    """Run expanding-window CV and return one row of metrics per validation season."""
    rows = []
    for season, train_mask, valid_mask in expanding_window_folds(df["season"]):
        train, valid = df[train_mask], df[valid_mask]
        model = model_factory().fit(train, train[TARGET])
        metrics = regression_metrics(valid[TARGET], model.predict(valid))
        rows.append({"valid_season": season_label(season), "n_train": len(train), "n_valid": len(valid), **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(PLAYER_SEASON_PATH)
    df = df[df["season"] < TEST_SEASON]  # the test season is not even loaded during development

    cv = cross_validate(BaselineModel, df)
    pd.set_option("display.width", 160, "display.float_format", "{:.3f}".format)
    print("Baseline (median log value by position x age band), expanding-window CV:")
    print(cv.to_string(index=False))
    print("\nMean over folds:")
    print(cv.drop(columns=["valid_season", "n_train", "n_valid"]).mean().to_string())


if __name__ == "__main__":
    main()

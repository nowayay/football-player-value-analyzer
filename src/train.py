"""Time-based splits and models.

Split design (never random - a random split would let the model learn from the future):
- Tuning: expanding-window CV. Each fold trains on all seasons before the validation
  season: validate 2021/22, 2022/23, 2023/24, 2024/25.
- Final model: refit on 2012/13-2024/25 with the chosen hyperparameters.
- Test: 2025/26, evaluated once at the end, never used for tuning.

Stage 3 scope: baseline + Ridge (with/without the season trend). XGBoost is added in stage 4.

Usage:
    python -m src.train
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import REPORTS_DIR, TEST_SEASON, season_label
from src.evaluate import regression_metrics
from src.features import FEATURES, FEATURES_PATH, RIDGE_EXTRA, TARGET

CV_VALID_SEASONS = [2021, 2022, 2023, 2024]
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]

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


def make_ridge(alpha: float, columns: list[str] = FEATURES) -> Pipeline:
    """Ridge on the given columns. Median imputation and scaling are fit on the training rows only."""
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), columns),
    ])
    return Pipeline([("prep", preprocess), ("ridge", Ridge(alpha=alpha))])


def cross_validate(model_factory: Callable[[], object], df: pd.DataFrame) -> pd.DataFrame:
    """Run expanding-window CV and return one row of metrics per validation season.

    mean_bias = mean(predicted - actual) log value; negative means the model under-predicts.
    """
    rows = []
    for season, train_mask, valid_mask in expanding_window_folds(df["season"]):
        train, valid = df[train_mask], df[valid_mask]
        model = model_factory().fit(train, train[TARGET])
        pred = model.predict(valid)
        rows.append({"valid_season": season_label(season), "n_train": len(train), "n_valid": len(valid),
                     **regression_metrics(valid[TARGET], pred), "mean_bias": float(np.mean(pred - valid[TARGET]))})
    return pd.DataFrame(rows)


def tune_ridge(df: pd.DataFrame, columns: list[str]) -> tuple[float, pd.DataFrame]:
    """Pick the Ridge alpha with the lowest mean CV RMSE (log). Returns (alpha, CV table for that alpha)."""
    results = {alpha: cross_validate(lambda a=alpha: make_ridge(a, columns), df) for alpha in RIDGE_ALPHAS}
    best = min(results, key=lambda a: results[a]["rmse_log"].mean())
    return best, results[best]


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["season"] < TEST_SEASON]  # the test season is not even loaded during development

    runs = {"baseline": cross_validate(BaselineModel, df)}
    for name, columns in [("ridge", FEATURES), ("ridge+season_index", FEATURES + RIDGE_EXTRA)]:
        alpha, cv = tune_ridge(df, columns)
        runs[f"{name} (alpha={alpha:g})"] = cv

    cv_all = pd.concat({name: cv for name, cv in runs.items()}, names=["model"]).reset_index(level=0)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_all.to_csv(REPORTS_DIR / "cv_results.csv", index=False)

    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print("Expanding-window CV, RMSE (log) and mean bias (pred - actual, log) per validation season:")
    print(cv_all.pivot(index="valid_season", columns="model", values=["rmse_log", "mean_bias"]).to_string())
    print("\nMean over folds:")
    metric_cols = ["rmse_log", "r2_log", "mae_eur", "median_ape", "within_25", "within_50", "mean_bias"]
    summary = cv_all.groupby("model", sort=False)[metric_cols].mean()
    print(summary.to_string(formatters={"mae_eur": "{:,.0f}".format}))
    print(f"\nChosen on mean CV RMSE: {summary['rmse_log'].idxmin()}")


if __name__ == "__main__":
    main()

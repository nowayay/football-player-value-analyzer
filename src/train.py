"""Time-based splits, model selection on expanding-window CV, and final model fitting.

Split design (never random - a random split would let the model learn from the future):
- Tuning: expanding-window CV. Each fold trains on all seasons before the validation
  season (validate 2021/22, 2022/23, 2023/24, 2024/25). Training rows dated on/after the
  first validation valuation are purged, so no training target is newer than a validation target.
- Final models: refit on 2012/13-2024/25 with the hyperparameters chosen on CV.
- Test: 2025/26, evaluated once by `python -m src.evaluate`. This module never loads it.

Models:
1. baseline_median      median log value by position group x age band
2. ridge_age_position   Ridge on age, age^2 and position group only
3. ridge                Ridge on all FEATURES
4. xgb                  XGBoost on all FEATURES
5. xgb_player_only      XGBoost without club/league context (PLAYER_ONLY_FEATURES)
Plus XGBoost quantile models (10th / 90th percentile) for an 80% prediction interval.

Usage:
    python -m src.train
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data import MODELS_DIR, REPORTS_DIR, TEST_SEASON, season_label
from src.evaluate import age_bucket, regression_metrics
from src.features import AGE_POSITION_FEATURES, FEATURES, FEATURES_PATH, PLAYER_ONLY_FEATURES, TARGET

SEED = 42
CV_VALID_SEASONS = [2021, 2022, 2023, 2024]
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
# Modest grid with a fixed number of trees: no early stopping on the validation fold (that would leak).
XGB_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [300, 600],
    "min_child_weight": [1, 10],
    "subsample": [0.7, 1.0],
}
INTERVAL_QUANTILES = (0.1, 0.9)  # 80% prediction interval

def expanding_window_folds(
    df: pd.DataFrame, valid_seasons: list[int] = CV_VALID_SEASONS
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (valid_season, train_mask, valid_mask): train on seasons < S, validate on season S.

    Training rows whose valuation_date is on/after the earliest validation valuation_date are
    dropped, so every training target is strictly older than every validation target.
    The test season is never part of any fold.
    """
    for season in valid_seasons:
        if season >= TEST_SEASON:
            raise ValueError(f"validation season {season} overlaps the test season {TEST_SEASON}")
        valid_mask = (df["season"] == season).to_numpy()
        first_valid_date = df.loc[valid_mask, "valuation_date"].min()
        train_mask = ((df["season"] < season) & (df["valuation_date"] < first_valid_date)).to_numpy()
        yield season, train_mask, valid_mask


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


def make_xgb(params: dict, columns: list[str] = FEATURES, quantile: float | None = None) -> Pipeline:
    """XGBoost on the given columns; NaNs are handled natively. quantile=q fits the q-th quantile instead."""
    objective = {"objective": "reg:quantileerror", "quantile_alpha": quantile} if quantile else {}
    model = XGBRegressor(**params, **objective, colsample_bytree=0.8, tree_method="hist", random_state=SEED,
                         n_jobs=-1)
    select = ColumnTransformer([("cols", "passthrough", columns)])
    return Pipeline([("select", select), ("xgb", model)])


def cross_validate(model_factory: Callable[[], object], df: pd.DataFrame) -> pd.DataFrame:
    """Run expanding-window CV and return one row of metrics per validation season.

    mean_bias = mean(predicted - actual) log value; negative means the model under-predicts.
    train_rmse_log is the in-sample error of the same fold, to spot overfitting.
    """
    rows = []
    for season, train_mask, valid_mask in expanding_window_folds(df):
        train, valid = df[train_mask], df[valid_mask]
        model = model_factory().fit(train, train[TARGET])
        pred = model.predict(valid)
        train_rmse = regression_metrics(train[TARGET], model.predict(train))["rmse_log"]
        rows.append({"valid_season": season_label(season), "n_train": len(train), "n_valid": len(valid),
                     **regression_metrics(valid[TARGET], pred), "mean_bias": float(np.mean(pred - valid[TARGET])),
                     "train_rmse_log": train_rmse})
    return pd.DataFrame(rows)


def tune(factory: Callable[[object], Callable[[], object]], candidates: list, df: pd.DataFrame,
         ) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    """Pick the candidate with the lowest mean CV RMSE (log).

    Returns (best candidate, its per-fold CV table, mean CV RMSE of every candidate).
    """
    results = {i: cross_validate(factory(c), df) for i, c in enumerate(candidates)}
    scores = pd.DataFrame({"candidate": [str(c) for c in candidates],
                           "cv_rmse_log": [results[i]["rmse_log"].mean() for i in range(len(candidates))]})
    best = int(scores["cv_rmse_log"].idxmin())
    return candidates[best], results[best], scores.sort_values("cv_rmse_log")


def interval_coverage_cv(params: dict, columns: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Per CV fold: share of actual values inside the [q10, q90] interval and the interval width."""
    rows = []
    for season, train_mask, valid_mask in expanding_window_folds(df):
        train, valid = df[train_mask], df[valid_mask]
        bounds = [make_xgb(params, columns, q).fit(train, train[TARGET]).predict(valid) for q in INTERVAL_QUANTILES]
        low, high = np.minimum(*bounds), np.maximum(*bounds)  # guard against quantile crossing
        y = valid[TARGET].to_numpy()
        rows.append({"valid_season": season_label(season), "coverage": float(np.mean((y >= low) & (y <= high))),
                     "below_interval": float(np.mean(y < low)), "above_interval": float(np.mean(y > high)),
                     "median_width_ratio": float(np.median(np.exp(high - low)))})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["season"] < TEST_SEASON].reset_index(drop=True)  # the test season is never loaded here
    grid = list(ParameterGrid(XGB_GRID))

    print("Tuning on expanding-window CV (validation seasons "
          f"{', '.join(season_label(s) for s in CV_VALID_SEASONS)}) ...")
    chosen: dict[str, object] = {}
    cv_tables: dict[str, pd.DataFrame] = {"baseline_median": cross_validate(BaselineModel, df)}
    grid_tables = []
    for name, columns in [("ridge_age_position", AGE_POSITION_FEATURES), ("ridge", FEATURES)]:
        alpha, cv_tables[name], scores = tune(lambda a, c=columns: (lambda: make_ridge(a, c)), RIDGE_ALPHAS, df)
        chosen[name] = {"alpha": alpha}
        grid_tables.append(scores.assign(model=name))
    for name, columns in [("xgb", FEATURES), ("xgb_player_only", PLAYER_ONLY_FEATURES)]:
        params, cv_tables[name], scores = tune(lambda p, c=columns: (lambda: make_xgb(p, c)), grid, df)
        chosen[name] = params
        grid_tables.append(scores.assign(model=name))
        print(f"  {name}: best of {len(grid)} configs -> {params}")

    cv_all = pd.concat(cv_tables, names=["model"]).reset_index(level=0)
    metric_cols = ["rmse_log", "r2_log", "mae_eur", "median_ape", "within_25", "within_50", "mean_bias",
                   "train_rmse_log"]
    summary = cv_all.groupby("model", sort=False)[metric_cols].mean()
    summary["overfit_gap"] = summary["rmse_log"] - summary["train_rmse_log"]
    main_model = summary.loc[["ridge", "xgb"], "rmse_log"].idxmin()
    coverage = interval_coverage_cv(chosen["xgb"], FEATURES, df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_all.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    summary.to_csv(REPORTS_DIR / "cv_summary.csv")
    pd.concat(grid_tables).to_csv(REPORTS_DIR / "cv_grid_search.csv", index=False)
    coverage.to_csv(REPORTS_DIR / "cv_interval_coverage.csv", index=False)

    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    print("\nCV per fold:")
    print(cv_all.drop(columns=["n_train", "n_valid"]).to_string(index=False, formatters={"mae_eur": "{:,.0f}".format}))
    print("\nCV mean over folds:")
    print(summary.to_string(formatters={"mae_eur": "{:,.0f}".format}))
    print(f"\nMain model (lowest CV RMSE of ridge vs xgb): {main_model}")
    print("\n80% interval (XGBoost q10-q90, main-model hyperparameters), CV coverage per fold:")
    print(coverage.to_string(index=False))

    # Refit every model on all development seasons and save it.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final = {
        "baseline_median": BaselineModel(),
        "ridge_age_position": make_ridge(chosen["ridge_age_position"]["alpha"], AGE_POSITION_FEATURES),
        "ridge": make_ridge(chosen["ridge"]["alpha"], FEATURES),
        "xgb": make_xgb(chosen["xgb"], FEATURES),
        "xgb_player_only": make_xgb(chosen["xgb_player_only"], PLAYER_ONLY_FEATURES),
        "quantile_q10": make_xgb(chosen["xgb"], FEATURES, INTERVAL_QUANTILES[0]),
        "quantile_q90": make_xgb(chosen["xgb"], FEATURES, INTERVAL_QUANTILES[1]),
    }
    for name, model in final.items():
        model.fit(df, df[TARGET])
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    config = {"main_model": main_model, "trained_on_seasons": [int(df["season"].min()), int(df["season"].max())],
              "test_season": TEST_SEASON, "hyperparameters": chosen, "interval_quantiles": INTERVAL_QUANTILES}
    (MODELS_DIR / "model_config.json").write_text(json.dumps(config, indent=2))
    print(f"\nRefit on {season_label(df['season'].min())}-{season_label(df['season'].max())} "
          f"({len(df):,} rows) and saved {len(final)} models to models/")


if __name__ == "__main__":
    main()

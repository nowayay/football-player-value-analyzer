"""Time-based splits, model selection on expanding-window CV, and final model fitting.

Split design (never random - a random split would let the model learn from the future):
- Tuning: expanding-window CV. Each fold trains on all seasons before the validation
  season (validate 2021/22, 2022/23, 2023/24, 2024/25). Training rows dated on/after the
  first validation valuation are purged, so no training target is newer than a validation target.
- Final models: refit on 2012/13-2024/25 with the hyperparameters chosen on CV.
- Test: 2025/26, evaluated once by `python -m src.evaluate`. This module never loads it.

Models (labels in src.evaluate.MODEL_LABELS):
1. baseline_median      median log value by position group x age band
2. ridge_age_position   Ridge on age, age^2 and position group only
3. ridge                Ridge on all FEATURES
4. xgb                  XGBoost on all FEATURES (main model)
5. xgb_player_market    XGBoost, player-only + league market level (PLAYER_MARKET_FEATURES)

Prediction intervals (both XGBoost variants): quantile models at 10% / 90% give a raw 80% interval,
which covered only ~74-75% in CV. A conformal correction widens both sides by a fixed log amount,
learned from out-of-fold (OOF) errors: when validating fold k, only folds < k are used; the final
correction for the test season uses all four folds and is saved to models/ before evaluation.

Usage:
    python -m src.train
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.data import MODELS_DIR, REPORTS_DIR, TEST_SEASON, season_label
from src.evaluate import MODEL_LABELS, regression_metrics
from src.features import AGE_POSITION_FEATURES, FEATURES, FEATURES_PATH, PLAYER_MARKET_FEATURES, TARGET
from src.models import BaselineModel, make_ridge, make_xgb

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
INTERVAL_QUANTILES = (0.1, 0.9)
INTERVAL_COVERAGE = 0.8
CONFORMAL_PATH = MODELS_DIR / "conformal_correction.json"


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


def conformity_scores(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """How far each actual value falls outside [low, high] in log units (negative = inside)."""
    return np.maximum(low - y, y - high)


def conformal_correction(scores: np.ndarray, coverage: float = INTERVAL_COVERAGE) -> float:
    """Per-side widening (log units) so that `coverage` of the given scores would fall inside.

    Uses the finite-sample quantile level ceil((n + 1) * coverage) / n. Never negative: the
    correction only ever widens the raw interval.
    """
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return max(0.0, float(np.quantile(scores, level, method="higher")))


def apply_correction(low: np.ndarray, high: np.ndarray, correction: float) -> tuple[np.ndarray, np.ndarray]:
    """Fix quantile crossing (sort the bounds), then widen each side by the correction."""
    low, high = np.minimum(low, high), np.maximum(low, high)
    correction = max(float(correction), 0.0)
    return low - correction, high + correction


def forward_corrections(fold_scores: dict[int, np.ndarray]) -> dict[int, float]:
    """Correction for each fold from the OOF scores of EARLIER folds only (NaN for the first fold)."""
    corrections = {}
    for season in sorted(fold_scores):
        past = [scores for s, scores in fold_scores.items() if s < season]
        corrections[season] = conformal_correction(np.concatenate(past)) if past else float("nan")
    return corrections


def interval_cv(params: dict, columns: list[str], df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Raw and conformally corrected 80% interval coverage per CV fold, plus the final correction.

    Returns (per-fold table, correction fit on the OOF scores of all folds).
    """
    folds = {}
    for season, train_mask, valid_mask in expanding_window_folds(df):
        train, valid = df[train_mask], df[valid_mask]
        low, high = (make_xgb(params, columns, q).fit(train, train[TARGET]).predict(valid) for q in INTERVAL_QUANTILES)
        low, high = apply_correction(low, high, 0.0)  # sort bounds only
        folds[season] = (valid[TARGET].to_numpy(), low, high)
    fold_scores = {s: conformity_scores(*f) for s, f in folds.items()}
    corrections = forward_corrections(fold_scores)

    rows = []
    for season, (y, low, high) in folds.items():
        c = corrections[season]
        row = {"valid_season": season_label(season), "coverage_raw": float(np.mean((y >= low) & (y <= high))),
               "width_ratio_raw": float(np.median(np.exp(high - low))), "conformal_correction": c,
               "coverage_conformal": float("nan"), "width_ratio_conformal": float("nan")}
        if not np.isnan(c):  # the first fold has no earlier fold to learn a correction from
            lo_c, hi_c = apply_correction(low, high, c)
            row["coverage_conformal"] = float(np.mean((y >= lo_c) & (y <= hi_c)))
            row["width_ratio_conformal"] = float(np.median(np.exp(hi_c - lo_c)))
        rows.append(row)
    final = conformal_correction(np.concatenate(list(fold_scores.values())))
    return pd.DataFrame(rows), final


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["season"] < TEST_SEASON].reset_index(drop=True)  # the test season is never loaded here
    grid = list(ParameterGrid(XGB_GRID))
    xgb_variants = {"xgb": FEATURES, "xgb_player_market": PLAYER_MARKET_FEATURES}

    print("Tuning on expanding-window CV (validation seasons "
          f"{', '.join(season_label(s) for s in CV_VALID_SEASONS)}) ...")
    chosen: dict[str, dict] = {}
    cv_tables: dict[str, pd.DataFrame] = {"baseline_median": cross_validate(BaselineModel, df)}
    grid_tables = []
    for name, columns in [("ridge_age_position", AGE_POSITION_FEATURES), ("ridge", FEATURES)]:
        alpha, cv_tables[name], scores = tune(lambda a, c=columns: (lambda: make_ridge(a, c)), RIDGE_ALPHAS, df)
        chosen[name] = {"alpha": alpha}
        grid_tables.append(scores.assign(model=name))
    for name, columns in xgb_variants.items():
        params, cv_tables[name], scores = tune(lambda p, c=columns: (lambda: make_xgb(p, c)), grid, df)
        chosen[name] = params
        grid_tables.append(scores.assign(model=name))
        print(f"  {name}: best of {len(grid)} configs -> {params}")

    # Intervals: raw quantile coverage, forward-validated conformal coverage, and the frozen final correction.
    interval_tables, final_corrections = {}, {}
    for name, columns in xgb_variants.items():
        interval_tables[name], final_corrections[name] = interval_cv(chosen[name], columns, df)
    intervals = pd.concat(interval_tables, names=["model"]).reset_index(level=0)

    cv_all = pd.concat(cv_tables, names=["model"]).reset_index(level=0)
    cv_all = cv_all.merge(intervals, on=["model", "valid_season"], how="left")
    metric_cols = ["rmse_log", "r2_log", "mae_eur", "median_ape", "within_25", "within_50", "mean_bias",
                   "train_rmse_log", "coverage_raw", "coverage_conformal"]
    summary = cv_all.groupby("model", sort=False)[metric_cols].mean()
    summary["overfit_gap"] = summary["rmse_log"] - summary["train_rmse_log"]
    summary.insert(0, "label", summary.index.map(MODEL_LABELS))
    main_model = summary.loc[["ridge", "xgb"], "rmse_log"].idxmin()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_all.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    summary.to_csv(REPORTS_DIR / "cv_summary.csv")
    pd.concat(grid_tables).to_csv(REPORTS_DIR / "cv_grid_search.csv", index=False)

    pd.set_option("display.width", 240, "display.float_format", "{:.3f}".format)
    print("\nCV per fold:")
    print(cv_all.drop(columns=["n_train", "n_valid"]).to_string(index=False, formatters={"mae_eur": "{:,.0f}".format}))
    print("\nCV mean over folds (coverage_conformal averages the 3 folds that have earlier folds):")
    print(summary.drop(columns="label").to_string(formatters={"mae_eur": "{:,.0f}".format}))
    print(f"\nMain model (lowest CV RMSE of ridge vs xgb): {main_model}")

    # Refit every model on all development seasons and save it, plus the frozen conformal corrections.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final = {
        "baseline_median": BaselineModel(),
        "ridge_age_position": make_ridge(chosen["ridge_age_position"]["alpha"], AGE_POSITION_FEATURES),
        "ridge": make_ridge(chosen["ridge"]["alpha"], FEATURES),
    }
    for name, columns in xgb_variants.items():
        final[name] = make_xgb(chosen[name], columns)
        for q in INTERVAL_QUANTILES:
            final[f"{name}_q{round(q * 100)}"] = make_xgb(chosen[name], columns, q)
    for name, model in final.items():
        model.fit(df, df[TARGET])
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    conformal = {name: {"correction_log": c, "target_coverage": INTERVAL_COVERAGE,
                        "fit_on": "OOF conformity scores of CV folds " + ", ".join(season_label(s) for s in CV_VALID_SEASONS)}
                 for name, c in final_corrections.items()}
    CONFORMAL_PATH.write_text(json.dumps(conformal, indent=2))
    config = {"main_model": main_model, "trained_on_seasons": [int(df["season"].min()), int(df["season"].max())],
              "test_season": TEST_SEASON, "hyperparameters": chosen, "interval_quantiles": INTERVAL_QUANTILES}
    (MODELS_DIR / "model_config.json").write_text(json.dumps(config, indent=2))
    print(f"\nFrozen conformal corrections (log units per side): "
          + ", ".join(f"{n} {c:+.3f} (x{np.exp(c):.2f})" for n, c in final_corrections.items()))
    print(f"Refit on {season_label(df['season'].min())}-{season_label(df['season'].max())} "
          f"({len(df):,} rows) and saved {len(final)} models + {CONFORMAL_PATH.name} to models/")


if __name__ == "__main__":
    main()

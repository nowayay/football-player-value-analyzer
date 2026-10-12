"""Backtest: do players flagged "undervalued" gain more market value over the next 12 months?

For each validation season S = 2021/22 ... 2024/25:
- Flags come from the out-of-fold models of that CV fold (main XGBoost + 10%/90% quantile models,
  trained only on seasons < S with the CV-chosen hyperparameters). The interval correction for S
  is learned from earlier folds only; 2021/22 has no earlier fold and uses the raw interval.
  Caveat: the hyperparameters themselves were chosen on all four folds.
- Flag rule: src.evaluate.add_gaps_and_flags (season-centred gap; raw gap as sensitivity).
- Outcome: log change from the season-S target valuation to the player's valuation closest to
  365 days later (330-400 days), from player_valuations in ANY league, so players who left the
  top-5 are kept. The 2024/25 outcome uses 2025/26 valuations: outcome only, never a feature or
  a tuning input.
- Sample: rated players (>= 900 minutes) with an outcome valuation.

Caveat on interpretation: market values are noisy, so a low value today tends to rise later
(mean reversion). A positive "undervalued" effect can partly be that, which is why the
regression controls for the current log value.

Robustness section (pooled, centred flags): the matched comparison and the regression are
re-run (a) on ages 21-32 only, (b) on market values >= EUR 2m only, (c) on both, which removes
the veteran / very-low-value players that dominate the top of the "undervalued" list, and
(i)/(ii) with missing outcomes set to the 10th / 25th percentile of the same season's observed
changes, because undervalued-flagged players lack an outcome more often (retirements, moves out
of tracked leagues). All CIs resample whole players. Regressions use age + age^2, not age bands.

Usage:
    python -m src.backtest
"""

from __future__ import annotations

import json
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import MODELS_DIR, REPORTS_DIR, TEST_SEASON, load_table, season_label
from src.evaluate import FIGURES_DIR, FLAG_MIN_MINUTES, add_gaps_and_flags
from src.features import FEATURES, FEATURES_PATH, TARGET
from src.models import make_xgb
from src.plot_style import SERIES, TEXT_MUTED, apply_style
from src.train import CV_VALID_SEASONS, INTERVAL_QUANTILES, apply_correction, conformity_scores, \
    expanding_window_folds, forward_corrections

SEED = 42
N_BOOT = 1000
OUTCOME_WINDOW_DAYS = (330, 400)
OUTCOME_TARGET_DAYS = 365
AGE_BANDS = {"bins": [0, 22, 26, 30, 100], "labels": ["<=21", "22-25", "26-29", "30+"]}
GROUPS = ["undervalued", "in range", "overvalued"]


def backtest_age_band(age: pd.Series) -> pd.Series:
    """Age bands used as controls in the backtest: <=21, 22-25, 26-29, 30+."""
    return pd.cut(age, bins=AGE_BANDS["bins"], labels=AGE_BANDS["labels"], right=False)


def out_of_fold_predictions(
    df: pd.DataFrame,
    make_point: Callable[[], object],
    make_low: Callable[[], object],
    make_high: Callable[[], object],
    valid_seasons: list[int] = CV_VALID_SEASONS,
) -> pd.DataFrame:
    """Predictions and 80% intervals for each validation season from models trained on earlier seasons only.

    Adds pred, pred_low, pred_high (log scale), interval_correction and trained_through
    (the latest season in that fold's training data, always < the row's season).
    """
    folds, scores = [], {}
    for season, train_mask, valid_mask in expanding_window_folds(df, valid_seasons):
        train, valid = df[train_mask], df[valid_mask]
        pred = make_point().fit(train, train[TARGET]).predict(valid)
        low, high = apply_correction(make_low().fit(train, train[TARGET]).predict(valid),
                                     make_high().fit(train, train[TARGET]).predict(valid), 0.0)
        scores[season] = conformity_scores(valid[TARGET].to_numpy(), low, high)
        folds.append(valid.assign(pred=pred, raw_low=low, raw_high=high, trained_through=int(train["season"].max())))
    out = pd.concat(folds, ignore_index=True)
    correction = out["season"].map(forward_corrections(scores)).fillna(0.0)  # first fold: raw interval
    return out.assign(interval_correction=correction, pred_low=out["raw_low"] - correction,
                      pred_high=out["raw_high"] + correction)


def next_year_outcome(rows: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    """Attach outcome_date and delta_log_value: change to the valuation closest to 365 days later.

    Only valuations 330-400 days after the row's valuation_date (any league) are considered;
    rows without one get NaN.
    """
    vals = valuations.loc[valuations["market_value_in_eur"] > 0, ["player_id", "date", "market_value_in_eur"]]
    cand = rows[["player_id", "season", "valuation_date", TARGET]].merge(vals, on="player_id")
    days = (cand["date"] - cand["valuation_date"]).dt.days
    lo, hi = OUTCOME_WINDOW_DAYS
    cand = cand[(days >= lo) & (days <= hi)].assign(distance=(days - OUTCOME_TARGET_DAYS).abs())
    best = cand.sort_values(["distance", "date"], kind="mergesort").drop_duplicates(["player_id", "season"])
    best = best.assign(delta_log_value=np.log(best["market_value_in_eur"]) - best[TARGET])
    best = best.rename(columns={"date": "outcome_date"})[["player_id", "season", "outcome_date", "delta_log_value"]]
    return rows.merge(best, on=["player_id", "season"], how="left", validate="one_to_one")


def _cluster_bootstrap_indices(players: np.ndarray, rng: np.random.Generator, n_boot: int):
    """Yield row indices for bootstrap samples that resample whole players (a player appears in several seasons)."""
    codes, uniques = pd.factorize(players)
    rows_by_player = pd.Series(np.arange(len(players))).groupby(codes).apply(np.array).to_numpy()
    for _ in range(n_boot):
        picked = rng.integers(0, len(uniques), size=len(uniques))
        yield np.concatenate(rows_by_player[picked])


def ols_with_bootstrap(df: pd.DataFrame, flag_col: str, season_effects: bool, n_boot: int = N_BOOT,
                       seed: int = SEED) -> pd.DataFrame:
    """OLS of delta_log_value on flag dummies + age + age^2 + position + log value (+ season fixed effects).

    Age is centred at 27 before squaring, so age and age^2 are not almost collinear.
    Returns coefficient and 95% CI (player-cluster bootstrap) for the flag dummies and log value.
    """
    age_c = df["age"] - 27
    design = pd.DataFrame({
        "intercept": 1.0,
        "flag_under": (df[flag_col] == "undervalued").astype(float),
        "flag_over": (df[flag_col] == "overvalued").astype(float),
        "log_value_now": df[TARGET] - df[TARGET].mean(),
        "age_c": age_c,
        "age_c_sq": age_c**2,
    }, index=df.index)
    controls = [pd.get_dummies(df["position_group"], prefix="pos", drop_first=True, dtype=float)]
    if season_effects:
        controls.append(pd.get_dummies(df["season"], prefix="season", drop_first=True, dtype=float))
    X = pd.concat([design, *controls], axis=1).to_numpy()
    y = df["delta_log_value"].to_numpy()
    names = ["flag_under", "flag_over", "log_value_now"]
    keep = [list(pd.concat([design, *controls], axis=1).columns).index(n) for n in names]

    coef = np.linalg.lstsq(X, y, rcond=None)[0][keep]
    rng = np.random.default_rng(seed)
    boot = np.array([np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][keep]
                     for idx in _cluster_bootstrap_indices(df["player_id"].to_numpy(), rng, n_boot)])
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return pd.DataFrame({"coef": coef, "ci_low": low, "ci_high": high}, index=names)


def _matched_excess_point(flagged: np.ndarray, delta: np.ndarray, cell: np.ndarray, n_cells: int) -> tuple:
    """Excess of flagged rows over the mean of non-flagged rows in the same cell (vectorised).

    Returns (n_flagged_matched, mean_flagged, mean_control, excess)."""
    sums = np.bincount(cell[~flagged], weights=delta[~flagged], minlength=n_cells)
    counts = np.bincount(cell[~flagged], minlength=n_cells)
    has_control = counts[cell[flagged]] > 0
    control = (sums / np.maximum(counts, 1))[cell[flagged]][has_control]
    own = delta[flagged][has_control]
    if len(own) == 0:
        return 0, np.nan, np.nan, np.nan
    return len(own), own.mean(), control.mean(), (own - control).mean()


def matched_excess(df: pd.DataFrame, flag_col: str, flag_value: str, n_boot: int = N_BOOT,
                   seed: int = SEED) -> dict[str, float]:
    """Mean delta of flagged players minus the mean delta of all other players in the same
    season x position group x age band cell. 95% CI from a whole-player bootstrap (control
    means are recomputed in every bootstrap sample)."""
    cell, _ = pd.factorize(pd.MultiIndex.from_frame(df[["season", "position_group", "age_band"]].astype(str)))
    flagged = (df[flag_col] == flag_value).to_numpy()
    delta = df["delta_log_value"].to_numpy()
    n_cells = cell.max() + 1 if len(cell) else 0
    n, mean_flagged, mean_control, excess = _matched_excess_point(flagged, delta, cell, n_cells)
    rng = np.random.default_rng(seed)
    boot = [_matched_excess_point(flagged[idx], delta[idx], cell[idx], n_cells)[3]
            for idx in _cluster_bootstrap_indices(df["player_id"].to_numpy(), rng, n_boot)]
    low, high = np.nanpercentile(boot, [2.5, 97.5]) if n else (np.nan, np.nan)
    return {"n_flagged": n, "mean_flagged": mean_flagged, "mean_control": mean_control, "excess": excess,
            "ci_low": low, "ci_high": high}


def impute_missing_outcomes(df: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """Bound for missing outcomes: fill them with the given quantile of the same season's observed changes."""
    fill = df.groupby("season")["delta_log_value"].transform(lambda s: s.quantile(quantile))
    return df.assign(delta_log_value=df["delta_log_value"].fillna(fill))


def robustness_table(rated: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Pooled undervalued / overvalued effects (centred flags) under alternative samples and bounds.

    Each row: matched excess and regression coefficient (season FE, age + age^2, position, log value),
    both with 95% whole-player bootstrap CIs.
    """
    observed = rated.dropna(subset=["delta_log_value"])
    core_age = observed["age"].between(21, 33, inclusive="left")  # 21-32
    min_value = observed["market_value_eur"] >= 2e6
    specs = {
        "main": observed,
        "(a) age 21-32": observed[core_age],
        "(b) market value >= EUR 2m": observed[min_value],
        "(c) age 21-32 and value >= EUR 2m": observed[core_age & min_value],
        "(i) missing outcome = season p10": impute_missing_outcomes(rated, 0.10),
        "(ii) missing outcome = season p25": impute_missing_outcomes(rated, 0.25),
    }
    rows = []
    for spec, part in specs.items():
        reg = ols_with_bootstrap(part, "flag", season_effects=True, n_boot=n_boot)
        for value, term in [("undervalued", "flag_under"), ("overvalued", "flag_over")]:
            m = matched_excess(part, "flag", value, n_boot=n_boot)
            rows.append({"spec": spec, "group": value, "n_rows": len(part), "n_flagged": m["n_flagged"],
                         "matched_excess": m["excess"], "matched_ci_low": m["ci_low"], "matched_ci_high": m["ci_high"],
                         "reg_coef": reg.loc[term, "coef"], "reg_ci_low": reg.loc[term, "ci_low"],
                         "reg_ci_high": reg.loc[term, "ci_high"]})
    return pd.DataFrame(rows)


def spearman_with_bootstrap(df: pd.DataFrame, gap_col: str, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Spearman correlation between a gap column and delta_log_value, with a player-cluster bootstrap CI."""
    rho = df[gap_col].corr(df["delta_log_value"], method="spearman")
    rng = np.random.default_rng(seed)
    gap, delta = df[gap_col].to_numpy(), df["delta_log_value"].to_numpy()
    boot = [pd.Series(gap[idx]).corr(pd.Series(delta[idx]), method="spearman")
            for idx in _cluster_bootstrap_indices(df["player_id"].to_numpy(), rng, n_boot)]
    return {"n": len(df), "spearman": rho, "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)}


def group_means(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    """Mean 12-month log value change per season and flag group, with a normal-approximation 95% CI."""
    g = df.groupby(["season", flag_col])["delta_log_value"].agg(["size", "mean", "std"])
    g["ci95"] = 1.96 * g["std"] / np.sqrt(g["size"])
    return g.rename(columns={"size": "n"}).drop(columns="std")


def plot_group_means(means: pd.DataFrame, path) -> None:
    """Dot plot with 95% error bars: mean 12-month log value change by flag group and season."""
    colors = {"undervalued": SERIES[0], "in range": TEXT_MUTED, "overvalued": SERIES[1]}
    seasons = sorted(means.index.get_level_values("season").unique())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, group in enumerate(GROUPS):
        sub = means.xs(group, level=1).reindex(seasons)
        x = np.arange(len(seasons)) + (i - 1) * 0.22
        ax.errorbar(x, sub["mean"], yerr=sub["ci95"], fmt="o", color=colors[group], markersize=6, capsize=3,
                    linewidth=1.5, label=f"{group}")
    ax.axhline(0, color=TEXT_MUTED, linewidth=1, linestyle="--")
    ax.set(xticks=range(len(seasons)), xticklabels=[season_label(s) for s in seasons],
           title="Next-12-month change in market value by flag (season-centred flags, 900+ min)",
           xlabel="season of the flag", ylabel="mean log change (0.1 = about +10%)")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    apply_style()
    params = json.loads((MODELS_DIR / "model_config.json").read_text())["hyperparameters"]["xgb"]
    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["season"] < TEST_SEASON].reset_index(drop=True)  # test-season rows are never flagged here

    print("Fitting out-of-fold models (one per validation season, trained on earlier seasons only) ...")
    oof = out_of_fold_predictions(df, lambda: make_xgb(params, FEATURES),
                                  lambda: make_xgb(params, FEATURES, INTERVAL_QUANTILES[0]),
                                  lambda: make_xgb(params, FEATURES, INTERVAL_QUANTILES[1]))
    flagged = next_year_outcome(add_gaps_and_flags(oof), load_table("player_valuations"))
    flagged["age_band"] = backtest_age_band(flagged["age"])
    rated = flagged[flagged["minutes"] >= FLAG_MIN_MINUTES]

    missing = rated.assign(no_outcome=rated["delta_log_value"].isna()).pivot_table(
        index="season", columns="flag", values="no_outcome", aggfunc=["size", "sum"], fill_value=0)
    missing.columns = [f"{'n' if a == 'size' else 'no_outcome'}_{b}" for a, b in missing.columns]
    sample = rated.dropna(subset=["delta_log_value"])

    means = {"adjusted": group_means(sample, "flag"), "raw": group_means(sample, "flag_raw")}
    excess_rows, reg_rows, rho_rows = [], [], []
    for variant, flag_col, gap_col in [("adjusted", "flag", "adjusted_gap"), ("raw", "flag_raw", "raw_gap")]:
        for scope, part in [("pooled", sample), *[(season_label(s), sample[sample["season"] == s]) for s in CV_VALID_SEASONS]]:
            for value in ["undervalued", "overvalued"]:
                excess_rows.append({"variant": variant, "scope": scope, "group": value,
                                    **matched_excess(part, flag_col, value)})
            reg = ols_with_bootstrap(part, flag_col, season_effects=scope == "pooled")
            reg_rows.append(reg.assign(variant=variant, scope=scope, n=len(part)).reset_index(names="term"))
            rho_rows.append({"variant": variant, "scope": scope, **spearman_with_bootstrap(part, gap_col)})
    excess = pd.DataFrame(excess_rows)
    regression = pd.concat(reg_rows, ignore_index=True)[["variant", "scope", "term", "n", "coef", "ci_low", "ci_high"]]
    spearman = pd.DataFrame(rho_rows)
    robustness = robustness_table(rated)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(means, names=["variant"]).to_csv(REPORTS_DIR / "backtest_group_means.csv")
    excess.to_csv(REPORTS_DIR / "backtest_matched_excess.csv", index=False)
    regression.to_csv(REPORTS_DIR / "backtest_regression.csv", index=False)
    spearman.to_csv(REPORTS_DIR / "backtest_spearman.csv", index=False)
    missing.to_csv(REPORTS_DIR / "backtest_missing_outcomes.csv")
    robustness.to_csv(REPORTS_DIR / "backtest_robustness.csv", index=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_group_means(means["adjusted"], FIGURES_DIR / "backtest_value_change.png")

    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    print(f"\nRated players (>= {FLAG_MIN_MINUTES} min) and missing outcomes (no valuation 330-400 days later):")
    print(missing.rename(index=season_label).to_string())
    print(f"\nBacktest sample: {len(sample):,} player-seasons with an outcome")
    for variant in ["adjusted", "raw"]:
        print(f"\nMean next-12-month log value change by flag ({variant} flags):")
        print(means[variant].unstack(1).rename(index=season_label).to_string())
    print("\nMatched comparison (flagged minus others in the same season x position x age band):")
    print(excess.to_string(index=False))
    print("\nRegression: delta_log_value ~ flags + age + age^2 + position + log value (+ season FE when pooled),"
          f" 95% CI from {N_BOOT} player-cluster bootstrap samples:")
    print(regression.to_string(index=False))
    print("\nSpearman(gap, delta_log_value), player-cluster bootstrap 95% CI:")
    print(spearman.to_string(index=False))
    print("\nRobustness (pooled, centred flags; 95% whole-player bootstrap CIs):")
    print(robustness.to_string(index=False))
    print("\nSaved reports/backtest_*.csv (incl. backtest_robustness.csv) and reports/figures/backtest_value_change.png")


if __name__ == "__main__":
    main()

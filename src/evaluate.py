"""Evaluation: metrics, error breakdowns, failure analysis and residual ranking.

Residual = log(predicted) - log(market value). Positive = the model expects more than the
market pays ("undervalued" candidate); negative = the market pays a premium ("overvalued").

`python -m src.evaluate` scores the saved models on the held-out test season (2025/26).
The models and the conformal interval correction were chosen and frozen on CV only; the test
result is reported, never used for tuning. The first run writes reports/.test_evaluated and
later runs refuse unless --force-rerun is given (which is then logged in test_metrics.csv).

Usage:
    python -m src.evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import MIN_MINUTES, MODELS_DIR, REPORTS_DIR, TEST_SEASON, TOP5_LEAGUES, season_label
from src.features import FEATURES_PATH
from src.plot_style import SERIES, TEXT_MUTED, apply_style

FIGURES_DIR = REPORTS_DIR / "figures"
FLAG_MIN_MINUTES = 900  # only flag players with 10+ full games of evidence
TEST_MARKER = REPORTS_DIR / ".test_evaluated"
MODEL_LABELS = {
    "baseline_median": "Median baseline (position x age band)",
    "ridge_age_position": "Ridge: age + position",
    "ridge": "Ridge: all features",
    "xgb": "XGBoost: all features (main)",
    "xgb_player_market": "XGBoost: player-only + league market level",
}
INTERVAL_MODELS = ["xgb", "xgb_player_market"]

AGE_BINS = [0, 21, 24, 27, 30, 33, 100]
AGE_LABELS = ["<21", "21-23", "24-26", "27-29", "30-32", "33+"]


def age_bucket(age: pd.Series) -> pd.Series:
    """Bucket ages into the bands used by the baseline and the error breakdown."""
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False)


def regression_metrics(y_true_log: np.ndarray | pd.Series, y_pred_log: np.ndarray | pd.Series) -> dict[str, float]:
    """Compute error metrics for predictions of log(market value in EUR).

    - rmse_log, r2_log: on the log scale the model is trained on.
    - mae_eur: average absolute error in euros (dominated by expensive players).
    - median_ape: median absolute % error in EUR; robust to a few huge misses.
    - within_25 / within_50: share of players whose prediction is within 25% / 50% of the market value.
    Pairs with a NaN or infinite value (e.g. log of a zero value) are skipped; n counts the pairs used.
    """
    t, p = np.asarray(y_true_log, dtype=float), np.asarray(y_pred_log, dtype=float)
    keep = np.isfinite(t) & np.isfinite(p)
    t, p = t[keep], p[keep]
    if len(t) == 0:
        return {k: float("nan") for k in ["rmse_log", "r2_log", "mae_eur", "median_ape", "within_25", "within_50"]} | {"n": 0}
    true_eur, pred_eur = np.exp(t), np.exp(p)
    ape = np.abs(pred_eur - true_eur) / true_eur
    ss_tot = np.sum((t - t.mean()) ** 2)
    return {
        "rmse_log": float(np.sqrt(np.mean((t - p) ** 2))),
        "r2_log": float(1 - np.sum((t - p) ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "mae_eur": float(np.mean(np.abs(pred_eur - true_eur))),
        "median_ape": float(np.median(ape)),
        "within_25": float(np.mean(ape <= 0.25)),
        "within_50": float(np.mean(ape <= 0.50)),
        "n": int(len(t)),
    }


def interval_flag(df: pd.DataFrame, min_minutes: int = FLAG_MIN_MINUTES) -> pd.Series:
    """Label each row by where the market value sits relative to the 80% prediction interval.

    Needs columns log_value, pred_low, pred_high, minutes. Players under min_minutes are 'not rated'.
    """
    flag = np.select(
        [df["minutes"] < min_minutes, df["log_value"] < df["pred_low"], df["log_value"] > df["pred_high"]],
        ["not rated", "undervalued", "overvalued"],
        default="in range",
    )
    return pd.Series(flag, index=df.index, name="flag")


def add_gaps_and_flags(df: pd.DataFrame, min_minutes: int = FLAG_MIN_MINUTES) -> pd.DataFrame:
    """Add raw and season-centred gaps plus over/undervalued flags (the one flag rule used everywhere).

    Needs columns season, minutes, log_value, pred, pred_low, pred_high (all log scale).

    - raw_gap = log(pred) - log(market). Positive = the model is above the market.
    - season_shift = median raw_gap of the season's rows with >= MIN_MINUTES (450) minutes.
    - adjusted_gap = raw_gap - season_shift, so its median is 0 within every season.
    - adjusted_low / adjusted_high: the 80% interval shifted by the same constant.
    - flag: undervalued/overvalued only if the market value falls outside the shifted interval
      and minutes >= min_minutes.
    - flag_raw: the same rule without centring (sensitivity variant).

    Why centre: the model under-predicts a whole season when market prices rise faster than the
    lagged league market level can follow (2025/26 test bias was -0.15 log, about -14%). That shift
    is common to everyone, so raw flags label most players "overvalued". Centring compares each
    player with the rest of the same season. It uses only that season's cross-section of
    predictions and market values (known at the valuation date), never future information.
    """
    out = df.copy()
    out["raw_gap"] = out["pred"] - out["log_value"]
    eligible = out["minutes"] >= MIN_MINUTES
    shift = out[eligible].groupby("season")["raw_gap"].median()
    out["season_shift"] = out["season"].map(shift)
    out["adjusted_gap"] = out["raw_gap"] - out["season_shift"]
    out["adjusted_low"] = out["pred_low"] - out["season_shift"]
    out["adjusted_high"] = out["pred_high"] - out["season_shift"]
    out["flag"] = interval_flag(out.assign(pred_low=out["adjusted_low"], pred_high=out["adjusted_high"]), min_minutes)
    out["flag_raw"] = interval_flag(out, min_minutes)
    return out


def rank_by_residual(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by residual (largest positive = most undervalued first); ties broken by player_id."""
    return df.sort_values(["residual", "player_id"], ascending=[False, True], kind="mergesort")


def _segment_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    m = regression_metrics(df["log_value"], df[pred_col])
    return {"n": m["n"], "rmse_log": m["rmse_log"], "median_ape": m["median_ape"], "within_50": m["within_50"],
            "mean_bias": float(np.mean(df[pred_col] - df["log_value"])) if len(df) else float("nan")}


def breakdown(df: pd.DataFrame, by: str | pd.Series, pred_col: str = "pred") -> pd.DataFrame:
    """Metrics per group (e.g. position group or age band)."""
    return pd.DataFrame({key: _segment_metrics(group, pred_col)
                         for key, group in df.groupby(by, observed=True)}).T


def failure_segments(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    """Metrics for the groups where a stats-based model is expected to struggle."""
    high_minutes = df["minutes"].quantile(0.9)
    segments = {
        "all players": df,
        "market value >= EUR 50m": df[df["market_value_eur"] >= 50e6],
        "goalkeepers": df[df["position_group"] == "GK"],
        "age <= 21": df[df["age"] <= 21],
        f"top-10% minutes (>= {high_minutes:,.0f})": df[df["minutes"] >= high_minutes],
        "450-899 minutes (injured / bench)": df[df["minutes"] < 900],
    }
    return pd.DataFrame({name: _segment_metrics(seg, pred_col) for name, seg in segments.items()}).T


def plot_predicted_vs_actual(df: pd.DataFrame, title: str, path) -> None:
    """Scatter of predicted vs market value on log axes, with the y = x line."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(df["market_value_eur"] / 1e6, np.exp(df["pred"]) / 1e6, s=10, alpha=0.35, color=SERIES[0],
               linewidths=0)
    lims = [0.05, 300]
    ax.plot(lims, lims, color=TEXT_MUTED, linewidth=1, linestyle="--")
    ax.annotate("prediction = market value", (60, 40), color=TEXT_MUTED, fontsize=8, rotation=38)
    ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims, title=title,
           xlabel="market value (EUR m, log scale)", ylabel="predicted value (EUR m, log scale)")
    fig.savefig(path)
    plt.close(fig)


def plot_residuals(df: pd.DataFrame, title: str, path) -> None:
    """Residual vs predicted value, with the binned median residual to show systematic bias."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pred_m = np.exp(df["pred"]) / 1e6
    ax.scatter(pred_m, df["residual"], s=10, alpha=0.3, color=SERIES[0], linewidths=0, label="player")
    bins = pd.qcut(df["pred"], 15)
    binned = df.groupby(bins, observed=True).agg(x=("pred", "median"), y=("residual", "median"))
    ax.plot(np.exp(binned["x"]) / 1e6, binned["y"], color=SERIES[1], marker="o", markersize=4,
            label="median residual (15 bins)")
    ax.axhline(0, color=TEXT_MUTED, linewidth=1, linestyle="--")
    ax.set(xscale="log", title=title, xlabel="predicted value (EUR m, log scale)",
           ylabel="residual = log(pred) - log(market)")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def test_intervals(name: str, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Conformally corrected 80% interval (log scale) from the saved quantile models and frozen correction."""
    from src.train import CONFORMAL_PATH, apply_correction  # local import: src.train imports this module

    correction = json.loads(CONFORMAL_PATH.read_text())[name]["correction_log"]
    low = joblib.load(MODELS_DIR / f"{name}_q10.joblib").predict(test)
    high = joblib.load(MODELS_DIR / f"{name}_q90.joblib").predict(test)
    return apply_correction(low, high, correction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force-rerun", action="store_true", help="evaluate the test set again (logged)")
    args = parser.parse_args()
    if TEST_MARKER.exists() and not args.force_rerun:
        sys.exit(f"The test season was already evaluated ({TEST_MARKER.read_text().strip()}). "
                 "It is spent: rerunning would invite tuning on it. Use --force-rerun only to regenerate reports.")
    first_run_note = TEST_MARKER.read_text().strip() if TEST_MARKER.exists() else ""

    apply_style()
    config = json.loads((MODELS_DIR / "model_config.json").read_text())
    main_name = config["main_model"]
    features = pd.read_parquet(FEATURES_PATH)
    test = features[features["season"] == TEST_SEASON].reset_index(drop=True)
    print(f"Test season {season_label(TEST_SEASON)}: {len(test):,} player-seasons. "
          "Models were chosen on CV only; this is a one-off report, not a tuning step.\n")

    preds = {name: joblib.load(MODELS_DIR / f"{name}.joblib").predict(test) for name in MODEL_LABELS}
    intervals = {name: test_intervals(name, test) for name in INTERVAL_MODELS}
    y = test["log_value"].to_numpy()

    rows = []
    for name, pred in preds.items():
        row = {"model": name, "label": MODEL_LABELS[name], **regression_metrics(y, pred),
               "mean_bias": float(np.mean(pred - y)), "interval_coverage_80": float("nan")}
        if name in intervals:
            low, high = intervals[name]
            row["interval_coverage_80"] = float(np.mean((y >= low) & (y <= high)))
        rows.append(row)
    test_metrics = pd.DataFrame(rows).set_index("model")
    test_metrics["evaluated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    test_metrics["forced_rerun"] = bool(args.force_rerun)
    test_metrics["first_evaluation"] = first_run_note or "this run"

    cv = pd.read_csv(REPORTS_DIR / "cv_summary.csv", index_col="model")
    comparison = pd.DataFrame({"cv_rmse_log": cv["rmse_log"], "test_rmse_log": test_metrics["rmse_log"],
                               "gap": test_metrics["rmse_log"] - cv["rmse_log"],
                               "cv_r2_log": cv["r2_log"], "test_r2_log": test_metrics["r2_log"],
                               "cv_coverage": cv["coverage_conformal"],
                               "test_coverage": test_metrics["interval_coverage_80"]})

    # Residual analysis for the main model.
    low, high = intervals[main_name]
    df = test.assign(pred=preds[main_name], pred_low=low, pred_high=high)
    df["residual"] = df["pred"] - df["log_value"]
    df["flag"] = interval_flag(df)
    df["age_band"] = age_bucket(df["age"])
    by_position = breakdown(df, "position_group").reindex(["GK", "DEF", "MID", "ATT"])
    by_age = breakdown(df, "age_band")
    by_league = breakdown(df, df["competition_id"].map(TOP5_LEAGUES))
    failures = failure_segments(df)

    test_metrics.to_csv(REPORTS_DIR / "test_metrics.csv")
    pd.concat({"position": by_position, "age_band": by_age, "league": by_league}).to_csv(
        REPORTS_DIR / "test_breakdown.csv")
    failures.to_csv(REPORTS_DIR / "test_failure_segments.csv")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    label = f"{season_label(TEST_SEASON)} test season, {MODEL_LABELS[main_name]}"
    plot_predicted_vs_actual(df, f"Predicted vs market value\n{label}", FIGURES_DIR / "pred_vs_actual.png")
    plot_residuals(df, f"Residuals vs prediction ({label})", FIGURES_DIR / "residuals_vs_pred.png")
    if not first_run_note:
        TEST_MARKER.write_text(f"first evaluated {test_metrics['evaluated_at'].iloc[0]}\n")

    pd.set_option("display.width", 240, "display.float_format", "{:.3f}".format)
    print("Test metrics (all models):")
    print(test_metrics.drop(columns=["label", "evaluated_at", "forced_rerun", "first_evaluation"]).to_string(
        formatters={"mae_eur": "{:,.0f}".format}))
    print("\nCV vs test (gap = test - CV RMSE, log):")
    print(comparison.to_string())
    big = comparison.index[comparison["gap"].abs() > 0.05].tolist()
    print(f"Models with |test - CV| RMSE gap > 0.05: {big or 'none'}")
    print(f"\nMain model ({main_name}) by position group:\n{by_position.to_string()}")
    print(f"\nBy age band:\n{by_age.to_string()}")
    print(f"\nBy league:\n{by_league.to_string()}")
    print(f"\nFailure analysis:\n{failures.to_string()}")
    print(f"\nInterval flags (>= {FLAG_MIN_MINUTES} min to be rated):\n{df['flag'].value_counts().to_string()}")

    cols = ["name", "club_name", "position_group", "age", "market_value_eur", "pred_eur", "gap_pct", "flag"]
    pl = rank_by_residual(df[df["competition_id"] == "GB1"]).assign(
        pred_eur=lambda d: np.exp(d["pred"]).round(-5), gap_pct=lambda d: 100 * (np.exp(d["residual"]) - 1))
    fmt = {"market_value_eur": "{:,.0f}".format, "pred_eur": "{:,.0f}".format, "gap_pct": "{:+.0f}%".format,
           "age": "{:.1f}".format}
    print("\nPremier League, largest positive residuals (model > market):")
    print(pl.head(10)[cols].to_string(index=False, formatters=fmt))
    print("\nPremier League, largest negative residuals (market > model):")
    print(pl.tail(10)[cols].iloc[::-1].to_string(index=False, formatters=fmt))
    print("\nSaved reports/test_metrics.csv, test_breakdown.csv, test_failure_segments.csv and 2 figures.")
    print("The test set is now spent: no further tuning based on it.")


if __name__ == "__main__":
    main()

"""SHAP explanations for the main XGBoost model.

Uses a fixed 2,000-row sample of development seasons (never the test season). SHAP values show
how each feature pushed the model's prediction for a player up or down relative to the average
prediction; they describe the model, not football. They show association with the model's
output, not causation: a large SHAP value for club strength does not mean that joining a better
club would raise a player's value by that amount.

Usage:
    python -m src.explain
"""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data import MODELS_DIR, REPORTS_DIR, TEST_SEASON
from src.evaluate import FIGURES_DIR
from src.features import FEATURES, FEATURES_PATH, ONE_HOT_FEATURES
from src.plot_style import SERIES, apply_style

SAMPLE_SIZE = 2000
SEED = 42
# One-hot columns are summed back into one group (SHAP values are additive, so this is exact).
ONE_HOT_GROUPS = {"pos_": "position group", "subpos_": "sub-position", "league_": "league", "foot_": "preferred foot"}


def grouped_shap(shap_values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Sum the SHAP values of each one-hot group per row; other features are kept as they are."""
    frame = pd.DataFrame(shap_values, columns=columns)
    # Match prefixes only on one-hot columns: league_median_log_value_prev also starts with "league_".
    group_of = {c: next((name for prefix, name in ONE_HOT_GROUPS.items() if c.startswith(prefix)), c)
                if c in ONE_HOT_FEATURES else c for c in columns}
    return frame.T.groupby(group_of).sum().T


def main() -> None:
    apply_style()
    model = joblib.load(MODELS_DIR / "xgb.joblib")
    df = pd.read_parquet(FEATURES_PATH)
    dev = df[df["season"] < TEST_SEASON]
    sample = dev.sample(n=SAMPLE_SIZE, random_state=SEED)
    X = sample[FEATURES]  # the pipeline's first step passes exactly these columns through, in this order

    explainer = shap.TreeExplainer(model.named_steps["xgb"])
    explanation = explainer(X)
    reconstructed = explanation.values.sum(axis=1) + explanation.base_values
    max_error = float(np.max(np.abs(reconstructed - model.predict(sample))))
    print(f"Additivity check: max |sum(SHAP) + base value - prediction| = {max_error:.2e} (log units)")
    if max_error > 1e-3:
        raise RuntimeError("SHAP values do not add up to the model's predictions")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("SHAP values, main XGBoost (2,000 development rows)", loc="left")
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png")
    plt.close("all")

    importance = grouped_shap(explanation.values, FEATURES).abs().mean().sort_values()
    top = importance.tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color=SERIES[0], height=0.6)
    ax.set(title="Mean |SHAP| per feature (one-hot columns grouped)", xlabel="mean |SHAP| (log value units)")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES_DIR / "shap_importance_grouped.png")
    plt.close(fig)

    importance.sort_values(ascending=False).rename("mean_abs_shap").to_csv(REPORTS_DIR / "shap_importance.csv")
    print("\nMean |SHAP| (grouped), top 15:")
    print(importance.sort_values(ascending=False).head(15).round(3).to_string())
    print("\nSaved reports/figures/shap_beeswarm.png, shap_importance_grouped.png and reports/shap_importance.csv")


if __name__ == "__main__":
    main()

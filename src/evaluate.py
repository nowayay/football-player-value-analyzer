"""Evaluation metrics on the log scale and the EUR scale.

Stage 2 scope: the metric function used for the baseline. Residual analysis,
backtest and SHAP are added in stages 4-5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true_log: np.ndarray | pd.Series, y_pred_log: np.ndarray | pd.Series) -> dict[str, float]:
    """Compute error metrics for predictions of log(market value in EUR).

    - rmse_log, r2_log: on the log scale the model is trained on.
    - mae_eur: average absolute error in euros (dominated by expensive players).
    - median_ape: median absolute % error in EUR; robust to a few huge misses.
    - within_25 / within_50: share of players whose prediction is within 25% / 50% of the market value.
    """
    y_true_log, y_pred_log = np.asarray(y_true_log, dtype=float), np.asarray(y_pred_log, dtype=float)
    true_eur, pred_eur = np.exp(y_true_log), np.exp(y_pred_log)
    ape = np.abs(pred_eur - true_eur) / true_eur
    ss_res = np.sum((y_true_log - y_pred_log) ** 2)
    ss_tot = np.sum((y_true_log - y_true_log.mean()) ** 2)
    return {
        "rmse_log": float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))),
        "r2_log": float(1 - ss_res / ss_tot),
        "mae_eur": float(np.mean(np.abs(pred_eur - true_eur))),
        "median_ape": float(np.median(ape)),
        "within_25": float(np.mean(ape <= 0.25)),
        "within_50": float(np.mean(ape <= 0.50)),
    }

"""Model definitions shared by training, evaluation and the app.

Kept in their own module (not in src.train, which runs as __main__) so that saved models
refer to a stable import path: joblib stores the class's module name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.evaluate import age_bucket
from src.features import FEATURES

SEED = 42


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

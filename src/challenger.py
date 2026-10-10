"""
Challenger model: gradient boosted trees (LightGBM) on the same raw features.

Same features and population as the champion scorecard, so any lift comes from
the algorithm (non-linearities and interactions), not from extra data.
Early stopping uses a random 20% of the development sample; the 2015
out-of-time sample is never touched during training.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .woe_scorecard import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=200,      # large leaves = stable, less overfit on 50k rows
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    n_estimators=2000,
    random_state=42,
    verbose=-1,
)


class GBMChallenger:
    def __init__(self, params: dict | None = None):
        self.params = params or PARAMS
        self.model: lgb.LGBMClassifier | None = None
        self.categories: dict[str, list] = {}

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES].copy()
        for c in CATEGORICAL_FEATURES:
            X[c] = pd.Categorical(X[c], categories=self.categories[c])
        return X

    def fit(self, df: pd.DataFrame) -> "GBMChallenger":
        self.categories = {c: sorted(df[c].dropna().unique()) for c in CATEGORICAL_FEATURES}
        X, y = self._prep(df), df[TARGET]
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.model = lgb.LGBMClassifier(**self.params)
        warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return self

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prep(df))[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return (pd.DataFrame({"feature": FEATURES, "gain": imp / imp.sum()})
                .sort_values("gain", ascending=False).reset_index(drop=True))

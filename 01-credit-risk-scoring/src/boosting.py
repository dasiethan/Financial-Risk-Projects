"""Gradient boosting challenger model (LightGBM)."""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

PARAMS = dict(
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)


def fit_lgbm(X: pd.DataFrame, y: pd.Series, seed: int) -> lgb.LGBMClassifier:
    """Pick the number of trees by early stopping on a validation split of the
    training set, then refit on the full training set with that number."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    probe = lgb.LGBMClassifier(n_estimators=5000, random_state=seed, **PARAMS)
    probe.fit(X_fit, y_fit, eval_X=X_val, eval_y=y_val, eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
    model = lgb.LGBMClassifier(n_estimators=probe.best_iteration_, random_state=seed, **PARAMS)
    return model.fit(X, y)

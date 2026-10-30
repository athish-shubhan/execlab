"""The ML layer: baselines first, then XGBoost tuned with Optuna under purged walk-forward.

Validation splits come from Skeptic (purged walk-forward with an embargo equal to
the label horizon), and every Optuna trial is written to Skeptic's ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skeptic.ledger import Ledger
from skeptic.splits import WalkForward


def wf_splits(n: int, horizon: int, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    t0 = np.arange(n)
    cv = WalkForward(n_splits=n_splits, min_train=n // 3, embargo=horizon)
    return [(s.train, s.test) for s in cv.split(t0, t0 + horizon)]


@dataclass
class Fitted:
    kind: str           # "logit" | "xgb" | "linear"
    model: Any
    task: str           # "multiclass" | "binary" | "regression"
    features: list[str]
    classes: np.ndarray | None = None

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        xx = x[self.features].to_numpy(float)
        if self.task == "regression":
            return np.asarray(self.model.predict(xx), float)
        p = self.model.predict_proba(xx)
        return p[:, 1] if self.task == "binary" else p


def _xgb(task: str, params: dict[str, Any], n_classes: int = 3) -> Any:
    from xgboost import XGBClassifier, XGBRegressor

    base = dict(tree_method="hist", n_jobs=2, verbosity=0, random_state=0, **params)
    if task == "regression":
        return XGBRegressor(objective="reg:squarederror", **base)
    if task == "binary":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **base)
    return XGBClassifier(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss", **base)


def _score(task: str, y: np.ndarray, pred: np.ndarray) -> float:
    if task == "regression":
        return float(mean_squared_error(y, pred))
    if task == "binary":
        return float(log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6), labels=[0, 1]))
    return float(log_loss(y, np.clip(pred, 1e-6, 1), labels=[0, 1, 2]))


def per_sample_loss(task: str, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if task == "regression":
        return (y - pred) ** 2
    if task == "binary":
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    p = np.clip(pred[np.arange(y.size), y.astype(int)], 1e-6, 1)
    return -np.log(p)


def fit_baseline(x: pd.DataFrame, y: np.ndarray, task: str, features: list[str]) -> Fitted:
    xx = x[features].to_numpy(float)
    if task == "regression":
        m = make_pipeline(StandardScaler(), LinearRegression()).fit(xx, y)
        return Fitted("linear", m, task, features)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0)).fit(xx, y.astype(int))
    return Fitted("logit", m, task, features, classes=np.unique(y))


def tune_xgb(x: pd.DataFrame, y: np.ndarray, task: str, features: list[str], horizon: int,
             n_trials: int, ledger: Ledger | None = None, study: str = "",
             seed: int = 0) -> tuple[Fitted, dict[str, Any], pd.DataFrame]:
    """Optuna search scored by purged walk-forward loss; refit the best on all of x.

    Each trial is logged to the ledger with its per-sample loss improvement over the
    baseline as the 'returns' series, so the search size is on record.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    xx = x[features].to_numpy(float)
    yy = y.astype(int) if task != "regression" else y
    splits = wf_splits(len(x), horizon)
    base_losses = []
    for tr_i, te_i in splits:
        b = fit_baseline(x.iloc[tr_i], y[tr_i], task, features)
        base_losses.append(per_sample_loss(task, yy[te_i], b.predict(x.iloc[te_i])))
    base_vec = np.concatenate(base_losses)
    rows = []

    def objective(trial: Any) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
        }
        losses = []
        for tr_i, te_i in splits:
            m = _xgb(task, params).fit(xx[tr_i], yy[tr_i])
            pred = m.predict(xx[te_i]) if task == "regression" else (
                m.predict_proba(xx[te_i])[:, 1] if task == "binary" else m.predict_proba(xx[te_i]))
            losses.append(per_sample_loss(task, yy[te_i], pred))
        vec = np.concatenate(losses)
        gain = base_vec - vec  # positive = better than baseline
        if ledger is not None:
            ledger.log(study, f"xgb_{task}", params, gain, seed=trial.number,
                       extra={"mean_loss": float(vec.mean()), "baseline_loss": float(base_vec.mean())})
        rows.append({"trial": trial.number, **params, "loss": float(vec.mean()),
                     "baseline_loss": float(base_vec.mean())})
        return float(vec.mean())

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(objective, n_trials=n_trials)
    best = dict(st.best_params)
    model = _xgb(task, best).fit(xx, yy)
    return Fitted("xgb", model, task, features), best, pd.DataFrame(rows)

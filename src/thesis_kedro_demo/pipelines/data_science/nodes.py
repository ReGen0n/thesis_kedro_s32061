"""Nodes of the data_science pipeline: split -> train -> evaluate -> explain."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def split_data(
    table: pd.DataFrame, target: str, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random train/test split controlled by `split.test_size` / `split.random_state`."""
    X = table.drop(columns=[target])
    y = table[[target]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )
    logger.info("Split: %d train / %d test rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict[str, Any]
) -> RegressorMixin:
    """Train the regressor selected by `model.type` in parameters.yml."""
    model_type = params["type"]
    if model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            random_state=params["random_state"], **params["gradient_boosting"]
        )
    elif model_type == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(**params["ridge"]))
    else:
        raise ValueError(f"Unknown model.type '{model_type}' (use gradient_boosting or ridge)")

    model.fit(X_train, y_train.iloc[:, 0])
    logger.info("Trained %s", model_type)
    return model


def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    y_train: pd.DataFrame,
    model_params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute RMSE / MAE / R² and compare with a naive mean-predictor baseline."""
    y_true = y_test.iloc[:, 0].to_numpy()
    y_pred = model.predict(X_test)
    baseline = np.full_like(y_true, y_train.iloc[:, 0].mean(), dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline)))
    metrics = {
        "model_type": model_params["type"],
        "rmse": round(rmse, 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "baseline_rmse": round(baseline_rmse, 3),
        "improvement_over_baseline_pct": round(100 * (1 - rmse / baseline_rmse), 1),
        "n_test": int(len(y_true)),
    }
    logger.info("Test metrics: %s", metrics)

    predictions = X_test[["age", "bmi"]].copy()
    predictions["actual"] = y_true
    predictions["predicted"] = y_pred.round(1)
    predictions["abs_error"] = (predictions["actual"] - predictions["predicted"]).abs()
    return predictions, metrics


def explain_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Permutation importance on the test set: how much R² drops per shuffled feature."""
    result = permutation_importance(
        model,
        X_test,
        y_test.iloc[:, 0],
        n_repeats=params["n_repeats"],
        random_state=params["random_state"],
        scoring="r2",
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean.round(4),
            "importance_std": result.importances_std.round(4),
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)
    logger.info("Top features: %s", ", ".join(importance["feature"].head(3)))
    return importance

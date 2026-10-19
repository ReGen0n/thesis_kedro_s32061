"""Unit tests for the pipeline nodes (pure functions, no Kedro session needed)."""

import numpy as np
import pandas as pd
import pytest

from thesis_kedro_demo.pipelines.data_processing.nodes import build_features, validate_patients
from thesis_kedro_demo.pipelines.data_science.nodes import (
    evaluate_model,
    explain_model,
    split_data,
    train_model,
)

VALIDATION = {
    "id_column": "patient_id",
    "min_rows": 10,
    "required_columns": ["patient_id", "age", "sex", "bmi", "blood_pressure", "progression"],
    "value_ranges": {"age": [0, 120], "bmi": [10, 70]},
}
FEATURES = {
    "target": "progression",
    "drop_columns": ["patient_id"],
    "bmi_bins": [0, 18.5, 25, 30, 100],
    "age_bins": [0, 40, 55, 120],
}
SPLIT = {"test_size": 0.25, "random_state": 0}
EVAL = {"n_repeats": 2, "random_state": 0}


@pytest.fixture
def raw() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 40
    bmi = rng.uniform(18, 40, n)
    return pd.DataFrame(
        {
            "patient_id": range(1, n + 1),
            "age": rng.integers(20, 80, n),
            "sex": rng.integers(1, 3, n),
            "bmi": bmi,
            "blood_pressure": rng.uniform(70, 130, n),
            "ldl": rng.uniform(60, 180, n),
            "hdl": rng.uniform(30, 90, n),
            "progression": bmi * 8 + rng.normal(0, 10, n),
        }
    )


def test_validate_passes(raw):
    assert validate_patients(raw, VALIDATION).equals(raw)


def test_validate_missing_column(raw):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_patients(raw.drop(columns="bmi"), VALIDATION)


def test_validate_out_of_range_and_duplicates(raw):
    bad = raw.copy()
    bad.loc[0, "age"] = 250
    bad.loc[1, "patient_id"] = bad.loc[2, "patient_id"]
    with pytest.raises(ValueError, match="outside"):
        validate_patients(bad, VALIDATION)


def test_build_features(raw):
    table = build_features(raw, FEATURES)
    assert "patient_id" not in table.columns and "sex" not in table.columns
    assert {"is_male", "bmi_category", "age_group", "ldl_hdl_ratio", "bmi_x_bp"} <= set(table.columns)
    assert table.columns[-1] == "progression"
    assert table["bmi_category"].between(0, 3).all()


@pytest.mark.parametrize("model_type", ["gradient_boosting", "ridge"])
def test_train_evaluate_explain(raw, model_type):
    model_params = {
        "type": model_type,
        "random_state": 0,
        "gradient_boosting": {"n_estimators": 20, "max_depth": 2},
        "ridge": {"alpha": 1.0},
    }
    table = build_features(raw, FEATURES)
    X_train, X_test, y_train, y_test = split_data(table, "progression", SPLIT)
    model = train_model(X_train, y_train, model_params)
    preds, metrics = evaluate_model(model, X_test, y_test, y_train, model_params)
    assert len(preds) == len(X_test) == 10
    assert metrics["model_type"] == model_type and metrics["rmse"] > 0
    importance = explain_model(model, X_test, y_test, EVAL)
    assert set(importance["feature"]) == set(X_test.columns)


def test_unknown_model_type(raw):
    table = build_features(raw, FEATURES)
    X_train, _, y_train, _ = split_data(table, "progression", SPLIT)
    with pytest.raises(ValueError, match="Unknown model.type"):
        train_model(X_train, y_train, {"type": "xgboost", "random_state": 0})

"""Nodes of the data_processing pipeline: validate -> build features."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def validate_patients(raw: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Check the raw patient table against a simple data contract.

    Checks: required columns, minimum number of rows, unique patient ids,
    no missing values and physiologically plausible value ranges.

    Raises:
        ValueError: listing every problem found, so the run stops early.
    """
    problems: list[str] = []

    missing = sorted(set(params["required_columns"]) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(raw) < params["min_rows"]:
        problems.append(f"only {len(raw)} rows (need >= {params['min_rows']})")

    id_col = params["id_column"]
    if raw[id_col].duplicated().any():
        problems.append(f"duplicated values in '{id_col}'")

    n_missing = int(raw.isna().sum().sum())
    if n_missing:
        problems.append(f"{n_missing} missing values")

    for column, (low, high) in params["value_ranges"].items():
        bad = ~raw[column].between(low, high)
        if bad.any():
            problems.append(f"{int(bad.sum())} rows with '{column}' outside [{low}, {high}]")

    if problems:
        raise ValueError("Raw data failed validation: " + "; ".join(problems))

    logger.info("Validation passed: %d patients, %d columns", *raw.shape)
    return raw


def build_features(patients: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Create the model input table.

    - drops identifier columns,
    - encodes sex as a 0/1 flag,
    - adds BMI and age categories (ordinal, from the bins in parameters.yml),
    - adds clinical ratios / interactions.
    """
    df = patients.drop(columns=params["drop_columns"]).copy()

    df["is_male"] = (df.pop("sex") == 2).astype(int)
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=params["bmi_bins"], labels=False, right=False
    ).astype(int)
    df["age_group"] = pd.cut(
        df["age"], bins=params["age_bins"], labels=False, right=False
    ).astype(int)
    df["ldl_hdl_ratio"] = (df["ldl"] / df["hdl"]).round(4)
    df["bmi_x_bp"] = (df["bmi"] * df["blood_pressure"]).round(2)

    # keep the target as the last column for readability
    target = params["target"]
    df = df[[c for c in df.columns if c != target] + [target]]
    logger.info("Model input table: %d rows, %d features", len(df), df.shape[1] - 1)
    return df

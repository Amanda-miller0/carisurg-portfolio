"""Data-loading and preparation utilities for the Mercer ED triage project.

This module refactors the Week 6–7 notebook data logic into reusable
functions. The target variable is the Emergency Severity Index (ESI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "esi"
RANDOM_STATE = 42

DEMOGRAPHIC_COLUMNS = [
    "age",
    "gender",
    "ethnicity",
    "race",
    "lang",
    "religion",
    "maritalstatus",
    "employstatus",
    "insurance_status",
]

ADMINISTRATIVE_COLUMNS = [
    "dep_name",
    "arrivalmode",
    "arrivalmonth",
    "arrivalday",
    "arrivalhour_bin",
]

LEAKAGE_COLUMNS = [
    "disposition",
    "previousdispo",
]


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Load the triage CSV file and perform basic validation.

    Parameters
    ----------
    data_path:
        Path to the Yale EMMLC triage CSV file.

    Returns
    -------
    pandas.DataFrame
        The loaded patient-level dataset.

    Raises
    ------
    FileNotFoundError
        If the configured CSV file does not exist.
    ValueError
        If the CSV is empty.
    KeyError
        If the target column is missing.
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}. "
            "Check the path in config.yaml or place the CSV in the expected folder."
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError("The dataset was loaded, but it contains no rows.")

    validate_schema(dataframe)
    return dataframe


def validate_schema(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    required_columns: Iterable[str] | None = None,
) -> None:
    """Check that the dataset contains the columns required by the pipeline."""
    expected = {target_column}

    if required_columns is not None:
        expected.update(required_columns)

    missing = sorted(expected.difference(dataframe.columns))

    if missing:
        raise KeyError(
            "Dataset schema validation failed. Missing column(s): "
            + ", ".join(missing)
        )


def get_feature_columns(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> list[str]:
    """Return model feature names after removing excluded columns.

    This follows the Week 6–7 notebook approach by excluding the target,
    demographic variables, administrative variables, and known leakage
    variables.
    """
    excluded = {
        target_column,
        *DEMOGRAPHIC_COLUMNS,
        *ADMINISTRATIVE_COLUMNS,
        *LEAKAGE_COLUMNS,
    }

    features = [column for column in dataframe.columns if column not in excluded]

    if not features:
        raise ValueError("No model features remain after excluded columns are removed.")

    return features


def prepare_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create the numeric feature matrix and integer target vector.

    The steps reproduce the Week 6–7 notebook logic:
    1. remove rows with missing target values;
    2. exclude demographic, administrative, and leakage columns;
    3. coerce model features to numeric;
    4. fill remaining feature gaps using column medians;
    5. convert ESI labels to integers.
    """
    validate_schema(dataframe, target_column=target_column)

    valid_rows = dataframe[target_column].notna()
    working_data = dataframe.loc[valid_rows].copy()

    feature_columns = get_feature_columns(
        working_data,
        target_column=target_column,
    )

    features = working_data[feature_columns].copy()
    target = working_data[target_column].copy()

    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    medians = features.median(numeric_only=True)
    features = features.fillna(medians)

    # Columns that are entirely missing cannot be repaired by a median.
    all_missing_columns = features.columns[features.isna().all()].tolist()
    if all_missing_columns:
        raise ValueError(
            "The following feature columns are entirely missing or non-numeric: "
            + ", ".join(all_missing_columns)
        )

    remaining_missing = int(features.isna().sum().sum())
    if remaining_missing:
        raise ValueError(
            f"{remaining_missing} missing feature values remain after preparation."
        )

    try:
        target = pd.to_numeric(target, errors="raise").astype(int)
    except (TypeError, ValueError) as error:
        raise ValueError("The ESI target must contain numeric class labels.") from error

    return features, target


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified training/test split."""
    if len(features) != len(target):
        raise ValueError("Features and target must contain the same number of rows.")

    if features.empty:
        raise ValueError("Cannot split an empty feature matrix.")

    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )


def load_and_split_data(
    data_path: str | Path,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Convenience function that loads, prepares, and splits the dataset."""
    dataframe = load_data(data_path)
    features, target = prepare_features_and_target(
        dataframe,
        target_column=target_column,
    )
    return split_data(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
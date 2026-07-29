from __future__ import annotations

from typing import Iterable

import pandas as pd

from .data import (
    TARGET_COLUMN,
    DEMOGRAPHIC_COLUMNS,
    ADMINISTRATIVE_COLUMNS,
    LEAKAGE_COLUMNS,
    validate_schema,
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
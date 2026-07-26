"""
Model building and evaluation functions for the Week 8 interim.
"""

from time import perf_counter
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(
    name: str = "logistic_regression",
    params: dict[str, Any] | None = None,
    seed: int = 42,
):
    """
    Build and return a machine-learning model.

    Parameters
    ----------
    name:
        Name of the model to create.
    params:
        Optional model parameters.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    sklearn estimator
        The requested model.
    """
    params = params.copy() if params else {}

    if name.lower() != "logistic_regression":
        raise ValueError(
            "For this interim, the supported model is "
            "'logistic_regression'."
        )

    params.setdefault("max_iter", 2000)
    params.setdefault("class_weight", "balanced")
    params.setdefault("random_state", seed)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**params)),
        ]
    )

    return model


def evaluate(model, X, y) -> dict[str, Any]:
    """
    Evaluate a trained classification model.

    Parameters
    ----------
    model:
        A fitted scikit-learn model.
    X:
        Test features.
    y:
        True test labels.

    Returns
    -------
    dict
        Accuracy, weighted precision, weighted recall,
        weighted F1 score, inference time, confusion matrix,
        and classification report.
    """
    start_time = perf_counter()
    predictions = model.predict(X)
    inference_time = perf_counter() - start_time

    results = {
        "accuracy": accuracy_score(y, predictions),
        "weighted_precision": precision_score(
            y,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "weighted_recall": recall_score(
            y,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "inference_time_seconds": inference_time,
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    print("Model Evaluation")
    print("----------------")
    print(f"Accuracy:           {results['accuracy']:.4f}")
    print(f"Weighted precision: {results['weighted_precision']:.4f}")
    print(f"Weighted recall:    {results['weighted_recall']:.4f}")
    print(f"Weighted F1 score:  {results['weighted_f1']:.4f}")
    print(f"Macro F1 score:     {results['macro_f1']:.4f}")
    print(f"Inference time:     {inference_time:.4f} seconds")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(np.array(results["confusion_matrix"]))

    return results
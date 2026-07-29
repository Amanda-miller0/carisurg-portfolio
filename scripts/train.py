"""Train and evaluate the final Mercer ED triage model."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import yaml

# Allow the script to import modules from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_data, split_data
from src.features import prepare_features_and_target
from src.model import build_model, evaluate
from src.utils import ensure_directory, save_json


def load_config(config_path: str | Path) -> dict:
    """Load the project configuration file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a YAML dictionary.")

    return config


def main() -> None:
    """Run the complete model-training pipeline."""
    config_path = PROJECT_ROOT / "config.yaml"
    config = load_config(config_path)

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    hyperparameters = config.get("hyperparameters", {})
    training_config = config.get("training", {})
    output_config = config.get("output", {})

    data_path = PROJECT_ROOT / data_config.get(
        "path",
        "data/triage_dataset.csv",
    )

    target_column = data_config.get("target", "esi")
    model_name = model_config.get("name", "logistic_regression")

    test_size = training_config.get("test_size", 0.20)
    random_state = training_config.get("random_state", 42)

    model_output_path = PROJECT_ROOT / output_config.get(
        "model_path",
        "outputs/final_model.joblib",
    )

    metrics_output_path = PROJECT_ROOT / output_config.get(
        "metrics_path",
        "outputs/metrics.json",
    )

    print("Loading dataset...")
    dataframe = load_data(data_path)

    print("Preparing features and target...")
    features, target = prepare_features_and_target(
        dataframe,
        target_column=target_column,
    )

    print(f"Rows available: {len(features)}")
    print(f"Number of features: {features.shape[1]}")

    print("Creating training and test sets...")
    X_train, X_test, y_train, y_test = split_data(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    print("Building model...")
    model = build_model(
        name=model_name,
        params=hyperparameters,
        seed=random_state,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate(model, X_test, y_test)

    ensure_directory(model_output_path.parent)
    joblib.dump(model, model_output_path)

    save_json(metrics, metrics_output_path)

    print(f"\nModel saved to: {model_output_path}")
    print(f"Metrics saved to: {metrics_output_path}")
    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
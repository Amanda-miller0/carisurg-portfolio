# Project Handover

## Project

Evaluating Machine Learning Models for Emergency Department Triage Using Physiological Vital Signs

## Repository Structure

```
src/
    data.py
    features.py
    model.py
    utils.py

scripts/
    train.py

tests/
    test_features.py
    test_model.py

docs/
    model-selection.md
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

The project configuration is stored in:

```
config.yaml
```

Update the dataset path if the dataset is stored in a different location.

## Running the Project

Train the model using:

```bash
python scripts/train.py
```

## Testing

Run the automated tests using:

```bash
pytest
```

## Outputs

The training script saves:

- Trained model (`.joblib`)
- Evaluation metrics (`.json`)

to the configured output directory.

## Notes

The emergency department dataset is not included in this repository because it contains governed research data. To reproduce the results, place the approved dataset in the location specified in `config.yaml`.

The repository has been organized into reusable modules to improve maintainability, testing, and future model development.
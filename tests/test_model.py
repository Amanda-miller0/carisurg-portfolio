from sklearn.pipeline import Pipeline

from src.model import build_model


def test_build_model_returns_pipeline():
    model = build_model()

    assert isinstance(model, Pipeline)
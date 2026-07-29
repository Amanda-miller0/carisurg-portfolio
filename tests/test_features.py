import pandas as pd

from src.features import prepare_features_and_target


def test_prepare_features_and_target():
    dataframe = pd.DataFrame(
        {
            "esi": [1, 2, 3],
            "heart_rate": [80, 95, 70],
            "temperature": [36.8, 37.5, 38.2],
        }
    )

    features, target = prepare_features_and_target(dataframe)

    assert len(features) == 3
    assert len(target) == 3
    assert "esi" not in features.columns
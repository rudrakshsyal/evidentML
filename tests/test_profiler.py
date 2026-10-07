import pandas as pd

from evidentml.data_engine.profiler import profile_dataset


def test_profile_dataset():
    df = pd.DataFrame(
        {
            "age": [25, 30, 30],
            "churn": [0, 1, 1],
        }
    )

    profile = profile_dataset(df)

    assert profile["rows"] == 3
    assert profile["columns"] == 2
    assert profile["duplicate_rows"] == 1
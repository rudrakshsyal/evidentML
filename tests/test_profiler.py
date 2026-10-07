import pandas as pd

from evidentml.data_engine.profiler import profile_dataset


def test_profile_dataset():
    df = pd.DataFrame(
        {
            "customer_id": ["A1", "A2", "A3"],
            "age": [25, 30, 30],
            "state": ["CA", "CA", "NY"],
            "churn": [0, 1, 1],
        }
    )

    profile = profile_dataset(df)

    assert profile["rows"] == 3
    assert profile["columns"] == 4
    assert profile["duplicate_rows"] == 0

    assert profile["columns_profile"]["age"]["physical_dtype"] == "int64"

    assert (
        profile["columns_profile"]["customer_id"]["semantic_type"]
        == "possible_identifier"
    )

    assert profile["columns_profile"]["state"]["semantic_type"] == "categorical"

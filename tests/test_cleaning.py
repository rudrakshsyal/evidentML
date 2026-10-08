import pandas as pd

from evidentml.data_engine.cleaning import (
    clean_categorical_variants,
    clean_mixed_numeric_column,
)


def test_clean_mixed_numeric_column():
    series = pd.Series(
        [
            "72000",
            "$85,000",
            "90k",
            None,
        ]
    )

    cleaned = clean_mixed_numeric_column(series)

    assert cleaned.iloc[0] == 72000
    assert cleaned.iloc[1] == 85000
    assert cleaned.iloc[2] == 90000
    assert pd.isna(cleaned.iloc[3])

def test_clean_categorical_variants():
    series = pd.Series(
        [
            "Plus",
            "PLUS",
            "plus",
            " Plus ",
            "Basic",
        ]
    )

    cleaned = clean_categorical_variants(series)

    assert cleaned.nunique() == 2
    assert set(cleaned) == {"Plus", "Basic"}
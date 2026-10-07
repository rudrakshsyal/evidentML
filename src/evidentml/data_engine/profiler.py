import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate a basic structural profile of a dataset."""

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_names": df.columns.tolist(),
    }
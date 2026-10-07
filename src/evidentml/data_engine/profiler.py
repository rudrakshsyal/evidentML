from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)


def infer_semantic_type(series: pd.Series) -> str:
    """
    Infer a conservative semantic type for a pandas Series.
    """

    non_null = series.dropna()

    if non_null.empty:
        return "unknown"

    if is_bool_dtype(series):
        return "boolean"

    if is_datetime64_any_dtype(series):
        return "datetime"

    # Numeric columns should stay numeric.
    # High uniqueness alone does NOT make them identifiers.
    if is_numeric_dtype(series):
        return "numeric"

    if is_string_dtype(series):
        sample = non_null.astype(str).head(500)

        # Detect datetime-like strings
        parsed_dates = pd.to_datetime(
            sample,
            errors="coerce",
            format="mixed",
        )

        date_success_rate = parsed_dates.notna().mean()

        if date_success_rate >= 0.90:
            return "datetime_like"

        unique_ratio = non_null.nunique() / len(non_null)

        # Identifier suspicion makes more sense for strings
        if unique_ratio > 0.98:
            return "possible_identifier"

        return "categorical"

    return "unknown"


def profile_column(series: pd.Series) -> dict[str, Any]:
    """Create a structural profile for a single column."""

    total_rows = len(series)
    non_null_count = series.notna().sum()
    null_count = series.isna().sum()

    unique_count = series.nunique(dropna=True)

    unique_ratio = unique_count / non_null_count if non_null_count > 0 else 0.0

    value_counts = series.value_counts(
        normalize=True,
        dropna=True,
    )

    dominant_value_pct = float(value_counts.iloc[0]) if not value_counts.empty else None

    profile: dict[str, Any] = {
        "physical_dtype": str(series.dtype),
        "semantic_type": infer_semantic_type(series),
        "null_count": int(null_count),
        "null_pct": float(null_count / total_rows) if total_rows > 0 else 0.0,
        "unique_count": int(unique_count),
        "unique_ratio": float(unique_ratio),
        "dominant_value_pct": dominant_value_pct,
        "is_constant": unique_count <= 1,
    }

    if is_numeric_dtype(series):
        profile.update(
            {
                "min": float(series.min()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
                "mean": float(series.mean()) if series.notna().any() else None,
                "median": float(series.median()) if series.notna().any() else None,
            }
        )

    return profile


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate a dataset-level structural profile.

    This function observes the dataset only.
    It does not clean or modify the data.
    """

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_names": df.columns.tolist(),
        "columns_profile": {
            column: profile_column(df[column]) for column in df.columns
        },
    }

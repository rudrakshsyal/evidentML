from __future__ import annotations

from typing import Any

import pandas as pd

from evidentml.data_engine.quality import _normalize_numeric_string


def clean_mixed_numeric_column(series: pd.Series,) -> pd.Series:
    """Convert mixed numeric strings into numeric values."""

    normalized = (
        series.astype("string")
        .map(
            lambda value: (
                _normalize_numeric_string(value)
                if pd.notna(value)
                else value
            )
        )
    )

    return pd.to_numeric(normalized, errors="coerce",)

def clean_categorical_variants(series: pd.Series,) -> pd.Series:
    """
    Normalize case/whitespace variants.

    Example:
    Plus, PLUS, plus -> Plus

    The most common cleaned spelling becomes canonical.
    """

    non_null = series.dropna().astype(str)

    if non_null.empty:
        return series

    stripped = non_null.str.strip()
    normalized = stripped.str.casefold()

    canonical_map: dict[str, str] = {}

    for key in normalized.unique():
        candidates = stripped[normalized == key]

        canonical = candidates.value_counts().index[0]

        canonical_map[key] = canonical

    cleaned = series.copy()

    mask = cleaned.notna()

    cleaned.loc[mask] = (
        cleaned.loc[mask]
        .astype(str)
        .str.strip()
        .map(
            lambda value: canonical_map[
                value.casefold()
            ]
        )
    )

    return cleaned

def apply_safe_cleaning(df: pd.DataFrame, audit: dict[str, Any],) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Apply only high-confidence cleaning operations.

    Returns:
        cleaned dataframe
        transformation lineage
    """

    cleaned_df = df.copy()

    lineage = []

    for issue in audit["issues"]:

        column = issue.get("column")
        issue_type = issue["type"]

        if column is None:
            continue

        if issue_type == "mixed_numeric_format":

            before_dtype = str(
                cleaned_df[column].dtype
            )

            cleaned_df[column] = (
                clean_mixed_numeric_column(
                    cleaned_df[column]
                )
            )

            lineage.append(
                {
                    "column": column,
                    "action": "convert_mixed_numeric",
                    "reason": issue_type,
                    "before_dtype": before_dtype,
                    "after_dtype": str(
                        cleaned_df[column].dtype
                    ),
                    "confidence": "high",
                }
            )

        elif issue_type == "dirty_categorical_variants":

            before_unique = int(
                cleaned_df[column].nunique(
                    dropna=True
                )
            )

            cleaned_df[column] = (
                clean_categorical_variants(
                    cleaned_df[column]
                )
            )

            after_unique = int(
                cleaned_df[column].nunique(
                    dropna=True
                )
            )

            lineage.append(
                {
                    "column": column,
                    "action": "normalize_categorical_variants",
                    "reason": issue_type,
                    "before_unique": before_unique,
                    "after_unique": after_unique,
                    "confidence": "high",
                }
            )

    return cleaned_df, lineage
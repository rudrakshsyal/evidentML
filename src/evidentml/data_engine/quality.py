from __future__ import annotations

from typing import Any

import pandas as pd


def create_issue(issue_type: str, severity: str, message: str, column: str | None = None, evidence: dict[str, Any] | None = None,) -> dict[str, Any]:
    """
    Create a standardized data-quality issue.

    Keeping issues structured makes them usable later by:
    - agents
    - benchmark scoring
    - reports/UI
    - cleaning logic
    """

    issue = {
        "type": issue_type,
        "severity": severity,
        "message": message,
        "evidence": evidence or {},
    }

    if column is not None:
        issue["column"] = column

    return issue

def detect_duplicate_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect exact duplicate rows."""

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        return []

    return [
        create_issue(
            issue_type="duplicate_rows",
            severity="warning",
            message=f"Dataset contains {duplicate_count} exact duplicate rows.",
            evidence={
                "duplicate_count": duplicate_count,
                "duplicate_pct": duplicate_count / len(df),
            },
        )
    ]

def detect_constant_columns(profile: dict[str, Any],) -> list[dict[str, Any]]:
    """Detect columns containing only one unique non-null value."""

    issues = []

    for column, column_profile in profile["columns_profile"].items():

        if column_profile["is_constant"]:
            issues.append(
                create_issue(
                    issue_type="constant_column",
                    severity="warning",
                    column=column,
                    message=f"Column '{column}' contains only one unique value.",
                    evidence={
                        "unique_count": column_profile["unique_count"],
                    },
                )
            )

    return issues

def detect_near_constant_columns(profile: dict[str, Any], threshold: float = 0.995,) -> list[dict[str, Any]]:
    """
    Detect columns dominated by a single value.

    A column is considered near-constant when the most common
    value represents at least `threshold` of non-null observations.
    """

    issues = []

    for column, column_profile in profile["columns_profile"].items():

        dominant_pct = column_profile.get("dominant_value_pct")

        if dominant_pct is None:
            continue

        if (
            dominant_pct >= threshold
            and not column_profile["is_constant"]
        ):
            issues.append(
                create_issue(
                    issue_type="near_constant_column",
                    severity="info",
                    column=column,
                    message=(
                        f"Column '{column}' is dominated by one value "
                        f"({dominant_pct:.1%} of observations)."
                    ),
                    evidence={
                        "dominant_value_pct": dominant_pct,
                        "threshold": threshold,
                    },
                )
            )

    return issues

def detect_missingness(profile: dict[str, Any], warning_threshold: float = 0.05, high_threshold: float = 0.50,) -> list[dict[str, Any]]:
    """Flag columns containing missing values."""

    issues = []

    for column, column_profile in profile["columns_profile"].items():

        null_pct = column_profile["null_pct"]

        if null_pct == 0:
            continue

        severity = "info"

        if null_pct >= high_threshold:
            severity = "high"
        elif null_pct >= warning_threshold:
            severity = "warning"

        issues.append(
            create_issue(
                issue_type="missing_values",
                severity=severity,
                column=column,
                message=(
                    f"Column '{column}' contains "
                    f"{null_pct:.1%} missing values."
                ),
                evidence={
                    "null_count": column_profile["null_count"],
                    "null_pct": null_pct,
                },
            )
        )

    return issues

def detect_possible_identifiers(profile: dict[str, Any],) -> list[dict[str, Any]]:
    """Flag columns whose semantic type suggests an identifier."""

    issues = []

    for column, column_profile in profile["columns_profile"].items():

        if column_profile["semantic_type"] == "possible_identifier":

            issues.append(
                create_issue(
                    issue_type="possible_identifier",
                    severity="info",
                    column=column,
                    message=(
                        f"Column '{column}' may be an identifier "
                        "and may not be appropriate as a model feature."
                    ),
                    evidence={
                        "unique_ratio": column_profile["unique_ratio"],
                    },
                )
            )

    return issues

def audit_data_quality(df: pd.DataFrame, profile: dict[str, Any],) -> dict[str, Any]:
    """
    Run deterministic data-quality checks.

    V0.1 intentionally starts with high-confidence checks.
    More semantic checks will be added incrementally.
    """

    issues = []

    issues.extend(detect_duplicate_rows(df))
    issues.extend(detect_constant_columns(profile))
    issues.extend(detect_near_constant_columns(profile))
    issues.extend(detect_missingness(profile))
    issues.extend(detect_possible_identifiers(profile))

    return {
        "issue_count": len(issues),
        "issues": issues,
    }
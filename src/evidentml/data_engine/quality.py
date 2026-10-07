from __future__ import annotations

import re
from typing import Any

import pandas as pd


def create_issue(
    issue_type: str,
    severity: str,
    message: str,
    column: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
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


def detect_constant_columns(
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
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


def detect_near_constant_columns(
    profile: dict[str, Any],
    threshold: float = 0.995,
) -> list[dict[str, Any]]:
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

        if dominant_pct >= threshold and not column_profile["is_constant"]:
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


def detect_missingness(
    profile: dict[str, Any],
    warning_threshold: float = 0.05,
    high_threshold: float = 0.50,
) -> list[dict[str, Any]]:
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
                message=(f"Column '{column}' contains {null_pct:.1%} missing values."),
                evidence={
                    "null_count": column_profile["null_count"],
                    "null_pct": null_pct,
                },
            )
        )

    return issues


def detect_possible_identifiers(
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
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


def _normalize_numeric_string(value: str) -> str:
    """Normalize common numeric string formats."""
    value = value.strip()
    value = value.replace("$", "").replace(",", "")

    if re.fullmatch(r"-?\d+(\.\d+)?[kK]", value):
        number = float(value[:-1]) * 1000
        return str(number)

    return value


def detect_mixed_numeric_formats(
    df: pd.DataFrame,
    min_parse_rate: float = 0.80,
) -> list[dict[str, Any]]:
    """
    Detect string columns that appear mostly numeric after normalization.

    Example:
    72000, "$72,000", "85k"
    """

    issues = []

    for column in df.columns:
        series = df[column]

        if not pd.api.types.is_string_dtype(series):
            continue

        non_null = series.dropna().astype(str)

        if len(non_null) == 0:
            continue

        normalized = non_null.map(_normalize_numeric_string)

        parsed = pd.to_numeric(
            normalized,
            errors="coerce",
        )

        parse_rate = parsed.notna().mean()

        if parse_rate >= min_parse_rate:
            issues.append(
                create_issue(
                    issue_type="mixed_numeric_format",
                    severity="warning",
                    column=column,
                    message=(
                        f"Column '{column}' appears numeric but contains "
                        "mixed string formatting."
                    ),
                    evidence={
                        "numeric_parse_rate": float(parse_rate),
                    },
                )
            )

    return issues


def detect_dirty_categorical_variants(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Detect categorical values that collapse after trimming/case normalization.

    Example:
    CA, ca, ' CA '
    """

    issues = []

    for column in df.columns:
        series = df[column]

        if not pd.api.types.is_string_dtype(series):
            continue

        non_null = series.dropna().astype(str)

        if len(non_null) == 0:
            continue

        normalized = non_null.str.strip().str.casefold()

        original_unique = non_null.nunique()
        normalized_unique = normalized.nunique()

        if normalized_unique < original_unique:
            issues.append(
                create_issue(
                    issue_type="dirty_categorical_variants",
                    severity="warning",
                    column=column,
                    message=(
                        f"Column '{column}' contains values that differ "
                        "only by case or surrounding whitespace."
                    ),
                    evidence={
                        "original_unique_count": int(original_unique),
                        "normalized_unique_count": int(normalized_unique),
                    },
                )
            )

    return issues


def detect_extreme_numeric_outliers(
    df: pd.DataFrame,
    robust_z_threshold: float = 6.0,
    min_non_null: int = 20,
) -> list[dict[str, Any]]:
    """
    Detect extreme numeric outliers using robust z-scores.

    Robust z-scores use the median and median absolute deviation (MAD),
    making them less sensitive to the outliers they are trying to detect.
    """

    issues = []

    for column in df.select_dtypes(include="number").columns:
        series = df[column].dropna()

        if len(series) < min_non_null:
            continue

        median = series.median()
        mad = (series - median).abs().median()

        if mad == 0:
            continue

        robust_z = 0.6745 * (series - median) / mad

        outlier_mask = robust_z.abs() > robust_z_threshold
        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            continue

        outlier_values = series[outlier_mask]

        issues.append(
            create_issue(
                issue_type="extreme_numeric_outliers",
                severity="warning",
                column=column,
                message=(
                    f"Column '{column}' contains "
                    f"{outlier_count} extreme numeric values."
                ),
                evidence={
                    "outlier_count": outlier_count,
                    "outlier_pct": float(outlier_count / len(series)),
                    "min_outlier": float(outlier_values.min()),
                    "max_outlier": float(outlier_values.max()),
                    "median": float(median),
                    "mad": float(mad),
                    "robust_z_threshold": robust_z_threshold,
                },
            )
        )

    return issues


def detect_target_leakage_candidates(
    df: pd.DataFrame,
    target: str,
    min_group_size: int = 20,
    missingness_gap_threshold: float = 0.95,
    purity_threshold: float = 0.995,
) -> list[dict[str, Any]]:
    """
    Detect high-confidence signals of possible target leakage.

    This does NOT declare leakage automatically.
    It flags suspicious relationships for further investigation.
    """

    issues = []

    if target not in df.columns:
        return issues

    y = df[target]

    classes = y.dropna().unique()

    if len(classes) != 2:
        return issues

    # Encode arbitrary binary labels as 0/1.
    class_mapping = {
        classes[0]: 0,
        classes[1]: 1,
    }

    y_binary = y.map(class_mapping)

    for column in df.columns:
        if column == target:
            continue

        series = df[column]

        # ---------------------------------------------------------
        # Signal 1: Feature is essentially a direct copy of target
        # ---------------------------------------------------------

        comparable = pd.DataFrame(
            {
                "feature": series,
                "target": y,
            }
        ).dropna()

        if len(comparable) >= min_group_size:
            match_rate = (comparable["feature"] == comparable["target"]).mean()

            if match_rate >= purity_threshold:
                issues.append(
                    create_issue(
                        issue_type="possible_target_leakage",
                        severity="high",
                        column=column,
                        message=(
                            f"Column '{column}' appears to closely "
                            "replicate the target."
                        ),
                        evidence={
                            "signal": "direct_target_copy",
                            "match_rate": float(match_rate),
                        },
                    )
                )

                continue

        # ---------------------------------------------------------
        # Signal 2: Missingness almost determines the target
        # ---------------------------------------------------------

        missing_mask = series.isna()

        if missing_mask.any() and (~missing_mask).any():
            target_when_missing = y_binary[missing_mask & y_binary.notna()]

            target_when_present = y_binary[(~missing_mask) & y_binary.notna()]

            if (
                len(target_when_missing) >= min_group_size
                and len(target_when_present) >= min_group_size
            ):
                missing_target_rate = target_when_missing.mean()
                present_target_rate = target_when_present.mean()

                gap = abs(missing_target_rate - present_target_rate)

                if gap >= missingness_gap_threshold:
                    issues.append(
                        create_issue(
                            issue_type="possible_target_leakage",
                            severity="high",
                            column=column,
                            message=(
                                f"Missingness in '{column}' is almost "
                                "perfectly associated with the target."
                            ),
                            evidence={
                                "signal": ("missingness_target_association"),
                                "target_rate_when_missing": float(missing_target_rate),
                                "target_rate_when_present": float(present_target_rate),
                                "rate_gap": float(gap),
                            },
                        )
                    )

        # ---------------------------------------------------------
        # Signal 3: Low-cardinality feature almost perfectly maps
        # to the target
        # ---------------------------------------------------------

        non_null = series.dropna()

        if len(non_null) == 0:
            continue

        unique_count = non_null.nunique()

        if 2 <= unique_count <= 50:
            temp = pd.DataFrame(
                {
                    "feature": series,
                    "target": y_binary,
                }
            ).dropna()

            if len(temp) < min_group_size:
                continue

            table = pd.crosstab(
                temp["feature"],
                temp["target"],
            )

            correct_if_memorized = table.max(axis=1).sum()

            weighted_purity = correct_if_memorized / table.to_numpy().sum()

            if weighted_purity >= purity_threshold:
                issues.append(
                    create_issue(
                        issue_type="possible_target_leakage",
                        severity="high",
                        column=column,
                        message=(
                            f"Column '{column}' almost perfectly determines the target."
                        ),
                        evidence={
                            "signal": "categorical_target_mapping",
                            "weighted_purity": float(weighted_purity),
                            "unique_count": int(unique_count),
                        },
                    )
                )

    return issues


def audit_data_quality(
    df: pd.DataFrame,
    profile: dict[str, Any],
    target: str | None = None,
) -> dict[str, Any]:
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
    issues.extend(detect_mixed_numeric_formats(df))
    issues.extend(detect_dirty_categorical_variants(df))
    issues.extend(detect_extreme_numeric_outliers(df))

    if target is not None:
        issues.extend(
            detect_target_leakage_candidates(
                df=df,
                target=target,
            )
        )

    return {
        "issue_count": len(issues),
        "issues": issues,
    }

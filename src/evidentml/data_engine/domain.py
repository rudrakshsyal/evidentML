from __future__ import annotations

from typing import Any

import pandas as pd


def validate_domain_rules(df: pd.DataFrame, rules: dict[str, dict[str, Any]],) -> list[dict[str, Any]]:
    """Validate explicit business/domain constraints."""

    issues = []

    for column, rule in rules.items():
        if column not in df.columns:
            continue

        series = df[column]

        if "min" in rule or "max" in rule:
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = pd.Series(False, index=df.index)

            if "min" in rule:
                invalid |= numeric < rule["min"]

            if "max" in rule:
                invalid |= numeric > rule["max"]

            invalid_count = int(invalid.sum())

            if invalid_count:
                issues.append(
                    {
                        "type": "domain_rule_violation",
                        "severity": "high",
                        "column": column,
                        "message": (
                            f"Column '{column}' contains "
                            f"{invalid_count} values outside "
                            "the allowed range."
                        ),
                        "evidence": {
                            "violation_count": invalid_count,
                            "min_allowed": rule.get("min"),
                            "max_allowed": rule.get("max"),
                            "sample_values": (
                                series[invalid]
                                .drop_duplicates()
                                .head(5)
                                .tolist()
                            ),
                        },
                    }
                )

    return issues
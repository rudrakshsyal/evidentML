import pandas as pd

from evidentml.data_engine.profiler import profile_dataset
from evidentml.data_engine.quality import audit_data_quality


def test_quality_audit_detects_basic_issues():

    df = pd.DataFrame(
        {
            "customer_id": ["A1", "A2", "A3", "A3"],
            "country": ["US", "US", "US", "US"],
            "status": ["active", "active", "active", "active"],
            "income": [100, None, 200, 200],
        }
    )

    profile = profile_dataset(df)

    audit = audit_data_quality(df, profile)

    issue_types = [issue["type"] for issue in audit["issues"]]

    assert "duplicate_rows" in issue_types
    assert "constant_column" in issue_types
    assert "missing_values" in issue_types


def test_detects_extreme_numeric_outlier():
    df = pd.DataFrame(
        {
            "value": [
                10,
                11,
                10,
                12,
                11,
                10,
                12,
                11,
                10,
                11,
                12,
                10,
                11,
                12,
                10,
                11,
                10,
                12,
                11,
                10,
                11,
                12,
                10,
                11,
                1000,
            ]
        }
    )

    profile = profile_dataset(df)
    audit = audit_data_quality(df, profile)

    issues = [
        issue
        for issue in audit["issues"]
        if issue["type"] == "extreme_numeric_outliers"
    ]

    assert len(issues) == 1
    assert issues[0]["column"] == "value"


def test_detects_missingness_based_target_leakage():
    df = pd.DataFrame(
        {
            "feature": [
                None,
                None,
                None,
                None,
                None,
            ]
            * 10
            + [
                "observed",
                "observed",
                "observed",
                "observed",
                "observed",
            ]
            * 10,
            "target": [0] * 50 + [1] * 50,
        }
    )

    profile = profile_dataset(df)

    audit = audit_data_quality(
        df,
        profile,
        target="target",
    )

    leakage_issues = [
        issue for issue in audit["issues"] if issue["type"] == "possible_target_leakage"
    ]

    assert len(leakage_issues) >= 1
    assert leakage_issues[0]["column"] == "feature"

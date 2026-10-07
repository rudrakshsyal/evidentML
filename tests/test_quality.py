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

    issue_types = [
        issue["type"]
        for issue in audit["issues"]
    ]

    assert "duplicate_rows" in issue_types
    assert "constant_column" in issue_types
    assert "missing_values" in issue_types
import pandas as pd

from evidentml.data_engine.domain import validate_domain_rules


def test_numeric_domain_rule():
    df = pd.DataFrame(
        {
            "age": [22, 40, -5, 999],
        }
    )

    rules = {
        "age": {
            "min": 18,
            "max": 80,
        }
    }

    issues = validate_domain_rules(df, rules)

    assert len(issues) == 1
    assert issues[0]["column"] == "age"
    assert issues[0]["evidence"]["violation_count"] == 2
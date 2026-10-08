from evidentml.evidence.benchmark import score_audit


def test_score_audit():

    audit = {
        "issues": [
            {
                "type": "duplicate_rows",
                "column": None,
            },
            {
                "type": "constant_column",
                "column": "country",
            },
        ]
    }

    ground_truth = {
        "expected_findings": [
            {
                "type": "duplicate_rows",
                "column": None,
            },
            {
                "type": "constant_column",
                "column": "country",
            },
        ]
    }

    score = score_audit(
        audit,
        ground_truth,
    )

    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert score["f1"] == 1.0
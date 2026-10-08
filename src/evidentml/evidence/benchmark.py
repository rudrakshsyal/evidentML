from __future__ import annotations

from typing import Any


def extract_detected_issue_keys(audit: dict[str, Any],) -> set[tuple[str, str | None]]:
    """
    Convert audit findings into comparable (issue_type, column) pairs.
    """

    return {
        (
            issue["type"],
            issue.get("column"),
        )
        for issue in audit["issues"]
    }

def build_expected_issue_keys(ground_truth: dict[str, Any],) -> set[tuple[str, str | None]]:
    """
    Return the findings a competent EvidentML audit
    is expected to produce for this benchmark.
    """

    expected_findings = ground_truth.get(
        "expected_findings",
        [],
    )

    return {
        (
            finding["type"],
            finding.get("column"),
        )
        for finding in expected_findings
    }


def score_audit(audit: dict[str, Any],ground_truth: dict[str, Any],) -> dict[str, Any]:
    """
    Compare EvidentML findings against benchmark ground truth.
    """

    detected = extract_detected_issue_keys(audit)
    expected = build_expected_issue_keys(ground_truth)

    true_positives = detected & expected
    false_positives = detected - expected
    false_negatives = expected - detected

    precision = (
        len(true_positives) / len(detected)
        if detected
        else 0.0
    )

    recall = (
        len(true_positives) / len(expected)
        if expected
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "expected_issue_count": len(expected),
        "detected_issue_count": len(detected),
        "true_positive_count": len(true_positives),
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": sorted(
            true_positives,
            key=str,
        ),
        "false_positives": sorted(
            false_positives,
            key=str,
        ),
        "false_negatives": sorted(
            false_negatives,
            key=str,
        ),
    }
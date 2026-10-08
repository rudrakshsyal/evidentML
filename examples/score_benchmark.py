import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from evidentml.data_engine.profiler import profile_dataset
from evidentml.data_engine.quality import audit_data_quality
from evidentml.evidence.benchmark import score_audit

console = Console()

DATASET_PATH = (
    "benchmarks/datasets/"
    "binary_classification_messy_v1.csv"
)

GROUND_TRUTH_PATH = (
    "benchmarks/ground_truth/"
    "binary_classification_messy_v1.yaml"
)


df = pd.read_csv(DATASET_PATH)

with open(GROUND_TRUTH_PATH) as f:
    ground_truth = yaml.safe_load(f)


profile = profile_dataset(df)

audit = audit_data_quality(
    df=df,
    profile=profile,
    target="target",
)

score = score_audit(
    audit=audit,
    ground_truth=ground_truth,
)


console.print()
console.rule("[bold]EvidentML Benchmark Evaluation[/bold]")
console.print()

summary = Table(show_header=False)

summary.add_column("Metric")
summary.add_column("Value")

summary.add_row(
    "Expected issues",
    str(score["expected_issue_count"]),
)

summary.add_row(
    "Detected issues",
    str(score["detected_issue_count"]),
)

summary.add_row(
    "True positives",
    str(score["true_positive_count"]),
)

summary.add_row(
    "False positives",
    str(score["false_positive_count"]),
)

summary.add_row(
    "False negatives",
    str(score["false_negative_count"]),
)

summary.add_row(
    "Precision",
    f"{score['precision']:.1%}",
)

summary.add_row(
    "Recall",
    f"{score['recall']:.1%}",
)

summary.add_row(
    "F1",
    f"{score['f1']:.1%}",
)

console.print(summary)


if score["false_negatives"]:

    console.print()
    console.print("[bold]Missed expected issues[/bold]")

    for issue_type, column in score["false_negatives"]:
        console.print(
            f"  • {issue_type}: {column or 'dataset'}"
        )


if score["false_positives"]:

    console.print()
    console.print("[bold]Unexpected findings[/bold]")

    for issue_type, column in score["false_positives"]:
        console.print(
            f"  • {issue_type}: {column or 'dataset'}"
        )


console.print()
import pandas as pd
from rich.console import Console
from rich.table import Table

from evidentml.data_engine.cleaning import apply_safe_cleaning
from evidentml.data_engine.profiler import profile_dataset
from evidentml.data_engine.quality import audit_data_quality

console = Console()

df = pd.read_csv("benchmarks/datasets/binary_classification_messy_v1.csv")

profile = profile_dataset(df)

audit = audit_data_quality(
    df=df,
    profile=profile,
    target="target",
)

cleaned_df, lineage = apply_safe_cleaning(
    df=df,
    audit=audit,
)

console.rule("[bold]EvidentML Cleaning Report[/bold]")

console.print(
    f"\nRows: {len(df):,}   "
    f"Columns: {len(df.columns)}   "
    f"Transformations: {len(lineage)}\n"
)

table = Table(
    title="Applied Safe Transformations",
    show_lines=True,
)

table.add_column("Column")
table.add_column("Action")
table.add_column("Before")
table.add_column("After")
table.add_column("Confidence")

for item in lineage:

    before = (
        item.get("before_dtype")
        or str(item.get("before_unique", "—"))
    )

    after = (
        item.get("after_dtype")
        or str(item.get("after_unique", "—"))
    )

    table.add_row(
        item["column"],
        item["action"],
        before,
        after,
        item["confidence"],
    )


console.print(table)


console.print("\n[bold]Quick verification[/bold]")

console.print(
    "Income dtype:",
    df["income"].dtype,
    "→",
    cleaned_df["income"].dtype,
)

console.print(
    "State unique values:",
    df["state"].nunique(),
    "→",
    cleaned_df["state"].nunique(),
)

console.print(
    "Subscription plan unique values:",
    df["subscription_plan"].nunique(),
    "→",
    cleaned_df["subscription_plan"].nunique(),
)

cleaned_profile = profile_dataset(cleaned_df)

cleaned_audit = audit_data_quality(
    df=cleaned_df,
    profile=cleaned_profile,
    target="target",
)

before_issues = {
    (issue["type"], issue.get("column"))
    for issue in audit["issues"]
}

after_issues = {
    (issue["type"], issue.get("column"))
    for issue in cleaned_audit["issues"]
}

resolved = before_issues - after_issues
remaining = after_issues

console.print("\n[bold]Cleaning Effectiveness[/bold]")

console.print(
    f"Issues before: {len(before_issues)}"
)

console.print(
    f"Issues after: {len(after_issues)}"
)

console.print(
    f"Resolved: {len(resolved)}"
)

console.print("\n[bold]Resolved issues[/bold]")

for issue_type, column in sorted(resolved, key=str):
    console.print(
        f"  ✓ {issue_type}: {column or 'dataset'}"
    )

console.print("\n[bold]Remaining issues[/bold]")

for issue_type, column in sorted(remaining, key=str):
    console.print(
        f"  • {issue_type}: {column or 'dataset'}"
    )
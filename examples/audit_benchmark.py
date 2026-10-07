import pandas as pd

from rich.console import Console
from rich.table import Table

from evidentml.data_engine.profiler import profile_dataset
from evidentml.data_engine.quality import audit_data_quality


console = Console()

df = pd.read_csv(
    "benchmarks/datasets/binary_classification_messy_v1.csv"
)

profile = profile_dataset(df)

audit = audit_data_quality(
    df=df,
    profile=profile,
)


console.print()
console.rule("[bold]EvidentML Data Quality Audit[/bold]")
console.print()

console.print(
    f"[bold]Rows:[/bold] {profile['rows']:,}   "
    f"[bold]Columns:[/bold] {profile['columns']}   "
    f"[bold]Issues detected:[/bold] {audit['issue_count']}"
)

console.print()


table = Table(
    title="Detected Data Quality Issues",
    show_lines=True,
)

table.add_column("Severity")
table.add_column("Issue")
table.add_column("Column")
table.add_column("Finding")


severity_order = {
    "high": 0,
    "warning": 1,
    "info": 2,
}


issues = sorted(
    audit["issues"],
    key=lambda x: severity_order.get(
        x["severity"],
        99,
    ),
)


for issue in issues:

    table.add_row(
        issue["severity"].upper(),
        issue["type"],
        issue.get("column", "—"),
        issue["message"],
    )


console.print(table)
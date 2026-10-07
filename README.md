# EvidentML

Evidence-driven autonomous data science and ML engineering.

EvidentML aims to make data understanding, cleaning, and modeling decisions
explicit, testable, and reproducible. The v0.1 development scope is tabular binary
classification; customer/subscription fields provide one benchmark scenario,
not a restriction to churn prediction.

## Current status

- A seeded synthetic benchmark and its evaluation answer key are committed.
- `profile_dataset(df)` in `src/evidentml/data_engine/profiler.py` reports row and
  column counts, column names, and exact duplicate rows. One unit test covers
  dimensions and duplicate counting.
- Quality auditing, cleaning, and evidence schemas are placeholders. The planted
  benchmark issues are evaluation targets, not a claim that all are detected.

See [the benchmark guide](docs/benchmarks.md) for the issue inventory, actual
check coverage, limitations, and how to add regression cases.

## Local setup and benchmark reproduction

Use Python 3.12 and the committed `uv.lock` dependency versions. From the repository
root:

```bash
uv sync --locked --extra dev
uv run --locked pytest
uv run --locked python benchmarks/generators/generate_binary_classification_v1.py
```

The generator overwrites the benchmark CSV and YAML. Review their Git diff after
regeneration; this command is not a benchmark evaluation or a detector test.
`uv` manages Python dependencies and environments; `pyproject.toml` declares the
project requirements and `uv.lock` records resolved versions.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/evidentml/data_engine/` | Profiling, then quality auditing and cleaning |
| `src/evidentml/evidence/` | Planned evidence schemas for decisions |
| `benchmarks/generators/` | Reproducible synthetic data generation |
| `benchmarks/datasets/` | Inputs presented to the system under evaluation |
| `benchmarks/ground_truth/` | Evaluation-only answer keys |
| `tests/` | Software correctness and future regression tests |
| `docs/benchmarks.md` | Benchmark contract and coverage history |

## Next milestone

Extend the profiler with per-column types, missingness, cardinality, frequency
and numeric summaries, plus optional target distribution. Keep profiling
observational and non-mutating. Use these measurements to build the quality
auditor next, followed by cleaning and evidence evaluation. See the guide for
acceptance criteria.

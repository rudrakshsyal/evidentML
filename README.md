# EvidentML

**Evidence-driven autonomous data science and ML engineering.**

EvidentML's thesis is that useful automation must make data understanding,
cleaning, and modeling decisions explicit, testable, and reproducible. Agents
should reason over evidence, deterministic tools should execute operations, and
evaluations should judge whether those decisions improve outcomes.

## Current scope and status

v0.1 focuses on a **tabular binary classification benchmark**. Customer and
subscription fields provide the first scenario; the project is not limited to
churn prediction. The current working slice is dataset profiling and deterministic
data-quality auditing, with structured findings and a Rich terminal report.
Cleaning, benchmark scoring, modeling, and agent orchestration are not implemented.

### Completed in this development checkpoint

- Expanded the profiler from dataset dimensions to per-column structural and
  semantic observations, including support for pandas string dtypes. Numeric
  uniqueness alone no longer makes a column an identifier candidate.
- Built the deterministic auditor, including numeric-format and categorical
  checks, robust outliers, and optional binary-target leakage candidates.
- Added a readable Rich audit report and focused profiler/auditor tests.
- Exercised the committed benchmark to inspect detections and expose the
  distinction between statistical anomalies and domain-invalid values.

## Benchmark-first development

Synthetic data provides reproducible inputs, deliberately planted failure modes,
and an evaluation answer key without using customer records. It allows us to
measure known successes and misses before introducing agents. It does not prove
performance on real-world datasets.

The seed-42 generator starts with 20,000 records, injects issues, appends 120 exact
duplicates, and shuffles the data. The committed
[`binary_classification_messy_v1.csv`](benchmarks/datasets/binary_classification_messy_v1.csv)
has **20,120 rows, 15 columns, and approximately 12.26% positive targets**.
Planted cases include identifiers, target-dependent cancellation dates,
constant/near-constant features, noise, invalid ages, dirty categories, missing
values, mixed numeric/date formats, and extreme spending values.

The [ground-truth YAML](benchmarks/ground_truth/binary_classification_messy_v1.yaml)
is an evaluator-only answer key. “Hidden” means hidden from the system being
evaluated, not secret in Git: profiler, auditor, cleaner, and future agents should
receive data and task metadata without consulting it. A separate evaluator will
compare their outputs afterward. This boundary is a design requirement; no
scoring harness currently enforces it. The YAML is not a row-level corruption
ledger or a clean reference dataset.

See the [benchmark contract](docs/benchmarks.md) for injection details, limitations,
and how new failure modes become regression cases.

## Architecture and implemented capabilities

**Profiler observes; Auditor flags; Cleaner changes.**

```text
Dataset → Profiler → observations
Dataset + observations + optional target → Auditor → structured findings → Rich report
                                                     ↓ planned
                                              Evidence and scoring
                                                     ↓ planned
                                              Cleaning engine
```

### Profiler

`profile_dataset(df)` reports dimensions, column names, and exact duplicate counts.
Each column profile includes physical dtype, inferred semantic type, null count
and fraction, distinct count and ratio, dominant non-null value fraction, and a
constant flag. Numeric columns also include minimum, maximum, mean, and median.

Conservative semantic labels include boolean, numeric, datetime, datetime-like
strings, categorical, possible identifier, and unknown. Date-like detection uses
a sample of up to 500 non-null strings; high-uniqueness strings can be identifier
candidates. These are heuristics, not authoritative domain types. Profiling does
not modify the input or currently report target class distributions.

### Deterministic data-quality auditor

`audit_data_quality(df, profile, target=None)` returns an issue count and findings
with a type, severity, message, evidence, and column when applicable.

| Check | Current behavior |
| --- | --- |
| Exact duplicates | Counts repeated rows beyond the first occurrence |
| Constant columns | Flags at most one distinct non-null value, including all-null columns |
| Near-constant columns | Flags a dominant value covering at least 99.5% of non-null observations |
| Missingness | Reports any nulls; warning at 5%, high severity at 50% |
| Possible identifiers | Flags the profiler's identifier candidates for review |
| Mixed numeric formats | Flags string columns at least 80% parseable after trimming, removing dollar signs/commas, and expanding `k` suffixes |
| Dirty categorical variants | Detects distinct labels that collapse after trimming and case normalization |
| Robust numeric outliers | Uses median/MAD robust z-scores above 6 in absolute value; requires 20 non-null values and skips zero MAD |
| Possible target leakage | For a supplied binary target, checks near-direct target copies, target-associated missingness, and near-pure low-cardinality target mappings |

Leakage checks use a default 99.5% match/purity threshold, a 95-percentage-point
missingness gap, and minimum sample guards. Categorical mapping considers 2–50
unique values and requires 20 comparable rows overall, not per category. These
signals warrant investigation of feature availability at prediction time; they
are not proof of leakage. Class imbalance and sparse categories need further
validation. Missing or non-binary target columns skip leakage detection.

Numeric-format detection also flags uniformly numeric strings; it does not prove
that multiple formats occur. Categorical normalization does not infer that
`California` and `CA` are equivalent. Outlier detection does not enforce domain
rules: on the benchmark it flags 40 extreme ages and 46 spending values, while
negative ages still require semantic validation. No finding automatically
changes, imputes, or removes data.

### Terminal audit output

[`examples/audit_benchmark.py`](examples/audit_benchmark.py) loads the CSV, profiles
it, and audits it with `target="target"`. Rich renders dataset dimensions and issue
count, followed by a severity-sorted table of issue type, column, and finding.
The structured audit retains numeric evidence for future scoring and reporting.

## Tests and validation

The current suite contains **four passing tests**:

| Test | What it validates |
| --- | --- |
| `test_profile_dataset` | Dimensions, zero duplicates on its fixture, numeric dtype, identifier and categorical inference |
| `test_quality_audit_detects_basic_issues` | Duplicate, constant-column, and missing-value issue types |
| `test_detects_extreme_numeric_outlier` | An extreme value produces one outlier issue on the expected column in a 25-row fixture |
| `test_detects_missingness_based_target_leakage` | A feature whose presence tracks the binary target is flagged |

These are focused correctness checks, not comprehensive benchmark evaluation.
Direct-copy and categorical leakage paths, negative controls, threshold edges,
near-constant detection, numeric formatting, categorical variants, and profiler
edge cases still need dedicated coverage. The cleaning test file is a placeholder.
No benchmark precision/recall, cleaning benefit, or model performance is claimed.

## Local setup and usage

Use Python 3.12 and the committed dependency lock. From the repository root:

```bash
uv sync --locked --extra dev
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python examples/profile_benchmark.py
uv run --locked python examples/audit_benchmark.py
```

To regenerate the benchmark:

```bash
uv run --locked python benchmarks/generators/generate_binary_classification_v1.py
```

Regeneration overwrites the CSV and YAML; review their diff. It is not a detector
evaluation. Keep the generator revision and locked environment with any result.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/evidentml/data_engine/profiler.py` | Observational dataset and column profiles |
| `src/evidentml/data_engine/quality.py` | Deterministic checks and structured issues |
| `src/evidentml/data_engine/cleaning.py` | Planned cleaning engine; placeholder |
| `src/evidentml/evidence/` | Planned evidence schemas; placeholder |
| `examples/` | Profile and Rich audit entry points |
| `benchmarks/` | Seeded generator, input CSV, evaluator answer keys |
| `tests/` | Focused correctness checks |
| `docs/benchmarks.md` | Benchmark contract, inventory, and coverage gaps |

## Design principles

- **Agents reason; tools execute; evals judge.** Build measurable tools before
  agent orchestration; keep claims tied to observed results.
- Keep profiling, auditing, and mutation separate. Findings are evidence for a
  decision, not instructions to drop or overwrite data.
- Preserve structured evidence and reproducibility so decisions can be reviewed.
- Keep evaluation answers outside system inputs. Add regression cases and clean
  controls when new failure modes appear.
- Distinguish statistical outliers from semantic invalidity and association from
  leakage. Future cleaning should record justified changes and their effects.

## Next

1. Finish leakage validation and tests: direct copies, categorical mappings,
   negative controls, class imbalance, sparse groups, and target edge cases.
2. Add domain/semantic validity checks, starting with explicitly supplied age
   constraints and evidence-backed category equivalences.
3. Build benchmark scoring and the evidence engine to compare findings with
   evaluator-only expectations and report misses and false positives.
4. Begin the cleaning engine with explicit transformations, a change log, and
   before/after validation.

# v0.1 benchmark contract

## Purpose and reproducibility

`binary_classification_messy_v1` is the first benchmark for EvidentML v0.1.
Synthetic data gives us known failure modes and a repeatable evaluation input
without customer records. It lets contributors ask whether a change detects a
known issue and whether a proposed correction is justified. It does not establish
performance on real-world data.

The generator uses seed **42**, starts with **20,000** customer-like records,
appends **120** exact copies after injecting issues, and shuffles the result.
The committed CSV has **20,120 rows, 15 columns**, and a binary `target` with about
**12.26% positives**. The target is sampled from a logistic probability depending
on tenure, usage, support tickets, spend, plan, and age. This is a generic binary
classification task expressed through subscription-related fields.

Reproduce from the repository root using the commands in the [README](../README.md).
Keep the generator revision, Python version, and locked dependencies with any
reported result; a fixed seed alone is not a promise of byte-identical output
across different software versions.

## Files and responsibilities

| File | Role |
| --- | --- |
| `benchmarks/generators/generate_binary_classification_v1.py` | Creates the clean data in memory, injects issues, writes CSV and YAML |
| `benchmarks/datasets/binary_classification_messy_v1.csv` | Committed messy input; clean records are not separately saved |
| `benchmarks/ground_truth/binary_classification_messy_v1.yaml` | Expected issue families, affected columns, selected values/mappings, and modeling expectations |
| `benchmarks/ground_truth/churn_messy_v1.yaml` | Empty legacy scaffold; not an additional benchmark or answer key |
| `src/evidentml/data_engine/profiler.py` | Current structural measurements |
| `tests/test_profiler.py` | Small unit test for dimensions and duplicate counting |

Paths in this table are relative to the repository root. The generator is the
source for injection mechanics; the YAML is the evaluation contract. Changes to
either must be reviewed together.

## Planted cases

Counts below describe the original 20,000 rows **before duplication**. Copies can
increase affected-row counts, and different issue families can overlap.

| Case | How it is represented | Intended evaluation target (unless noted, not implemented) |
| --- | --- | --- |
| Identifier | `customer_id` is unique before duplication | Recognize an identifier candidate; avoid treating uniqueness alone as proof to drop it |
| Target leakage | `cancellation_date` exists exactly when `target == 1` | Flag perfect target-dependent missingness and investigate availability at prediction time |
| Constant | `country` is always `US` | Detect a constant feature |
| Near constant | `account_status` is `active` except for 40 `special` values | Report dominance and rare values |
| Noise | `random_noise` is sampled independently from a normal distribution | Assess utility through validation; randomness is not provable from a profile |
| Invalid numeric values | `age`: 40 values of `-5`, 40 of `999`; generated valid range is 18–80 | Flag domain violations when a domain constraint is supplied |
| Dirty state labels | 300 `CA` values become 100 each of `California`, `ca`, and ` CA ` | Identify equivalent categories; semantic mapping needs evidence |
| Patterned missing income | Up to 700 selected rows with `support_tickets >= 3` lose income | Measure missingness and its association with observed features |
| Mixed numeric formats | 300 nonmissing incomes become 150 dollar/comma strings and 150 `k` strings | Recognize numeric-like strings and explicit parsing needs |
| Mixed date formats | 400 signup dates become 200 `%m/%d/%Y` and 200 `%d-%b-%Y` strings | Identify multiple formats alongside ISO dates |
| Missing plan | 250 `subscription_plan` values become null | Report categorical missingness |
| Dirty plan labels | 150 remaining `Plus` values become 75 `plus` and 75 `PLUS`, if enough are available | Identify case variants |
| Extreme spend | 50 `monthly_spend` values are multiplied by 20 | Surface outlier candidates without automatically deleting them |
| Exact duplicates | 120 already-messy rows are copied, then all rows are shuffled | Count redundant rows beyond the first occurrence; **implemented** in the profiler |

For the committed CSV, income has **703** missing values and plan has **254**
after duplication. `cancellation_date` has **17,653** nulls by construction;
these are part of the leakage case, not ordinary missing-data corruption.

## What “hidden ground truth” means

The YAML is an answer key for a separate evaluator. It is visible in Git, so
“hidden” describes an evaluation boundary, not access control. Future profiler,
auditor, cleaner, and agent runs should receive the input dataset and declared
task metadata (such as the target), without reading the answer key. Compare
outputs against the YAML only afterward; otherwise the evaluation leaks its
answers. This separation is a requirement for the future harness, not a mechanism
currently enforced by the repository.

The YAML records expected issue categories and modeling requirements: a baseline,
a holdout set, leakage checks, and avoiding accuracy as the primary metric. These
are expectations, not completed modeling features or passing checks. The key is
not a row-level corruption ledger, a clean reference dataset, or a prescription
to remove every listed column.

## Actual coverage and limitations

At this stage, `profile_dataset(df)` returns `rows`, `columns`, `duplicate_rows`,
and `column_names`. Its unit test asserts dimensions and duplicate count on a
three-row fixture. No automated benchmark evaluator or tests asserting the full
planted inventory exist yet. `quality.py`, `cleaning.py`, and the evidence schemas
are empty. Generating a case does not demonstrate that EvidentML detects it.

- Coverage is one seeded synthetic CSV for binary classification. There is no
  demonstrated coverage for Parquet, regression, multiclass, forecasting,
  causal inference, multi-table data, or production distributions.
- There are no dedicated cases for schema drift, timezones, malformed dates,
  missing-value sentinel strings, near duplicates, conflicting labels, broader
  unit conversions, or unseen category behavior across train/test splits.
- Missing income depends on an observed feature. This exercises non-random
  missingness relative to rows, but does not prove a missing-not-at-random
  mechanism involving unobserved income.
- Currency and `k` formatting round income values, so exact original values
  cannot always be recovered. The answer key does not record affected row IDs
  or original values; correction accuracy cannot yet be scored precisely.
- The age range and category equivalences are benchmark assumptions. On unseen
  data they need domain context. Outliers, rare categories, identifiers, and
  apparently noisy features must not be silently removed based on one heuristic.
- Copies can leak across random train/test splits. Future modeling evaluation
  must keep duplicate records together or resolve them before splitting, and fit
  learned preprocessing only on training data.
- No detector precision/recall, false-positive rate, cleaning benefit, or model
  performance has been established. Add clean control cases and held-out
  scenarios before claiming generalization.

## Adding a case when unseen data exposes a failure

1. Record the observed failure, expected behavior, and whether it concerns
   detection, correction, or evaluation. Use a minimal synthetic or appropriately
   sanitized example; do not commit private source records.
2. Add a deterministic fixture or generator with a stable case ID and seed.
   Include a clean counterexample so the check must avoid false positives.
3. Define evaluator-only expectations: affected columns/rows, expected findings,
   justified actions or abstention, and acceptance thresholds where applicable.
   Include original values if correction fidelity is being measured.
4. Add a regression test that exposes the failure, then implement and verify the
   check. Keep the evaluator's answer key out of the system's inputs.
5. Preserve this v1 case for comparison. Use a new case ID or `v2` artifact when
   data or expected semantics change; document corrections to existing contracts
   explicitly. Update generator, dataset, answer key, tests, and this inventory
   together as applicable.
6. In the commit or PR, state the failure mode, added coverage, validation result,
   and remaining gaps. Mark coverage as implemented only after a passing test;
   retain failures as documented gaps rather than silently claiming support.

## Next step: extend `profiler.py`

Preserve the existing `profile_dataset(df)` behavior while adding JSON-serializable
per-column measurements: observed dtype, null count/fraction, distinct non-null
count, most-common values/frequencies, and numeric summaries for numeric columns.
Accept an optional target column for class counts and proportions. Report observed
types before any coercion; mixed income strings should remain visible.

Add focused tests for mixed types, missing and all-null columns, empty inputs,
constant/near-constant columns, target validation, serialization, and unchanged
input data. Exercise the committed benchmark to confirm dimensions, duplicates,
and missingness, without importing ground truth into the profiler. Defer domain
judgments, leakage decisions, normalization, imputation, and feature removal to
the subsequent auditor/cleaner stages, using the profile as evidence.

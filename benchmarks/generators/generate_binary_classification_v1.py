from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RANDOM_SEED = 42
N_ROWS = 20_000

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "benchmarks/datasets/binary_classification_messy_v1.csv"
GROUND_TRUTH_PATH = ROOT / "benchmarks/ground_truth/binary_classification_messy_v1.yaml"


def generate_clean_dataset(rng: np.random.Generator) -> pd.DataFrame:
    """Generate a realistic binary-classification dataset."""

    n = N_ROWS

    age = rng.integers(18, 81, n)
    tenure_months = rng.integers(0, 121, n)
    monthly_spend = np.maximum(rng.normal(120, 45, n), 5)
    usage_frequency = np.maximum(rng.normal(18, 7, n), 0)
    support_tickets = rng.poisson(1.5, n)

    states = rng.choice(
        ["CA", "NY", "TX", "FL", "WA"],
        size=n,
        p=[0.30, 0.18, 0.22, 0.18, 0.12],
    )

    plans = rng.choice(
        ["Basic", "Plus", "Premium"],
        size=n,
        p=[0.45, 0.35, 0.20],
    )

    income = np.maximum(rng.normal(85_000, 30_000, n), 20_000)

    signup_date = pd.Timestamp("2018-01-01") + pd.to_timedelta(
        rng.integers(0, 2500, n), unit="D"
    )

    logit = (
        -2.0
        - 0.018 * tenure_months
        - 0.025 * usage_frequency
        + 0.30 * support_tickets
        + 0.006 * monthly_spend
        + 0.35 * (plans == "Basic")
        + 0.25 * (age < 25)
    )

    probability = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:06d}" for i in range(n)],
            "age": age,
            "income": income.round(2),
            "state": states,
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend.round(2),
            "usage_frequency": usage_frequency.round(2),
            "support_tickets": support_tickets,
            "subscription_plan": plans,
            "signup_date": signup_date,
            "target": target,
        }
    )

    return df


def inject_data_quality_issues(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Plant known issues so EvidentML can be evaluated objectively."""

    df = df.copy()

    n = len(df)

    # 1. Leakage feature
    df["cancellation_date"] = pd.NaT
    positive_mask = df["target"] == 1
    df.loc[positive_mask, "cancellation_date"] = pd.Timestamp(
        "2026-01-01"
    ) + pd.to_timedelta(
        rng.integers(0, 180, positive_mask.sum()),
        unit="D",
    )

    # 2. Constant feature
    df["country"] = "US"

    # 3. Near-constant feature
    df["account_status"] = "active"
    rare_idx = rng.choice(df.index, size=40, replace=False)
    df.loc[rare_idx, "account_status"] = "special"

    # 4. Random noise feature
    df["random_noise"] = rng.normal(0, 1, n)

    # 5. Invalid age values
    bad_age_idx = rng.choice(df.index, size=80, replace=False)
    half = len(bad_age_idx) // 2
    df.loc[bad_age_idx[:half], "age"] = -5
    df.loc[bad_age_idx[half:], "age"] = 999

    # 6. Dirty state labels
    ca_idx = df.index[df["state"] == "CA"]
    dirty_ca_idx = rng.choice(ca_idx, size=min(300, len(ca_idx)), replace=False)

    df.loc[dirty_ca_idx[:100], "state"] = "California"
    df.loc[dirty_ca_idx[100:200], "state"] = "ca"
    df.loc[dirty_ca_idx[200:], "state"] = " CA "

    # 7. Missing income with non-random pattern
    high_ticket_mask = df["support_tickets"] >= 3
    candidate_idx = df.index[high_ticket_mask]

    missing_income_idx = rng.choice(
        candidate_idx,
        size=min(700, len(candidate_idx)),
        replace=False,
    )
    df.loc[missing_income_idx, "income"] = np.nan

    # 8. Mixed income formatting
    formatted_income_idx = rng.choice(
        df.index.difference(missing_income_idx),
        size=300,
        replace=False,
    )

    df["income"] = df["income"].astype(object)

    for idx in formatted_income_idx[:150]:
        value = float(df.at[idx, "income"])
        df.at[idx, "income"] = f"${value:,.0f}"

    for idx in formatted_income_idx[150:]:
        value = float(df.at[idx, "income"])
        df.at[idx, "income"] = f"{value / 1000:.0f}k"

    # 9. Multiple date formats
    df["signup_date"] = df["signup_date"].astype(str)

    date_idx = rng.choice(df.index, size=400, replace=False)

    for idx in date_idx[:200]:
        date = pd.Timestamp(df.at[idx, "signup_date"])
        df.at[idx, "signup_date"] = date.strftime("%m/%d/%Y")

    for idx in date_idx[200:]:
        date = pd.Timestamp(df.at[idx, "signup_date"])
        df.at[idx, "signup_date"] = date.strftime("%d-%b-%Y")

    # 10. Missing categorical values
    plan_missing_idx = rng.choice(df.index, size=250, replace=False)
    df.loc[plan_missing_idx, "subscription_plan"] = None

    # 11. Dirty plan categories
    plus_idx = df.index[df["subscription_plan"] == "Plus"]

    if len(plus_idx) >= 150:
        dirty_plus = rng.choice(plus_idx, size=150, replace=False)
        df.loc[dirty_plus[:75], "subscription_plan"] = "plus"
        df.loc[dirty_plus[75:], "subscription_plan"] = "PLUS"

    # 12. Extreme spend outliers
    outlier_idx = rng.choice(df.index, size=50, replace=False)
    df.loc[outlier_idx, "monthly_spend"] *= 20

    # 13. Duplicate rows
    duplicate_rows = df.sample(
        n=120,
        random_state=RANDOM_SEED,
    )

    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # 14. Shuffle rows
    df = df.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    return df


def write_ground_truth():
    """Create hidden benchmark answer key."""

    ground_truth = {
        "benchmark": "binary_classification_messy_v1",
        "task": {
            "type": "binary_classification",
            "target": "target",
        },
        "planted_issues": {
            "identifier_columns": ["customer_id"],
            "target_leakage": ["cancellation_date"],
            "constant_columns": ["country"],
            "near_constant_columns": ["account_status"],
            "noise_columns": ["random_noise"],
            "invalid_numeric_values": {
                "age": {
                    "expected_range": [18, 80],
                    "planted_values": [-5, 999],
                }
            },
            "dirty_categories": {
                "state": {
                    "CA": ["CA", "California", "ca", " CA "],
                },
                "subscription_plan": {
                    "Plus": ["Plus", "plus", "PLUS"],
                },
            },
            "mixed_numeric_formats": ["income"],
            "mixed_date_formats": ["signup_date"],
            "missing_values": [
                "income",
                "subscription_plan",
            ],
            "non_random_missingness": ["income"],
            "extreme_outliers": ["monthly_spend"],
            "duplicate_rows": 120,
        },
        "modeling_expectations": {
            "avoid_accuracy_as_primary_metric": True,
            "requires_baseline": True,
            "requires_holdout_set": True,
            "requires_leakage_check": True,
        },
    }

    with GROUND_TRUTH_PATH.open("w") as f:
        yaml.safe_dump(
            ground_truth,
            f,
            sort_keys=False,
        )


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    clean_df = generate_clean_dataset(rng)
    messy_df = inject_data_quality_issues(clean_df, rng)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)

    messy_df.to_csv(DATASET_PATH, index=False)
    write_ground_truth()

    print("Benchmark generated")
    print(f"Rows: {len(messy_df):,}")
    print(f"Columns: {len(messy_df.columns)}")
    print(f"Positive rate: {messy_df['target'].mean():.3f}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Ground truth: {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()

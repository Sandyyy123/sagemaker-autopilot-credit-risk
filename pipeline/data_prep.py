"""
Data loading, time-based splitting, and feature engineering.
Handles synthetic demo mode and real Plaid/PostgreSQL data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _make_synthetic_data(n=5000, seed=42):
    """Generate synthetic Plaid-like transaction features for demo mode."""
    rng = np.random.default_rng(seed)

    n_new = int(n * 0.35)  # ~35% new users (Plaid-only)
    n_ret = n - n_new

    def _user_block(n_users, is_new, default_rate):
        data = {
            # Account features
            "account_age_days": rng.integers(1, 30 if is_new else 730, n_users),
            "available_balance": rng.uniform(0, 2000 if is_new else 5000, n_users),
            "days_since_last_transaction": rng.integers(0, 90, n_users),
            # Transaction aggregates (30-day window)
            "tx_count_30d": rng.integers(0, 80, n_users),
            "tx_amount_mean_30d": rng.uniform(5, 500, n_users),
            "tx_amount_std_30d": rng.uniform(0, 300, n_users),
            "salary_detected": rng.integers(0, 2, n_users),
            "payroll_days_until_next": rng.integers(0, 31, n_users),
            "neobank_flag": rng.integers(0, 2, n_users),
            # Device / user attributes
            "device_os_ios": rng.integers(0, 2, n_users),
            "user_age": rng.integers(18, 65, n_users),
            # Advance history (returning only)
            "prior_advances": rng.integers(0, 1 if is_new else 15, n_users),
            "prior_default_rate": rng.uniform(0, 0.3 if is_new else 0.2, n_users),
            # Target
            "is_default": (rng.uniform(0, 1, n_users) < default_rate).astype(int),
            "user_segment": "new" if is_new else "returning",
        }
        # Plausible date range for temporal split
        base_date = pd.Timestamp("2024-01-01")
        data["application_date"] = [
            base_date + pd.Timedelta(days=int(rng.integers(0, 540)))
            for _ in range(n_users)
        ]
        return pd.DataFrame(data)

    df = pd.concat([
        _user_block(n_new, is_new=True, default_rate=0.15),
        _user_block(n_ret, is_new=False, default_rate=0.10),
    ], ignore_index=True)

    df = df.sort_values("application_date").reset_index(drop=True)
    return df


def load_and_split(mode="demo", s3_bucket="", test_ratio=0.20):
    """
    Load data and perform a time-based train/test split.
    - demo: uses synthetic data
    - full: reads from s3_bucket/data/advances.csv
    Returns (df_train, df_test).
    """
    if mode == "demo":
        df = _make_synthetic_data()
    else:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=s3_bucket, Key="data/advances.csv")
        df = pd.read_csv(obj["Body"], parse_dates=["application_date"])
        df = df.sort_values("application_date").reset_index(drop=True)

    cutoff_idx = int(len(df) * (1 - test_ratio))
    df_train = df.iloc[:cutoff_idx].copy()
    df_test = df.iloc[cutoff_idx:].copy()
    return df_train, df_test


FEATURE_COLS = [
    "account_age_days", "available_balance", "days_since_last_transaction",
    "tx_count_30d", "tx_amount_mean_30d", "tx_amount_std_30d",
    "salary_detected", "payroll_days_until_next", "neobank_flag",
    "device_os_ios", "user_age", "prior_advances", "prior_default_rate",
]

TARGET_COL = "is_default"


def build_features(df_train, df_test):
    """
    Extract feature matrices and target vectors.
    Applies StandardScaler fitted on train only (no leakage).
    Returns (X_train, y_train, X_test, y_test, feature_names).
    """
    X_train_raw = df_train[FEATURE_COLS].fillna(0).values
    X_test_raw = df_test[FEATURE_COLS].fillna(0).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    y_train = df_train[TARGET_COL].values
    y_test = df_test[TARGET_COL].values

    return X_train, y_train, X_test, y_test, FEATURE_COLS

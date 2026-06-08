"""
Data loading and validation for credit card fraud detection.

Dataset: European cardholders, September 2013 (Kaggle / ULB ML Group)
  - 284,807 transactions over 2 days
  - 492 fraudulent transactions  (rho = 0.00173)
  - Features: V1-V28 (PCA-transformed), Time, Amount
  - Target: Class in {0: legitimate, 1: fraudulent}

The extreme class imbalance (rho ~ 1.73e-3) means naive accuracy is
degenerate: a zero-predictor achieves 99.83% by never predicting fraud.
All modelling decisions downstream must account for this.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load and schema-validate the credit card transaction CSV.

    Args:
        path: Filesystem path to creditcard.csv.

    Returns:
        DataFrame with 31 columns: Time, V1-V28, Amount, Class.

    Raises:
        FileNotFoundError: Dataset file not found at the given path.
        AssertionError:    Schema or integrity check fails.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from Kaggle:\n"
            "  kaggle datasets download -d mlg-ulb/creditcardfraud"
        )

    df = pd.read_csv(csv_path)

    assert df.shape[1] == 31, f"Expected 31 columns, got {df.shape[1]}"
    assert "Class" in df.columns, "Target column 'Class' is missing"
    assert set(df["Class"].unique()) <= {0, 1}, "Class must be binary {0, 1}"

    n_null = int(df.isnull().sum().sum())
    assert n_null == 0, f"Dataset contains {n_null} null values"

    return df


def split_by_class(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partition transactions into legitimate (Class=0) and fraudulent (Class=1).

    Returns:
        (legit, fraud): Reset-indexed DataFrames for each class.
    """
    legit = df[df["Class"] == 0].reset_index(drop=True)
    fraud = df[df["Class"] == 1].reset_index(drop=True)
    return legit, fraud


def imbalance_ratio(legit: pd.DataFrame, fraud: pd.DataFrame) -> float:
    """
    Compute rho = |fraud| / |legit|.

    For this dataset rho ~ 0.00173, meaning the classifier sees roughly
    578 legitimate transactions for every single fraudulent one.
    """
    return len(fraud) / len(legit)


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a dictionary of key dataset statistics."""
    legit, fraud = split_by_class(df)
    rho = imbalance_ratio(legit, fraud)
    return {
        "total": len(df),
        "n_legit": len(legit),
        "n_fraud": len(fraud),
        "imbalance_ratio": rho,
        "fraud_pct": 100.0 * len(fraud) / len(df),
        "n_features": df.shape[1] - 1,
    }

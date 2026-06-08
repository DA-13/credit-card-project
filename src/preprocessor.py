"""
Feature preprocessing for credit card fraud detection.

V1-V28 are already PCA-transformed (zero mean, unit variance by construction).
Two columns require additional handling:

  Amount : raw monetary value in EUR. Standardized as
               A_tilde = (Amount - mu) / sigma
           where mu, sigma are estimated on the TRAINING set only to prevent
           data leakage into the test distribution.

  Time   : elapsed seconds from the first transaction in the window.
           Encodes temporal ordering of this specific 2-day window, not a
           generalizable signal. Dropped to prevent overfitting to a
           dataset-specific artifact.

Undersampling construction:
  Given minority set F (|F| = 492) and majority set L (|L| = 284315),
  the balanced dataset is:

      D_balanced = F  union  sample(L, |F|, without_replacement=True)

  This gives a balanced class prior P(y=1) = P(y=0) = 0.5 at training time,
  while the test set retains the natural imbalance for realistic evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


def drop_time(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the Time column; see module docstring for justification."""
    return df.drop(columns=["Time"], errors="ignore")


def prepare_features(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Full preprocessing pipeline: drop Time, standardize Amount, extract (X, y).

    Args:
        df:          Input DataFrame including the 'Class' column.
        scaler:      Pre-fitted StandardScaler. Required when fit_scaler=False.
        fit_scaler:  If True, fit a new scaler on df['Amount'].
                     If False, transform using the provided scaler.

    Returns:
        X      : Feature matrix in R^{n x 29}  (V1-V28 + NormalizedAmount)
        y      : Target vector in {0, 1}^n
        scaler : The fitted StandardScaler (new or provided)
    """
    df = drop_time(df)

    if fit_scaler:
        scaler = StandardScaler()
        df = df.copy()
        df["Amount"] = scaler.fit_transform(df[["Amount"]])
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when fit_scaler=False.")
        df = df.copy()
        df["Amount"] = scaler.transform(df[["Amount"]])

    X = df.drop(columns=["Class"]).values.astype(np.float64)
    y = df["Class"].values.astype(np.int32)
    return X, y, scaler


def undersample_balanced(
    legit: pd.DataFrame,
    fraud: pd.DataFrame,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Construct balanced dataset D = fraud union sample(legit, |fraud|).

    Sampling is performed without replacement, so each iteration in the
    Monte Carlo ensemble draws a distinct subset of the majority class,
    ensuring coverage of different regions of the legitimate-transaction
    manifold across iterations.

    Args:
        legit:        DataFrame of legitimate transactions (Class=0).
        fraud:        DataFrame of fraudulent transactions (Class=1).
        random_state: Seed for reproducibility across MC iterations.

    Returns:
        Balanced DataFrame with 2*|fraud| rows, shuffled index reset.
    """
    n_fraud = len(fraud)
    legit_sample = legit.sample(n=n_fraud, replace=False, random_state=random_state)
    balanced = pd.concat([legit_sample, fraud], axis=0)
    return balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

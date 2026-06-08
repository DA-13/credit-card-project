"""
Classification models for credit card fraud detection.

Two model families are provided, each suited to different aspects of the
fraud detection problem:

Random Forest
─────────────
An ensemble of T decorrelated decision trees. Each tree h_t is trained on
a bootstrap sample B_t subset X_train with a random feature subset of size
floor(sqrt(p)) at each split. The ensemble prediction is:

    f(x) = argmax_{c in {0,1}}  sum_{t=1}^{T}  1[h_t(x) = c]

Tree decorrelation reduces variance without increasing bias. Critical for
high-dimensional PCA feature spaces where individual trees may overfit.

Logistic Regression
───────────────────
Linear probabilistic classifier:

    P(y=1 | x) = sigma(w^T x + b)   where sigma(z) = 1 / (1 + e^{-z})

Trained by minimising the L2-regularised log-loss:

    L(w, b) = -sum_i [ y_i log p_hat_i + (1-y_i) log(1-p_hat_i) ]
              + (1/C) ||w||_2^2

Smaller C => stronger regularisation. Provides an interpretable linear
baseline and calibrated probability estimates useful for threshold tuning.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@dataclass
class FitResult:
    """Container for a trained model and all associated split data."""
    model: object
    X_train: np.ndarray
    X_test:  np.ndarray
    y_train: np.ndarray
    y_test:  np.ndarray
    y_train_pred: np.ndarray
    y_test_pred:  np.ndarray
    y_test_proba: np.ndarray


def build_random_forest(
    n_estimators: int = 100,
    max_depth: int | None = None,
    class_weight: str | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Instantiate a Random Forest with sensible defaults.

    n_jobs=-1 utilises all available cores. random_state ensures
    reproducibility across experiments.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
    )


def build_logistic_regression(
    C: float = 1.0,
    class_weight: str | None = None,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Instantiate a Logistic Regression classifier.

    solver='lbfgs' is preferred for small-to-medium datasets; it supports
    L2 regularisation and multi-class problems natively. max_iter=1000
    ensures convergence for the balanced undersampled datasets used here.
    """
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
    )


def fit_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 2,
) -> FitResult:
    """
    Stratified train/test split, fit, and predict.

    Stratification preserves the class ratio in both splits, which is
    critical for the small balanced datasets (984 rows) used in
    undersampling iterations — without it, the test set might contain
    zero fraud examples by chance.

    Args:
        model:        Unfitted sklearn-compatible estimator.
        X:            Feature matrix in R^{n x p}.
        y:            Binary target vector in {0,1}^n.
        test_size:    Fraction of data reserved for evaluation.
        random_state: Seed for the split RNG.

    Returns:
        FitResult with fitted model, split arrays, and predictions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    y_train_pred  = model.predict(X_train)
    y_test_pred   = model.predict(X_test)
    y_test_proba  = model.predict_proba(X_test)[:, 1]

    return FitResult(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        y_test_proba=y_test_proba,
    )

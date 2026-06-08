"""
Training pipelines for credit card fraud detection.

Three experimental conditions mirror the original notebook, extended with
a complete evaluation suite and the LR implementation bug corrected.

Experiment 1 — RF on full imbalanced dataset
─────────────────────────────────────────────
Baseline. The model trains on 284,807 transactions with rho ~ 0.0017.
High accuracy (>99.9%) is expected but misleading. Useful as a ceiling
estimate for recall (the model has access to all legitimate examples).

Experiment 2 — RF with Monte Carlo Undersampling
──────────────────────────────────────────────────
Algorithm:
    best_auc <- -inf
    for i = 1 to N_iter:
        S_i  <- sample(legit, |fraud|, seed=i)   // without replacement
        D_i  <- fraud union S_i                  // |D_i| = 2 * |fraud| = 984
        X, y <- prepare_features(D_i)
        M_i  <- train(RandomForest, X_train, y_train)
        s_i  <- AUC-ROC(M_i, X_test, y_test)
        if s_i > best_auc:
            best_auc <- s_i ; M* <- M_i
    return M*, metrics(M*)

With N_iter=500 iterations each drawing 492 of 284315 legitimate samples,
the expected number of distinct legitimate transactions seen across all
iterations is approx. |legit| * (1 - (1 - 492/284315)^500) ~ 219,000,
giving broad coverage of the majority class manifold.

Experiment 3 — Logistic Regression with Monte Carlo Undersampling
───────────────────────────────────────────────────────────────────
Identical procedure to Experiment 2, using LogisticRegression instead of
RandomForestClassifier. This was implemented incorrectly in the original
notebook (used RF for both). The correct linear model is used here.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from .data_loader import dataset_summary, load_dataset, split_by_class
from .evaluator import compute_metrics, format_metrics
from .models import FitResult, build_logistic_regression, build_random_forest, fit_model
from .preprocessor import prepare_features, undersample_balanced


def run_full_dataset_rf(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Experiment 1: Random Forest on the full imbalanced dataset.

    Returns:
        {
          'model':   fitted RandomForestClassifier,
          'metrics': dict from compute_metrics(),
          'scaler':  fitted StandardScaler,
        }
    """
    X, y, scaler = prepare_features(df, fit_scaler=True)
    result: FitResult = fit_model(build_random_forest(), X, y)
    metrics = compute_metrics(result.y_test, result.y_test_pred, result.y_test_proba)

    if verbose:
        print(format_metrics(metrics, "Random Forest — Full Dataset (Imbalanced Baseline)"))

    return {"model": result.model, "metrics": metrics, "scaler": scaler}


def run_mc_undersampling(
    legit: pd.DataFrame,
    fraud: pd.DataFrame,
    model_fn: Callable,
    n_iter: int = 500,
    test_size: float = 0.2,
    verbose: bool = True,
    label: str = "Model",
) -> Dict:
    """
    Monte Carlo undersampling: train n_iter models, return the best by AUC-ROC.

    Selection criterion is AUC-ROC because it is threshold-independent and
    rewards models that rank fraud transactions above legitimate ones globally,
    not just at the default 0.5 decision boundary.

    Args:
        legit:    DataFrame of legitimate transactions.
        fraud:    DataFrame of fraudulent transactions.
        model_fn: Zero-argument callable returning an unfitted estimator.
        n_iter:   Number of Monte Carlo iterations.
        test_size:Fraction of balanced dataset held out each iteration.
        verbose:  Print iteration progress and final result.
        label:    Display name for this model family.

    Returns:
        {
          'model':       best fitted estimator,
          'metrics':     metrics of the best model,
          'scaler':      scaler fitted to the best iteration's training data,
          'auc_history': List[float] of AUC-ROC per iteration,
        }
    """
    best_auc: float = -np.inf
    best_result: FitResult | None = None
    best_metrics: Dict | None = None
    best_scaler = None
    auc_history: List[float] = []

    for i in range(n_iter):
        balanced = undersample_balanced(legit, fraud, random_state=i)
        X, y, scaler = prepare_features(balanced, fit_scaler=True)
        result = fit_model(model_fn(), X, y, test_size=test_size, random_state=2)
        metrics = compute_metrics(result.y_test, result.y_test_pred, result.y_test_proba)

        auc_history.append(metrics["auc_roc"])

        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_result = result
            best_metrics = metrics
            best_scaler = scaler

        if verbose and (i + 1) % 100 == 0:
            print(f"  [{label}] Iter {i + 1:>4}/{n_iter} | best AUC-ROC: {best_auc:.4f}")

    if verbose:
        print(format_metrics(best_metrics, f"{label} — MC Undersampling (best of {n_iter} iters)"))

    return {
        "model":       best_result.model,
        "metrics":     best_metrics,
        "scaler":      best_scaler,
        "auc_history": auc_history,
    }


def run_all_experiments(
    dataset_path: str = "creditcard.csv",
    n_iter: int = 500,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Execute all three experimental conditions and return collected results.

    Args:
        dataset_path: Path to creditcard.csv.
        n_iter:       Monte Carlo iterations for undersampling experiments.
        verbose:      Stream progress to stdout.

    Returns:
        {
          'rf_full':        results for Experiment 1,
          'rf_undersample': results for Experiment 2,
          'lr_undersample': results for Experiment 3,
        }
    """
    df = load_dataset(dataset_path)

    if verbose:
        s = dataset_summary(df)
        print("\nDataset Summary")
        print(f"  Total transactions : {s['total']:>10,}")
        print(f"  Legitimate         : {s['n_legit']:>10,}")
        print(f"  Fraudulent         : {s['n_fraud']:>10,}")
        print(f"  Imbalance ratio rho: {s['imbalance_ratio']:>14.6f}")
        print(f"  Fraud percentage   : {s['fraud_pct']:>13.4f}%")

    legit, fraud = split_by_class(df)
    results: Dict[str, Dict] = {}

    if verbose:
        print("\n[1/3] Random Forest on full imbalanced dataset...")
    results["rf_full"] = run_full_dataset_rf(df, verbose=verbose)

    if verbose:
        print(f"\n[2/3] Random Forest with MC undersampling ({n_iter} iterations)...")
    results["rf_undersample"] = run_mc_undersampling(
        legit, fraud,
        model_fn=build_random_forest,
        n_iter=n_iter,
        verbose=verbose,
        label="Random Forest",
    )

    if verbose:
        print(f"\n[3/3] Logistic Regression with MC undersampling ({n_iter} iterations)...")
    results["lr_undersample"] = run_mc_undersampling(
        legit, fraud,
        model_fn=build_logistic_regression,
        n_iter=n_iter,
        verbose=verbose,
        label="Logistic Regression",
    )

    return results

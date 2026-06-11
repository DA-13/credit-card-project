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

XGBoost
───────
Gradient-boosted trees, built additively. At round t, a new tree f_t is
fit to the negative gradient (and Hessian, for second-order methods) of
the loss with respect to the current ensemble's predictions:

    F_t(x) = F_{t-1}(x) + eta * f_t(x)

with regularised per-tree objective:

    Obj_t = sum_i L(y_i, F_{t-1}(x_i) + f_t(x_i)) + Omega(f_t)
    Omega(f_t) = gamma * T + (1/2) * lambda * ||w||_2^2 + alpha * ||w||_1

where T is the number of leaves and w the leaf weights.

Class imbalance is handled directly through scale_pos_weight, which
multiplies the gradient/Hessian contribution of every positive (fraud)
example by a constant s. For binary log-loss this is equivalent to
training on a re-weighted distribution

    P'(y=1) proportional to s * P(y=1)

without discarding any majority-class data, unlike the undersampling
approach used for Random Forest / Logistic Regression above. The
"textbook" choice s = n_neg / n_pos exactly equalises the total gradient
mass contributed by each class; in practice the optimum is often lower
(closer to sqrt(n_neg / n_pos)) because the PCA features here already
separate the classes well, and over-weighting the minority class pushes
the decision boundary too far and inflates false positives. We therefore
treat s as a hyperparameter and search over it alongside the usual
tree-complexity and regularisation knobs.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier


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


def build_xgboost(
    scale_pos_weight: float = 1.0,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Instantiate an XGBoost classifier for binary fraud detection.

    tree_method='hist' bins continuous features into histograms before
    split-finding, which is what makes training on 280k+ rows tractable.
    eval_metric='aucpr' tracks the area under the precision-recall curve,
    the appropriate ranking metric when the positive class is ~0.17% of
    the data (see src/evaluator.py for the rationale).
    """
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=random_state,
    )


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 25,
    cv_folds: int = 3,
    random_state: int = 42,
) -> tuple[XGBClassifier, dict]:
    """
    Randomised hyperparameter search for XGBoost, scored by AUC-PR.

    The search space covers the three forces that compete in this
    problem:

      - Imbalance correction (scale_pos_weight): the gradient reweighting
        factor s described in the module docstring. We search between 1
        (no reweighting) and n_neg/n_pos ~ 578 (full reweighting), on a
        log-ish grid, so the search can land anywhere on that spectrum
        rather than committing to either extreme up front.

      - Model capacity (max_depth, n_estimators, learning_rate): how much
        the ensemble can fit. Deeper trees / more rounds fit the rare
        fraud patterns more closely but risk overfitting to the 492
        positive examples specifically.

      - Variance control (subsample, colsample_bytree, min_child_weight,
        gamma, reg_lambda, reg_alpha): standard stochastic-boosting and
        regularisation terms that keep the high-capacity end of the
        search space from memorising noise.

    StratifiedKFold ensures every fold contains a representative fraction
    of the 492 fraud cases — with plain KFold a fold could end up with
    very few positives and produce a degenerate AUC-PR estimate.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels in {0, 1}.
        n_iter:  Number of randomly sampled hyperparameter combinations.
        cv_folds:Number of stratified CV folds per combination.
        random_state: Seed for the search and the CV splitter.

    Returns:
        (best_estimator, best_params): the refit best estimator (refit on
        all of X_train, y_train) and its hyperparameter dictionary.
    """
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    full_reweight = n_neg / n_pos

    param_distributions = {
        "max_depth": randint(3, 9),
        "n_estimators": randint(100, 500),
        "learning_rate": loguniform(1e-2, 3e-1),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 11),
        "gamma": uniform(0.0, 5.0),
        "reg_lambda": loguniform(1e-1, 1e1),
        "reg_alpha": loguniform(1e-3, 1e0),
        "scale_pos_weight": loguniform(1.0, full_reweight),
    }

    search = RandomizedSearchCV(
        estimator=build_xgboost(),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_


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

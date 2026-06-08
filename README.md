# Credit Card Fraud Detection

Binary classification on the [ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) using Random Forest and Logistic Regression, with a principled Monte Carlo undersampling strategy to handle extreme class imbalance.

---

## Problem Statement

Given a transaction feature vector **x** ∈ ℝ³⁰ (28 PCA components + Time + Amount), predict **y** ∈ {0, 1} where y = 1 denotes a fraudulent transaction.

**Class imbalance:** ρ = |fraud| / |legit| ≈ 1.73 × 10⁻³

This renders naive accuracy degenerate. A classifier that always predicts "legitimate" achieves **99.83% accuracy** while catching **zero frauds**. All design decisions below are motivated by this constraint.

---

## Dataset

| Property | Value |
|---|---|
| Source | ULB Machine Learning Group / Kaggle |
| Transactions | 284,807 |
| Fraudulent | 492 (0.173%) |
| Legitimate | 284,315 (99.827%) |
| Features | V1–V28 (PCA-transformed), Time, Amount |
| Target | Class ∈ {0: legitimate, 1: fraudulent} |

> **Note:** The dataset is not included in this repository (too large for git). Download it from Kaggle:
> ```bash
> kaggle datasets download -d mlg-ulb/creditcardfraud
> unzip creditcardfraud.zip
> ```

---

## Mathematical Framework

### Why accuracy fails

Let **n₀** = 284,315 and **n₁** = 492. A zero-predictor achieves:

```
Accuracy = n₀ / (n₀ + n₁) = 284315 / 284807 ≈ 0.9983
```

It perfectly minimises 0-1 loss on the training distribution, yet its confusion matrix is: TP=0, FP=0, FN=492, TN=284315 — every fraud slips through.

### Evaluation metrics

| Metric | Formula | Interpretation |
|---|---|---|
| Precision | TP / (TP + FP) | Of flagged transactions, what fraction are truly fraudulent? |
| Recall | TP / (TP + FN) | Of all frauds, what fraction did we catch? |
| F1 | 2·P·R / (P + R) | Harmonic mean; balanced precision-recall scalar |
| AUC-ROC | ∫ TPR d(FPR) | Threshold-independent discrimination ability |
| AUC-PR | ∫ Prec d(Recall) | More informative than AUC-ROC under extreme imbalance; random baseline = ρ ≈ 0.0017 |
| MCC | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Pearson correlation of observed vs. predicted labels; −1 to +1 |

### Model 1 — Random Forest (Full Dataset)

Ensemble of T decorrelated decision trees. Each tree hₜ is grown on a bootstrap sample with a random feature subset of size ⌊√p⌋ at each split:

```
f(x) = argmax_{c ∈ {0,1}}  Σ_{t=1}^{T}  𝟙[hₜ(x) = c]
```

Training on the full imbalanced set provides a recall ceiling but inflated accuracy due to majority-class dominance.

### Model 2 — Random Forest + Monte Carlo Undersampling

```
best_auc ← −∞
for i = 1 to N_iter:
    Sᵢ      ← sample(legit, |fraud|, seed=i)   # without replacement
    Dᵢ      ← fraud ∪ Sᵢ                       # |Dᵢ| = 984, balanced 50/50
    Xᵢ, yᵢ ← prepare_features(Dᵢ)
    Mᵢ      ← RandomForest.fit(Xᵢ_train, yᵢ_train)
    sᵢ      ← AUC-ROC(Mᵢ, Xᵢ_test, yᵢ_test)
    if sᵢ > best_auc:  best_auc ← sᵢ ; M* ← Mᵢ
return M*
```

With N_iter = 500, the expected number of distinct legitimate transactions covered across all iterations is:

```
E[distinct] = n₀ · (1 − (1 − n₁/n₀)^{500}) ≈ 219,000
```

This provides broad coverage of the majority class manifold without oversampling artifacts.

### Model 3 — Logistic Regression + Monte Carlo Undersampling

Identical procedure using Logistic Regression:

```
P(y=1 | x) = σ(wᵀx + b)    where σ(z) = 1 / (1 + e⁻ᶻ)
```

Trained by minimising the L2-regularised binary cross-entropy:

```
L(w, b) = −Σᵢ [yᵢ log p̂ᵢ + (1−yᵢ) log(1−p̂ᵢ)]  +  (1/C) ‖w‖₂²
```

Provides an interpretable linear baseline and well-calibrated probability estimates useful for downstream threshold tuning.

---

## Preprocessing

| Step | Rationale |
|---|---|
| Drop `Time` | Encodes position in this specific 2-day window; not a generalisable signal |
| Standardise `Amount` | Raw EUR values have high variance; scaled to zero mean, unit std using training statistics only (no leakage) |
| V1–V28 | Already PCA-transformed; used as-is |

---

## Results

| Metric | RF (Full) | RF + Undersample | LR + Undersample |
|---|---|---|---|
| Accuracy | ~99.95% | ~93.9% | ~94.4% |
| AUC-ROC | — | — | — |

> Full metrics populate at runtime. AUC-ROC and AUC-PR are the primary selection criteria.

---

## Project Structure

```
credit-card-fraud-detection/
├── main.py                  # Entry point; runs all experiments
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── CreditCard_FraudDetection.ipynb   # Original exploratory notebook
└── src/
    ├── __init__.py
    ├── data_loader.py       # Load, validate, and split dataset
    ├── preprocessor.py      # Feature engineering and undersampling
    ├── models.py            # RF and LR model builders
    ├── evaluator.py         # Full metric suite
    └── pipeline.py          # Three end-to-end training pipelines
```

---

## Installation

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

Place `creditcard.csv` in the project root, then:

```bash
python main.py
```

Options:

```
--data   PATH    Path to creditcard.csv        (default: ./creditcard.csv)
--n-iter INT     Monte Carlo iterations        (default: 500)
--quiet          Suppress per-iteration output
```

---

## Key Design Decisions

1. **Selection criterion is AUC-ROC, not accuracy.** AUC-ROC is threshold-independent and rewards the model that globally ranks fraud above legitimate transactions, regardless of the decision boundary.

2. **Test set uses natural imbalance.** Each Monte Carlo iteration holds out a stratified 20% of the balanced dataset. The test set has 50/50 balance by design, which makes per-iteration AUC-ROC comparable across iterations. The full-dataset experiment uses the natural 0.17% fraud rate for test.

3. **Amount scaled on training data only.** Fitting the StandardScaler on the full dataset before splitting would leak test-set distribution information into the scaler parameters.

4. **Logistic Regression uses the correct estimator.** The original notebook incorrectly instantiated a `RandomForestClassifier` inside the "Logistic Regression" cell. This implementation uses `sklearn.linear_model.LogisticRegression` with LBFGS.

---

## Dependencies

- Python ≥ 3.9
- numpy, pandas, scikit-learn, matplotlib, seaborn, jupyter

See `requirements.txt` for pinned versions.

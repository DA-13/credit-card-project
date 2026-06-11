# Credit Card Fraud Detection

A small project where I built a model to detect fraudulent credit card transactions using the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset is really imbalanced - only 492 out of ~285k transactions are fraud (0.17%). So a model that just predicts "not fraud" every time gets 99.8% accuracy and is completely useless. That was the main thing I had to deal with.

## What I did

- Loaded and explored the data (no missing values, checked the class imbalance)
- Trained a Random Forest on the full dataset as a baseline
- Tried undersampling: took all 492 fraud cases, randomly sampled an equal number of legit transactions, repeated this 500 times and kept the best run, for both Random Forest and Logistic Regression
- Added XGBoost trained on the full dataset, using `scale_pos_weight` to handle the imbalance instead of throwing away data, and ran a randomized search (25 combos, 3-fold CV) over the main hyperparameters

## Results

Test set numbers (the undersampled models are tested on a small balanced split, the other two on the full natural imbalance):

| Metric    | RF (full) | RF (undersample) | LR (undersample) | XGBoost (tuned) |
|-----------|-----------|-------------------|-------------------|------------------|
| Accuracy  | 0.9995    | 0.9695            | 0.9645            | 0.9996           |
| Precision | 0.9176    | 0.9894            | 0.9505            | 0.9011           |
| Recall    | 0.7959    | 0.9490            | 0.9796            | 0.8367           |
| F1        | 0.8525    | 0.9688            | 0.9648            | 0.8677           |
| AUC-ROC   | 0.9630    | 0.9985            | 0.9969            | 0.9886           |
| AUC-PR    | 0.8810    | 0.9983            | 0.9970            | 0.8856           |
| MCC       | 0.8544    | 0.9398            | 0.9294            | 0.8681           |

Honestly the undersampled Random Forest came out on top here, which I wasn't expecting going in - I figured XGBoost with proper tuning would just win. My guess is the undersampled models get evaluated on a small 50/50 test split (only ~197 rows), so those numbers are a bit noisy/optimistic, while XGBoost is being tested on the full ~57k row imbalanced test set, which is a much harder and more realistic test. Still, XGBoost catches fraud with way fewer false positives on a much bigger test set (9 FP vs having to scale up the small undersampled numbers), so I wouldn't say it's "worse", just evaluated more honestly.

XGBoost's best hyperparameters from the search: `max_depth=6`, `n_estimators=483`, `learning_rate=0.054`, `scale_pos_weight=7.09` (so quite a bit lower than the "textbook" value of ~578, which matches the idea that over-correcting for imbalance hurts precision).

## Files

- `notebooks/CreditCard_FraudDetection.ipynb` - the original notebook where I worked through everything
- `src/` - same logic cleaned up and split into separate files
- `main.py` - runs everything end to end

## How to run

You'll need `creditcard.csv` from the Kaggle link above (too big for github). Put it in the project folder, then:

```
pip install -r requirements.txt
python main.py
```

## Things I'd improve

- Use a consistent test set across all models so the comparison is fair
- Try threshold tuning instead of the default 0.5 cutoff
- More hyperparameter search iterations for XGBoost (only did 25 to keep runtime sane)

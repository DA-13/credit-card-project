# Credit Card Fraud Detection

A small project where I built a model to detect fraudulent credit card transactions using the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset is really imbalanced - only 492 out of ~285k transactions are fraud (0.17%). So a model that just predicts "not fraud" every time gets 99.8% accuracy and is completely useless. That was the main thing I had to deal with.


- Loaded and explored the data (no missing values, checked the class imbalance)
- Trained a Random Forest on the full dataset as a baseline
- Tried undersampling: took all 492 fraud cases, randomly sampled an equal number of legit transactions, repeated this 500 times and kept the best run, for both Random Forest and Logistic Regression
- Added XGBoost trained on the full dataset, using `scale_pos_weight` to handle the imbalance instead of throwing away data, and ran a randomized search (25 combos, 3-fold CV) over the main hyperparameters


## Things I'd improve

- Use a consistent test set across all models so the comparison is fair
- Try threshold tuning instead of the default 0.5 cutoff
- More hyperparameter search iterations for XGBoost (only did 25 to keep runtime sane)

# Credit Card Fraud Detection

A small project where I built a model to detect fraudulent credit card transactions using the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset is really imbalanced - only 492 out of ~285k transactions are fraud (0.17%). So a model that just predicts "not fraud" every time gets 99.8% accuracy and is completely useless. That was the main thing I had to deal with.

## What I did

- Loaded and explored the data (no missing values, checked the class imbalance)
- Trained a Random Forest on the full dataset - got ~99.95% accuracy, but that number doesn't mean much given the imbalance
- Tried undersampling: took all 492 fraud cases, randomly sampled an equal number of legit transactions, repeated this 500 times and kept the best run. Got ~94% accuracy
- Did the same thing with Logistic Regression, got ~94.4%

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

- Use precision/recall/F1 instead of accuracy since accuracy is misleading with this much imbalance
- Try other models like XGBoost
- Tune hyperparameters - currently just using sklearn defaults

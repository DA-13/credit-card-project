"""
Credit Card Fraud Detection — entry point.

Usage
─────
  python main.py
  python main.py --data path/to/creditcard.csv --n-iter 200 --xgb-search-iter 25
  python main.py --quiet

The script trains four models and prints a side-by-side metric comparison.
Exit code 0 on success, 1 on dataset error.
"""

import argparse
import sys

from src.pipeline import run_all_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — RF, Logistic Regression & XGBoost"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="creditcard.csv",
        help="Path to creditcard.csv (default: ./creditcard.csv)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=500,
        help="Monte Carlo undersampling iterations (default: 500)",
    )
    parser.add_argument(
        "--xgb-search-iter",
        type=int,
        default=25,
        help="Randomised hyperparameter search iterations for XGBoost (default: 25)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-iteration progress output",
    )
    return parser.parse_args()


def print_comparison(results: dict) -> None:
    metrics_to_show = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "mcc"]
    col_w = 13
    keys = ("rf_full", "rf_undersample", "lr_undersample", "xgboost")
    headers = ("RF Full", "RF Under.", "LR Under.", "XGBoost")

    print("\n\n" + "=" * (18 + col_w * len(keys)))
    print("  Final Comparison")
    print("=" * (18 + col_w * len(keys)))
    print(f"{'Metric':<18}" + "".join(f"{h:>{col_w}}" for h in headers))
    print("-" * (18 + col_w * len(keys)))

    for m in metrics_to_show:
        row = f"{m:<18}"
        for key in keys:
            val = results[key]["metrics"].get(m, float("nan"))
            row += f" {val:>{col_w}.4f}"
        print(row)

    print(f"\nXGBoost best params: {results['xgboost']['best_params']}")


def main() -> int:
    args = parse_args()

    try:
        results = run_all_experiments(
            dataset_path=args.data,
            n_iter=args.n_iter,
            xgb_search_iter=args.xgb_search_iter,
            verbose=not args.quiet,
        )
        print_comparison(results)
        return 0

    except FileNotFoundError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

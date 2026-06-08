"""
Credit Card Fraud Detection — entry point.

Usage
─────
  python main.py
  python main.py --data path/to/creditcard.csv --n-iter 200
  python main.py --quiet

The script trains three models and prints a side-by-side metric comparison.
Exit code 0 on success, 1 on dataset error.
"""

import argparse
import sys

from src.pipeline import run_all_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — Random Forest & Logistic Regression"
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
        "--quiet",
        action="store_true",
        help="Suppress per-iteration progress output",
    )
    return parser.parse_args()


def print_comparison(results: dict) -> None:
    metrics_to_show = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "mcc"]
    col_w = 13

    print("\n\n" + "=" * 60)
    print("  Final Comparison")
    print("=" * 60)
    header = f"{'Metric':<18}" + "".join(
        f"{'RF Full':>{col_w}}"
        f"{'RF Under.':>{col_w}}"
        f"{'LR Under.':>{col_w}}"
    )
    print(f"{'Metric':<18} {'RF Full':>{col_w}} {'RF Under.':>{col_w}} {'LR Under.':>{col_w}}")
    print("-" * 60)

    for m in metrics_to_show:
        row = f"{m:<18}"
        for key in ("rf_full", "rf_undersample", "lr_undersample"):
            val = results[key]["metrics"].get(m, float("nan"))
            row += f" {val:>{col_w}.4f}"
        print(row)


def main() -> int:
    args = parse_args()

    try:
        results = run_all_experiments(
            dataset_path=args.data,
            n_iter=args.n_iter,
            verbose=not args.quiet,
        )
        print_comparison(results)
        return 0

    except FileNotFoundError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

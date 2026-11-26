"""Fit and audit a regressor, then save a reusable model artifact."""

import argparse
import csv
from pathlib import Path

from regression import read_csv, save_model, save_report, train_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/regressor.joblib"))
    parser.add_argument("--report", type=Path, default=Path("regression_metrics.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1 or args.seed < 0:
        parser.error("--jobs must be positive and --seed must be nonnegative")
    if len({args.train.resolve(), args.model.resolve(), args.report.resolve()}) != 3:
        parser.error("training data, model, and report paths must be distinct")
    try:
        x, y = read_csv(args.train, training=True)
        artifact, report = train_candidates(x, y, args.seed, args.jobs)
        save_model(args.model, artifact)
        save_report(args.report, report)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"{report['selected']}: audit RMSE {report['audit_rmse']:.6f} "
          f"(mean baseline {report['audit_baseline_rmse']:.6f})")


if __name__ == "__main__":
    main()

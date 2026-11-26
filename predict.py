"""Generate a one-column CSV with one prediction per input row."""

import argparse
import csv
from pathlib import Path

from regression import estimate, load_model, read_csv, save_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=Path("hidden_test.csv"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/regressor.joblib"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    if args.output.resolve() in {args.test.resolve(), args.model.resolve()}:
        parser.error("output must differ from inputs")
    try:
        x, _ = read_csv(args.test, training=False)
        artifact = load_model(args.model)
        save_predictions(args.output, estimate(artifact["model"], x), len(x))
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Wrote {len(x):,} predictions to {args.output}")


if __name__ == "__main__":
    main()

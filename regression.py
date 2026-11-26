"""Data contracts, candidates, and artifact handling for tabular regression."""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = tuple(str(i) for i in range(53))
TARGET = "target"
FORMAT_VERSION = 1
MISSING = {"", "na", "nan", "null"}


def read_csv(path: Path, *, training: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Read numeric data, align features by name, and validate the exact schema."""
    expected = FEATURES + (TARGET,) if training else FEATURES
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if len(header) != len(set(header)) or set(header) != set(expected):
            missing = sorted(set(expected) - set(header))
            unexpected = sorted(set(header) - set(expected))
            raise ValueError(f"{path}: invalid header (missing={missing}, unexpected={unexpected}, "
                             "column names must be unique)")
        index = [header.index(name) for name in expected]

        def values() -> Iterator[float]:
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"{path}:{reader.line_num}: expected {len(header)} fields, "
                                     f"got {len(row)}")
                for name, column in zip(expected, index):
                    value = row[column].strip()
                    if name != TARGET and value.casefold() in MISSING:
                        yield np.nan
                        continue
                    try:
                        number = float(value)
                    except ValueError as error:
                        raise ValueError(f"{path}:{reader.line_num}: {name} is not numeric: {value!r}") from error
                    if not math.isfinite(number):
                        raise ValueError(f"{path}:{reader.line_num}: {name} must be finite")
                    yield number

        matrix = np.fromiter(values(), dtype=np.float64).reshape(-1, len(expected))
    if training and len(matrix) < 30:
        raise ValueError(f"{path}: training requires at least 30 rows for the three-way split")
    return (np.ascontiguousarray(matrix[:, :-1]), matrix[:, -1]) if training else (matrix, None)


def split_indices(size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disjoint fit, model-selection, and final audit splits at 80/10/10."""
    shuffled = np.random.default_rng(seed).permutation(size)
    holdout = size // 10
    return shuffled[2 * holdout :], shuffled[:holdout], shuffled[holdout : 2 * holdout]


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the competition metric."""
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


class MagnitudeRemainderRegressor:
    """Regress on a signed-root feature's magnitude and a fractional feature."""

    @staticmethod
    def basis(x: np.ndarray) -> np.ndarray:
        return np.column_stack((np.rint(np.square(x[:, 6])), x[:, 7]))

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.imputer_ = SimpleImputer(strategy="mean", keep_empty_features=True)
        values = self.imputer_.fit_transform(self.basis(x))
        linear = LinearRegression().fit(values, y)
        parameters = np.r_[linear.intercept_, linear.coef_]
        integer = np.rint(parameters)
        # A rounding step is justified only if the learned coefficients are
        # already close to integers and those integers reproduce every fit label.
        integral_prediction = integer[0] + integer[1] * values[:, 0] + integer[2] * values[:, 1]
        if np.allclose(parameters, integer, rtol=0, atol=1e-10) and np.array_equal(
            integral_prediction, y
        ):
            parameters = integer
        self.intercept_, self.magnitude_coefficient_, self.remainder_coefficient_ = parameters
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = self.imputer_.transform(self.basis(x))
        return (self.intercept_ + self.magnitude_coefficient_ * values[:, 0]
                + self.remainder_coefficient_ * values[:, 1])


def candidate_models(seed: int, jobs: int) -> dict:
    """Build conservative and flexible candidates without fitting any data."""
    def trees(leaves: int, min_child: int, estimators: int):
        return LGBMRegressor(
            n_estimators=estimators, learning_rate=0.04, num_leaves=leaves,
            min_child_samples=min_child, colsample_bytree=0.8, reg_lambda=20.0,
            random_state=seed, n_jobs=jobs, force_col_wise=True, deterministic=True,
            verbosity=-1,
        )

    return {
        "mean": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(), Ridge(alpha=30_000),
        ),
        "binned_ridge": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            KBinsDiscretizer(n_bins=8, strategy="quantile", subsample=None),
            Ridge(alpha=750, solver="lsqr"),
        ),
        "lightgbm_small": trees(leaves=7, min_child=150, estimators=160),
        "lightgbm_flexible": trees(leaves=31, min_child=80, estimators=240),
        "magnitude_remainder": MagnitudeRemainderRegressor(),
    }


def estimate(model, x: np.ndarray) -> np.ndarray:
    """Run a fitted regressor and validate its output."""
    if len(x) == 0:
        return np.empty(0, dtype=np.float64)
    if isinstance(model, LGBMRegressor):
        return np.asarray(model.booster_.predict(x, num_threads=model.n_jobs), dtype=np.float64)
    return np.asarray(model.predict(x), dtype=np.float64)


def train_candidates(x: np.ndarray, y: np.ndarray, seed: int, jobs: int) -> tuple[dict, dict]:
    """Select on one split, audit on another, then refit on all labels."""
    fit, selection, audit = split_indices(len(y), seed)
    models = candidate_models(seed, jobs)
    selected = "mean"
    scores = {}
    models_fitted = {}
    with threadpool_limits(limits=jobs):
        for name, model in models.items():
            model.fit(x[fit], y[fit])
            models_fitted[name] = model
            prediction = estimate(model, x[selection])
            loss = np.square(y[selection] - prediction)
            if name == "mean":
                baseline_loss = loss
            improvement = baseline_loss - loss
            mean_gain = float(improvement.mean())
            standard_error = float(improvement.std(ddof=1) / np.sqrt(len(selection)))
            # One-sided, conservative screen for several candidate comparisons.
            lower_bound = mean_gain - 2.5 * standard_error
            scores[name] = {
                "selection_rmse": float(np.sqrt(loss.mean())),
                "mse_gain_over_mean": mean_gain,
                "mse_gain_lower_bound": lower_bound,
            }
            if lower_bound > 0 and scores[name]["selection_rmse"] < scores[selected]["selection_rmse"]:
                selected = name

        # Compute audit results after the selection decision is fixed.
        for name, model in models_fitted.items():
            scores[name]["audit_rmse"] = rmse(y[audit], estimate(model, x[audit]))

        final_model = candidate_models(seed, jobs)[selected]
        final_model.fit(x, y)

    artifact = {"version": FORMAT_VERSION, "features": FEATURES, "name": selected, "model": final_model}
    report = {
        "seed": seed,
        "rows": {"fit": len(fit), "selection": len(selection), "audit": len(audit), "final": len(y)},
        "selected": selected,
        "audit_rmse": scores[selected]["audit_rmse"],
        "audit_baseline_rmse": scores["mean"]["audit_rmse"],
        "candidates": scores,
    }
    if isinstance(final_model, MagnitudeRemainderRegressor):
        report["fitted_coefficients"] = {
            "intercept": float(final_model.intercept_),
            "rounded_feature_6_squared": float(final_model.magnitude_coefficient_),
            "feature_7": float(final_model.remainder_coefficient_),
        }
    return artifact, report


@contextmanager
def atomic_path(destination: Path) -> Iterator[Path]:
    """Stage a file beside its destination and rename it only after success."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                     suffix=".tmp", delete=False) as temp:
        path = Path(temp.name)
    try:
        yield path
        os.replace(path, destination)
    finally:
        path.unlink(missing_ok=True)


def save_model(path: Path, artifact: dict) -> None:
    with atomic_path(path) as temp:
        joblib.dump(artifact, temp, compress=3)


def load_model(path: Path) -> dict:
    """Load only artifacts produced by a trusted training run."""
    try:
        artifact = joblib.load(path)
    except (EOFError, pickle.UnpicklingError, KeyError, IndexError) as error:
        raise ValueError(f"{path}: unreadable regression artifact") from error
    if (not isinstance(artifact, dict) or artifact.get("version") != FORMAT_VERSION
            or artifact.get("features") != FEATURES or not hasattr(artifact.get("model"), "predict")):
        raise ValueError(f"{path}: unsupported regression artifact")
    return artifact


def save_report(path: Path, report: dict) -> None:
    with atomic_path(path) as temp:
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def save_predictions(path: Path, prediction: np.ndarray, expected_rows: int) -> None:
    if prediction.shape != (expected_rows,) or not np.isfinite(prediction).all():
        raise ValueError("prediction must contain one finite value per input row")
    with atomic_path(path) as temp:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow([TARGET])
            writer.writerows((format(float(value), ".17g"),) for value in prediction)

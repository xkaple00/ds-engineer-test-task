"""Train reference digit models and evaluate on the official MNIST test split."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from digit_classifier import cnn_features
from regression import atomic_path, save_report

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


def load_mnist(path: Path) -> tuple[np.ndarray, ...]:
    def verify(candidate):
        with candidate.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != MNIST_SHA256:
                raise ValueError("MNIST archive checksum does not match the official Keras file")

    if not path.is_file():
        with atomic_path(path) as temp:
            urllib.request.urlretrieve(MNIST_URL, temp)
            verify(temp)
    verify(path)
    with np.load(path, allow_pickle=False) as data:
        x, y, test, labels = (data[key] for key in ("x_train", "y_train", "x_test", "y_test"))
    expected = ((60000, 28, 28), (60000,), (10000, 28, 28), (10000,))
    if tuple(a.shape for a in (x, y, test, labels)) != expected:
        raise ValueError("expected the official MNIST train and test splits")
    if x.dtype != np.uint8 or test.dtype != np.uint8 or not all(
        np.issubdtype(a.dtype, np.integer) and a.min() >= 0 and a.max() <= 9 for a in (y, labels)
    ):
        raise ValueError("invalid MNIST image or label types")
    return x, y, test, labels


def train_cnn(x: np.ndarray, y: np.ndarray, epochs: int, seed: int) -> dict:
    """Optimize convolution and classifier weights with mini-batch Adam."""
    rng = np.random.default_rng(seed)
    filters = 16
    kernel = rng.normal(0, np.sqrt(2 / 25), size=(25, filters)).astype(np.float32)
    bias = np.zeros(filters, dtype=np.float32)
    dense = rng.normal(0, np.sqrt(2 / (144 * filters)), size=(144 * filters, 10)).astype(np.float32)
    output_bias = np.zeros(10, dtype=np.float32)
    parameters = (kernel, bias, dense, output_bias)
    first = [np.zeros_like(p) for p in parameters]
    second = [np.zeros_like(p) for p in parameters]
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        correct = 0
        for start in range(0, len(y), 96):
            idx = order[start : start + 96]
            image = x[idx].astype(np.float32) / 255
            pooled, patches, response = cnn_features(image, kernel, bias)
            logits = pooled @ dense + output_bias
            rows = np.arange(len(idx))
            labels = y[idx]
            correct += int(np.sum(logits.argmax(axis=1) == labels))
            gradient = np.exp(logits - logits.max(axis=1, keepdims=True))
            gradient /= gradient.sum(axis=1, keepdims=True)
            gradient[rows, labels] -= 1
            gradient /= len(idx)
            pool_gradient = (gradient @ dense.T).reshape(len(idx), 12, 12, filters)
            conv_gradient = pool_gradient.repeat(2, axis=1).repeat(2, axis=2).reshape(response.shape)
            conv_gradient *= (response > 0) * np.float32(0.25)
            updates = (
                patches.T @ conv_gradient,
                conv_gradient.sum(axis=0),
                pooled.T @ gradient,
                gradient.sum(axis=0),
            )
            step += 1
            rate = 0.0025 * (0.82 ** epoch)
            for p, m, v, update in zip(parameters, first, second, updates):
                m *= 0.9
                m += 0.1 * update
                v *= 0.999
                v += 0.001 * update ** 2
                p -= rate * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-7)
        print(f"CNN epoch {epoch + 1}/{epochs}: online accuracy {correct / len(y):.4f}", flush=True)
    return dict(zip(("kernel", "bias", "dense", "output_bias"), parameters))


def evaluate_cnn(weights: dict, images: np.ndarray, labels: np.ndarray) -> float:
    correct = 0
    for start in range(0, len(labels), 96):
        stop = start + 96
        pooled, _, _ = cnn_features(
            images[start:stop].astype(np.float32) / 255, weights["kernel"], weights["bias"]
        )
        scores = pooled @ weights["dense"] + weights["output_bias"]
        correct += int(np.sum(scores.argmax(axis=1) == labels[start:stop]))
    return correct / len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist", type=Path, default=Path("data/mnist.npz"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--report", type=Path, default=Path("digit_metrics.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cnn-rows", type=int, default=24_000)
    parser.add_argument("--rf-rows", type=int, default=24_000)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    if args.seed < 0 or args.jobs < 1 or args.epochs < 1 or not (
        100 <= min(args.cnn_rows, args.rf_rows) <= max(args.cnn_rows, args.rf_rows) <= 60_000
    ):
        parser.error("invalid seed, jobs, epochs or training rows (100 to 60000)")
    x, y, test, labels = load_mnist(args.mnist)
    order = np.random.default_rng(args.seed).permutation(len(y))
    cnn_idx, rf_idx = order[:args.cnn_rows], order[:args.rf_rows]
    rf = RandomForestClassifier(
        n_estimators=64, max_depth=19, min_samples_leaf=2, n_jobs=args.jobs,
        random_state=args.seed,
    )
    paths = [args.output / "cnn.npz", args.output / "rf.joblib", args.report]
    if len({path.resolve() for path in [args.mnist, *paths]}) != 4:
        parser.error("input and output paths must be distinct")
    with threadpool_limits(limits=args.jobs):
        weights = train_cnn(x[cnn_idx], y[cnn_idx], args.epochs, args.seed)
        cnn_accuracy = evaluate_cnn(weights, test, labels)
        print("Training random forest", flush=True)
        rf.fit(x[rf_idx].reshape(-1, 784).astype(np.float32) / 255, y[rf_idx])
        rf_accuracy = float(rf.score(test.reshape(-1, 784).astype(np.float32) / 255, labels))

    with atomic_path(paths[0]) as temp, temp.open("wb") as stream:
        np.savez_compressed(stream, **weights)
    rf.n_jobs = 1
    with atomic_path(paths[1]) as temp:
        joblib.dump(rf, temp, compress=3)
    report = {
        "seed": args.seed, "cnn_train_rows": args.cnn_rows, "rf_train_rows": args.rf_rows,
        "epochs": args.epochs, "mnist_test_rows": len(labels), "test_accuracy": {
            "cnn": cnn_accuracy, "rf": rf_accuracy,
            "rand": float(np.mean(np.random.default_rng(args.seed).integers(10, size=len(labels)) == labels)),
        },
    }
    save_report(paths[2], report)
    print(report)


if __name__ == "__main__":
    main()

"""Regression and end-to-end checks for the three assessment deliverables."""

import csv
import io
import random
from itertools import product
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np

from digit_classifier import DigitClassifier, DigitClassificationInterface
from islands import count_islands, read_grid
from regression import (FEATURES, MagnitudeRemainderRegressor, atomic_path, read_csv,
                        save_predictions, split_indices)

ROOT = Path(__file__).resolve().parents[1]


def flood_fill(grid):
    """Independent reference implementation for small grids."""
    remaining = {(i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if cell}
    count = 0
    while remaining:
        stack = [remaining.pop()]
        count += 1
        while stack:
            i, j = stack.pop()
            for neighbor in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def csv_file(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


class IslandsTests(unittest.TestCase):
    def test_given_examples(self):
        cases = [
            ([[0, 1, 0], [0, 0, 0], [0, 1, 1]], 2),
            ([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], 3),
            ([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]], 2),
        ]
        for grid, expected in cases:
            with self.subTest(grid=grid):
                self.assertEqual(count_islands(iter(grid)), expected)

    def test_every_4_by_4_grid(self):
        for cells in product((0, 1), repeat=16):
            grid = [cells[i : i + 4] for i in range(0, 16, 4)]
            self.assertEqual(count_islands(grid), flood_fill(grid), grid)

    def test_random_rectangles_match_the_oracle(self):
        rng = random.Random(2026)
        for _ in range(180):
            height, width = rng.randint(1, 55), rng.randint(1, 31)
            probability = rng.random()
            grid = [[int(rng.random() < probability) for _ in range(width)] for _ in range(height)]
            self.assertEqual(count_islands(iter(grid), width), flood_fill(grid))

    def test_empty_degenerate_and_large_stream(self):
        self.assertEqual(count_islands([]), 0)
        self.assertEqual(count_islands([[], []]), 0)
        self.assertEqual(count_islands(([1] for _ in range(30_000))), 1)
        self.assertEqual(count_islands([[1, 0] * 5000]), 5000)
        self.assertEqual(count_islands([[0] * 100] * 100), 0)
        self.assertEqual(count_islands([[1] * 100] * 100), 1)
        self.assertEqual(count_islands([[1, 0, 1, 0, 1], [1, 1, 1, 1, 1]]), 1)

    def test_invalid_input(self):
        for raw in ("", "3 4\n1 1 1\n", "-1 0\n", "1 2\n0 2\n",
                    "1 1\n1\n0\n", "1 1\nX\n", "1\n1\n", "1 2\n0\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                width, rows = read_grid(io.StringIO(raw))
                count_islands(rows, width)

    def test_command_line(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "islands.py")], input="2 2\n1 0\n0 1\n",
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), "2")


class RegressionTests(unittest.TestCase):
    def test_exact_magnitude_and_fraction_with_missing_feature_fallback(self):
        x = np.zeros((8, 53))
        x[:, 6] = np.sqrt(np.arange(8)) * np.array([1, -1, 1, -1, 1, -1, 1, -1])
        x[:, 7] = [0.1, 0.81, 0.41, 0.25, 0.99, 0.14, 0.27, 0.6]
        y = np.arange(8) + x[:, 7]
        model = MagnitudeRemainderRegressor().fit(x, y)
        np.testing.assert_array_equal(model.predict(x), y)
        example = x[:2].copy()
        example[0, 6] = np.nan
        example[1, 7] = np.nan
        np.testing.assert_allclose(model.predict(example), [3.5 + 0.1, 1 + x[:, 7].mean()])

    def test_split_is_disjoint_reproducible_and_complete(self):
        a = split_indices(103, 42)
        b = split_indices(103, 42)
        self.assertEqual(tuple(map(len, a)), (83, 10, 10))
        for first, second in zip(a, b):
            np.testing.assert_array_equal(first, second)
        self.assertEqual(set(np.concatenate(a)), set(range(103)))

    def test_load_reordered_columns_and_explicit_missing_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            csv_file(path, FEATURES[::-1], [[*range(52, 0, -1), "NA"]])
            x, y = read_csv(path, training=False)
            self.assertTrue(np.isnan(x[0, 0]))
            np.testing.assert_array_equal(x[0, 1:], np.arange(1, 53))
            self.assertIsNone(y)
            csv_file(path, FEATURES, [])
            self.assertEqual(read_csv(path, training=False)[0].shape, (0, 53))

    def test_malformed_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            cases = [
                (FEATURES[:-1], [[1] * 52], False),
                (FEATURES + ("0",), [[1] * 54], False),
                (FEATURES, [[1] * 52], False),
                (FEATURES, [[1] * 52 + ["Inf"]], False),
                (FEATURES, [[1] * 52 + ["oops"]], False),
                (FEATURES + ("target",), [[1] * 53 + ["nan"]] * 30, True),
                (FEATURES + ("target",), [[1] * 54], True),
            ]
            for header, rows, training in cases:
                with self.subTest(header=header[-3:], row=rows[0][-3:]), self.assertRaises(ValueError):
                    csv_file(path, header, rows)
                    read_csv(path, training=training)

    def test_atomic_output_preserves_existing_data_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            path.write_text("original", encoding="utf-8")
            with self.assertRaises(ValueError):
                save_predictions(path, np.array([np.inf]), 1)
            with self.assertRaises(RuntimeError):
                with atomic_path(path) as staged:
                    staged.write_text("bad", encoding="utf-8")
                    raise RuntimeError("interrupted write")
            self.assertEqual(path.read_text(encoding="utf-8"), "original")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_train_and_predict_clis_with_learnable_signal(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(500, 53))
        y = 20 * x[:, 0] + 2
        with tempfile.TemporaryDirectory() as folder:
            train, test, model, output, report = (Path(folder) / filename for filename in
                ("train.csv", "test.csv", "model.joblib", "predictions.csv", "report.json"))
            csv_file(train, FEATURES + ("target",), np.column_stack((x, y)))
            csv_file(test, FEATURES, x[:20])
            subprocess.run([
                sys.executable, str(ROOT / "train.py"), "--train", str(train),
                "--model", str(model), "--report", str(report), "--jobs", "1",
            ], check=True, capture_output=True, text=True)
            self.assertNotEqual(joblib.load(model)["name"], "mean")
            subprocess.run([
                sys.executable, str(ROOT / "predict.py"), "--test", str(test),
                "--model", str(model), "--output", str(output),
            ], check=True, capture_output=True, text=True)
            prediction = np.loadtxt(output, skiprows=1)
            self.assertEqual(prediction.shape, (20,))
            self.assertTrue(np.isfinite(prediction).all())
            self.assertGreater(np.corrcoef(y[:20], prediction)[0, 1], 0.7)
            csv_file(test, FEATURES, [])
            subprocess.run([
                sys.executable, str(ROOT / "predict.py"), "--test", str(test),
                "--model", str(model), "--output", str(output),
            ], check=True, capture_output=True, text=True)
            self.assertEqual(output.read_text(encoding="utf-8"), "target\n")
            model.write_bytes(b"")
            result = subprocess.run([
                sys.executable, str(ROOT / "predict.py"), "--test", str(test),
                "--model", str(model), "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("unreadable regression artifact", result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "target\n")


class RecordingModel(DigitClassificationInterface):
    def predict(self, image):
        self.image = image
        return 3


class DigitTests(unittest.TestCase):
    def test_each_actual_model_returns_a_python_digit(self):
        image = np.zeros((28, 28, 1), dtype=np.uint8)
        image[6:22, 12:16, 0] = 255
        for name in ("cnn", "rf", "rand"):
            with self.subTest(algorithm=name):
                result = DigitClassifier(name).predict(image)
                self.assertIs(type(result), int)
                self.assertIn(result, range(10))

    def test_common_shape_and_exact_internal_adapters(self):
        values = np.arange(784, dtype=np.float32).reshape(28, 28, 1) / 784
        for name, expected in (("cnn", values), ("rf", values.ravel()),
                               ("rand", values[9:19, 9:19, 0])):
            classifier = DigitClassifier(name)
            classifier.model = RecordingModel()
            self.assertEqual(classifier.predict(values), 3)
            np.testing.assert_array_equal(classifier.model.image, expected)

    def test_validation_and_training_stub(self):
        model = DigitClassifier("rand", seed=17)
        for shape in ((28, 28), (1, 28, 28, 1), (28, 28, 3), (0,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model.predict(np.zeros(shape))
        for value in (-1, 256, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model.predict(np.full((28, 28, 1), value))
        with self.assertRaises(TypeError):
            model.predict(np.full((28, 28, 1), "1"))
        with self.assertRaises(ValueError):
            DigitClassifier("svm")
        with self.assertRaises(NotImplementedError):
            model.train()

    def test_missing_or_invalid_model_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                DigitClassifier("cnn", model_path=root / "missing.npz")
            with self.assertRaises(FileNotFoundError):
                DigitClassifier("rf", model_path=root / "missing.joblib")
            np.savez(root / "broken.npz", kernel=np.zeros((25, 16)))
            with self.assertRaisesRegex(ValueError, "invalid CNN weights"):
                DigitClassifier("cnn", model_path=root / "broken.npz")
            joblib.dump({"n_features_in_": 10}, root / "broken.joblib")
            with self.assertRaisesRegex(ValueError, "invalid random forest model"):
                DigitClassifier("rf", model_path=root / "broken.joblib")

    def test_normalization_and_seed(self):
        image = np.full((28, 28, 1), 255, dtype=np.uint8)
        model = DigitClassifier("rand")
        model.model = RecordingModel()
        model.predict(image)
        np.testing.assert_array_equal(model.model.image, np.ones((10, 10)))
        first = DigitClassifier("rand", seed=17)
        second = DigitClassifier("rand", seed=17)
        self.assertEqual([first.predict(image) for _ in range(25)],
                         [second.predict(image) for _ in range(25)])


if __name__ == "__main__":
    unittest.main()

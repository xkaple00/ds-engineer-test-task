# Data Science Engineer test task

This repository contains a streaming island counter, a reproducible tabular regression workflow, and a single prediction interface for three MNIST models.

## Setup

Use Python 3.12. From the repository root, create an environment and install the pinned runtime dependencies.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Keep the supplied `train.csv` and `hidden_test.csv` outside Git. Put them in the project root or specify their paths on the command line. The MNIST models are provided under `models/`.

## 1. Count islands

```text
python islands.py
3 4
0 0 0 1
0 0 1 1
0 1 0 1
```

The program prints `2`. An island is a maximal set of land cells connected horizontally or vertically, as established by the diagonal test case in the brief. Rows are read from standard input. Call `count_islands(rows, columns)` for an iterator-based API that leaves its input intact.

The algorithm maintains only the component labels that touch the previous row. Each land run in the current row starts as a provisional island. Vertical overlaps create a bipartite graph of current runs and previous components. Traversing a connected component merges its vertices and decrements the total once per newly reached vertex. Previous components that touch no new run are already counted and drop out of memory. By induction, the running total counts the connected components after each row.

A row has at most `N` runs and at most `N` vertical overlaps. Each row therefore takes `O(N)` time and `O(N)` auxiliary space, for `O(MN)` total time on a nonempty `M × N` grid. The worst-case time bound is optimal. On an all-water grid, leaving any cell unexamined makes it impossible to distinguish that grid from one with a single island at that cell.

## 2. Tabular regression

```bash
python train.py --train /path/to/train.csv --jobs 4
python predict.py --test /path/to/hidden_test.csv --output predictions.csv
python -m pip install notebook
jupyter notebook eda.ipynb
```

The notebook reads the CSVs from the repository root by default. Set the `DS_TASK_DATA` environment variable to the directory containing both files if they live elsewhere.

- `eda.ipynb` contains the exploratory analysis, including target distribution, feature association, duplicate and missing-value checks, and hidden-feature distribution comparison.
- `train.py` splits the labeled data into fit, selection, and audit sets using a fixed seed. It compares the mean, ridge regression, binned ridge regression, two LightGBM configurations, and a fitted model using the magnitude and fractional features identified in EDA.
- Model selection requires a positive paired squared-error improvement over the mean after subtracting 2.5 standard errors. The audit outcomes are calculated after selection. The selected model family is refit on every labeled row before inference.
- `predict.py` writes a single `target` column in hidden-input row order, with one finite prediction per row. It accepts reordered feature columns and rejects missing, duplicate, or unexpected headers.
- `regression_metrics.json` records both RMSE splits for every candidate. Hidden-test RMSE is unavailable because the supplied file contains no labels.

## 3. MNIST inference

```python
import numpy as np
from digit_classifier import DigitClassifier

image = np.zeros((28, 28, 1), dtype=np.uint8)
image[5:23, 13:16, 0] = 255

for algorithm in ("cnn", "rf", "rand"):
    classifier = DigitClassifier(algorithm, seed=42)
    digit = classifier.predict(image)
    print(algorithm, digit)
```

`DigitClassifier` accepts finite grayscale pixel values in the required shape. Integer values use the `[0, 255]` scale. Floating-point values in `[0, 1]` are used directly, and floating-point values above 1 use the byte scale. The wrapper passes a normalized `28 × 28 × 1` image to the CNN, a 784-element vector to the random forest, or the centered `10 × 10` crop to the seeded or unseeded random baseline. Every backend implements `DigitClassificationInterface` and returns a single Python integer in `0–9`. The shared `train()` method raises `NotImplementedError`.

The CNN uses a learned 5 × 5 convolution, ReLU, average pooling, and a ten-class output layer implemented in NumPy. A separate training script reproduces the checked-in CNN and forest artifacts and evaluates them on the official 10,000-image test split.

```bash
python train_digits.py --jobs 4
# If the archive is already available locally:
python train_digits.py --mnist /path/to/mnist.npz --jobs 4
```

The training script uses the checksum-verified [MNIST distribution documented by Keras](https://keras.io/api/datasets/mnist/). The RF and regression artifacts use Joblib. Load only artifacts you trust.

## Verified results

| Check | Result |
|---|---|
| Islands | All 65,536 binary `4 × 4` grids matched an independent flood-fill oracle. Another 180 randomized rectangles and an iterative 30,000-row stream passed. |
| Regression | Held-out audit RMSE **0.000000** on 9,000 labeled rows, compared with **28.866642** for the mean and **0.194906** for the best tree candidate. |
| Hidden predictions | `predictions.csv` contains 10,000 finite values in input row order. Hidden labels were unavailable. |
| MNIST | On the official 10,000-image test split, CNN accuracy was **98.17%**, random forest accuracy was **95.79%**, and randomized baseline accuracy was **10.31%**. |
| Test suite | All 17 automated tests passed, including both regression CLIs and the three digit backends. |

The regression estimator learns weights for the rounded square of feature `6` and the fractional value in feature `7`. The exact fitted coefficients give `round(feature_6 ** 2) + feature_7`. The 90,000 supplied training records and the hidden feature ranges are consistent with the discovered relationship. The unusually exact association warrants verifying feature provenance before treating this as a production model.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests compare the island counter against an independent flood-fill reference on all 65,536 binary 4 × 4 grids, cover malformed and degenerate inputs, exercise both regression CLIs with a learnable synthetic dataset, and verify model input adapters and output contracts using the trained digit artifacts.

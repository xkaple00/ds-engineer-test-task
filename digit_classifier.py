"""Uniform MNIST inference interface for CNN, random forest, and random models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from zipfile import BadZipFile

import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"


class DigitClassificationInterface(ABC):
    """Interface for a digit model that accepts its own input representation."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """Return a digit from 0 through 9."""


def cnn_features(images, kernel, bias):
    """A learned 5-by-5 convolution, ReLU, and 2-by-2 average pooling."""
    windows = sliding_window_view(images, (5, 5), axis=(1, 2))
    patches = np.ascontiguousarray(windows).reshape(-1, 25)
    response = patches @ kernel + bias
    maps = np.maximum(response, 0).reshape(len(images), 12, 2, 12, 2, len(bias))
    pooled = maps.mean(axis=(2, 4)).reshape(len(images), -1)
    return pooled, patches, response


class CNN(DigitClassificationInterface):
    """Predict from a normalized 28×28×1 image with trained NumPy weights."""

    def __init__(self, model_path: Path):
        try:
            with np.load(model_path, allow_pickle=False) as artifact:
                self.kernel = artifact["kernel"].astype(np.float32)
                self.bias = artifact["bias"].astype(np.float32)
                self.dense = artifact["dense"].astype(np.float32)
                self.output_bias = artifact["output_bias"].astype(np.float32)
        except (BadZipFile, EOFError, KeyError, TypeError) as error:
            raise ValueError("invalid CNN weights") from error
        channels = len(self.bias) if self.bias.ndim == 1 else 0
        weights = (self.kernel, self.bias, self.dense, self.output_bias)
        expected = ((25, channels), (channels,), (144 * channels, 10), (10,))
        if not channels or any(w.shape != shape or not np.isfinite(w).all()
                               for w, shape in zip(weights, expected)):
            raise ValueError("invalid CNN weights")

    def predict(self, image: np.ndarray) -> int:
        pooled, _, _ = cnn_features(image[..., 0][None, ...], self.kernel, self.bias)
        return int(np.argmax(pooled @ self.dense + self.output_bias))


class RandomForest(DigitClassificationInterface):
    """Predict from a flat 784-element normalized image."""

    def __init__(self, model_path: Path):
        # Joblib is executable serialization. Load only trusted model files.
        self.model = joblib.load(model_path)
        if getattr(self.model, "n_features_in_", None) != 784 or not callable(
            getattr(self.model, "predict", None)
        ):
            raise ValueError("invalid random forest model")

    def predict(self, image: np.ndarray) -> int:
        return int(self.model.predict(image.reshape(1, 784))[0])


class RandomModel(DigitClassificationInterface):
    """Use the input shape required by the random baseline."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def predict(self, image: np.ndarray) -> int:
        if image.shape != (10, 10):
            raise ValueError("the random model expects a 10×10 center crop")
        return int(self.rng.integers(10))


class DigitClassifier:
    """Accept a 28×28×1 image and return an int for every algorithm."""

    def __init__(self, algorithm: str, *, model_path: Path | None = None,
                 seed: int | None = None):
        if algorithm == "cnn":
            self.model = CNN(model_path if model_path is not None else MODEL_DIRECTORY / "cnn.npz")
        elif algorithm == "rf":
            self.model = RandomForest(model_path if model_path is not None else MODEL_DIRECTORY / "rf.joblib")
        elif algorithm == "rand":
            self.model = RandomModel(seed)
        else:
            raise ValueError("algorithm must be one of: cnn, rf, rand")
        self.algorithm = algorithm

    def predict(self, image: np.ndarray) -> int:
        """Validate and normalize the image, then adapt it to the model."""
        pixels = np.asarray(image)
        if pixels.shape != (28, 28, 1):
            raise ValueError(f"expected a 28×28×1 image, got {pixels.shape}")
        floating = np.issubdtype(pixels.dtype, np.floating)
        integer = np.issubdtype(pixels.dtype, np.integer)
        binary = np.issubdtype(pixels.dtype, np.bool_)
        if not (floating or integer or binary):
            raise TypeError("image must contain real numeric or Boolean pixels")
        lower, upper = pixels.min(), pixels.max()
        if not np.isfinite(pixels).all() or lower < 0 or upper > 255:
            raise ValueError("pixels must be finite and in [0, 255]")
        normalized = pixels.astype(np.float32)
        if integer or (floating and upper > 1):
            normalized /= 255

        if self.algorithm == "rf":
            normalized = normalized.reshape(784)
        elif self.algorithm == "rand":
            normalized = normalized[9:19, 9:19, 0]
        result = self.model.predict(normalized)
        if not isinstance(result, (int, np.integer)) or not 0 <= result <= 9:
            raise ValueError("model returned an invalid digit")
        return int(result)

    def train(self, *args, **kwargs):
        """Training is deliberately outside the shared inference interface."""
        raise NotImplementedError("use train_digits.py to regenerate the model artifacts")

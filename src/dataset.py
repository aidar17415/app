"""
Dataset utilities for the flying-object classification project.

Features used (all numeric, normalised before training):
    0  speed_kmh       – speed in km/h
    1  altitude_m      – altitude in metres
    2  size_m          – approximate size (wingspan / body length) in metres
    3  aspect_ratio    – width-to-height ratio
    4  flapping        – 1.0 if flapping motion detected, else 0.0
    5  rotor_sound     – rotor/propeller sound intensity  (0-1)
    6  heat_signature  – infrared heat signature           (0-1)

Class index → label mapping:
    0 → bird
    1 → drone
    2 → airplane
    3 → helicopter
"""

from __future__ import annotations

import numpy as np

from src.model import CLASSES, NUM_FEATURES

# ---------------------------------------------------------------------------
# Typical feature ranges used for synthetic data generation
# ---------------------------------------------------------------------------

# Each entry: (speed_kmh, altitude_m, size_m, aspect_ratio,
#              flapping, rotor_sound, heat_signature)
_CLASS_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "bird": {
        "speed_kmh":      (20.0,  80.0),
        "altitude_m":     (10.0, 500.0),
        "size_m":         (0.1,   1.5),
        "aspect_ratio":   (3.0,   8.0),
        "flapping":       (0.7,   1.0),
        "rotor_sound":    (0.0,   0.1),
        "heat_signature": (0.3,   0.6),
    },
    "drone": {
        "speed_kmh":      (10.0,  80.0),
        "altitude_m":     (5.0,  300.0),
        "size_m":         (0.2,   1.0),
        "aspect_ratio":   (0.8,   1.5),
        "flapping":       (0.0,   0.0),
        "rotor_sound":    (0.5,   0.9),
        "heat_signature": (0.1,   0.3),
    },
    "airplane": {
        "speed_kmh":      (200.0, 900.0),
        "altitude_m":     (1000.0, 12000.0),
        "size_m":         (10.0,  80.0),
        "aspect_ratio":   (6.0,  12.0),
        "flapping":       (0.0,   0.0),
        "rotor_sound":    (0.0,   0.1),
        "heat_signature": (0.6,   1.0),
    },
    "helicopter": {
        "speed_kmh":      (50.0,  300.0),
        "altitude_m":     (50.0, 3000.0),
        "size_m":         (5.0,  20.0),
        "aspect_ratio":   (0.5,   2.0),
        "flapping":       (0.0,   0.0),
        "rotor_sound":    (0.6,   1.0),
        "heat_signature": (0.5,   0.9),
    },
}


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_samples_per_class: int = 200,
    random_seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic labelled dataset of flying-object observations.

    Args:
        n_samples_per_class: Number of samples to generate per class.
        random_seed:         Seed for reproducibility.

    Returns:
        X: (N, NUM_FEATURES) float64 feature matrix (unnormalised).
        y: (N,) int64 label array with values in {0, 1, 2, 3}.
    """
    rng = np.random.default_rng(random_seed)
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for class_idx, class_name in enumerate(CLASSES):
        profile = _CLASS_PROFILES[class_name]
        samples = np.zeros((n_samples_per_class, NUM_FEATURES))

        feature_order = [
            "speed_kmh", "altitude_m", "size_m", "aspect_ratio",
            "flapping", "rotor_sound", "heat_signature",
        ]
        for feat_idx, feat_name in enumerate(feature_order):
            lo, hi = profile[feat_name]
            if lo == hi:  # constant feature (e.g. flapping=0 for machines)
                samples[:, feat_idx] = lo
            else:
                samples[:, feat_idx] = rng.uniform(lo, hi, n_samples_per_class)

        # Add small Gaussian noise to non-binary features
        noise_mask = np.array([True, True, True, True, False, True, True])
        noise_std = (samples[:, noise_mask].max(axis=0)
                     - samples[:, noise_mask].min(axis=0)) * 0.05
        samples[:, noise_mask] += rng.normal(0, noise_std, (n_samples_per_class, noise_mask.sum()))

        X_parts.append(samples)
        y_parts.append(np.full(n_samples_per_class, class_idx, dtype=np.int64))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Normalisation (z-score)
# ---------------------------------------------------------------------------

class FeatureNormalizer:
    """Z-score normaliser computed on training data and applied to any split.

    Usage::

        normaliser = FeatureNormalizer()
        X_train_norm = normaliser.fit_transform(X_train)
        X_test_norm  = normaliser.transform(X_test)
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        """Compute mean and std from X.

        Args:
            X: (N, NUM_FEATURES) training feature matrix.

        Returns:
            self
        """
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Avoid division by zero for constant features
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalisation.

        Args:
            X: (N, NUM_FEATURES) feature matrix.

        Returns:
            Normalised feature matrix with the same shape.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureNormalizer must be fitted before transform.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: (N, NUM_FEATURES) training feature matrix.

        Returns:
            Normalised feature matrix.
        """
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into random train and test subsets.

    Args:
        X:           Feature matrix.
        y:           Label array.
        test_size:   Fraction of data used for the test split (default 0.2).
        random_seed: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    rng = np.random.default_rng(random_seed)
    n = len(y)
    indices = rng.permutation(n)
    n_test = int(n * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def label_to_index(label: str) -> int:
    """Convert a class-name string to its integer index.

    Args:
        label: One of 'bird', 'drone', 'airplane', 'helicopter'.

    Returns:
        Integer index (0-3).

    Raises:
        ValueError: If the label is not recognised.
    """
    if label not in CLASSES:
        raise ValueError(
            f"Unknown label '{label}'. Must be one of {CLASSES}."
        )
    return CLASSES.index(label)


def index_to_label(index: int) -> str:
    """Convert an integer class index to its human-readable name.

    Args:
        index: Integer in [0, NUM_CLASSES).

    Returns:
        Class-name string.

    Raises:
        IndexError: If the index is out of range.
    """
    if index < 0 or index >= len(CLASSES):
        raise IndexError(
            f"Class index {index} is out of range [0, {len(CLASSES)})."
        )
    return CLASSES[index]

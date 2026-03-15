"""
Unit tests for src/dataset.py
"""

import unittest
import numpy as np

from src.dataset import (
    FeatureNormalizer,
    generate_dataset,
    index_to_label,
    label_to_index,
    train_test_split,
)
from src.model import CLASSES, NUM_CLASSES, NUM_FEATURES


class TestGenerateDataset(unittest.TestCase):
    def test_output_shapes(self):
        n = 50
        X, y = generate_dataset(n_samples_per_class=n, random_seed=0)
        self.assertEqual(X.shape, (n * NUM_CLASSES, NUM_FEATURES))
        self.assertEqual(y.shape, (n * NUM_CLASSES,))

    def test_label_values(self):
        X, y = generate_dataset(n_samples_per_class=20, random_seed=1)
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y < NUM_CLASSES))

    def test_all_classes_present(self):
        X, y = generate_dataset(n_samples_per_class=10, random_seed=2)
        self.assertEqual(set(y.tolist()), set(range(NUM_CLASSES)))

    def test_reproducibility(self):
        X1, y1 = generate_dataset(n_samples_per_class=10, random_seed=99)
        X2, y2 = generate_dataset(n_samples_per_class=10, random_seed=99)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, _ = generate_dataset(n_samples_per_class=20, random_seed=1)
        X2, _ = generate_dataset(n_samples_per_class=20, random_seed=2)
        self.assertFalse(np.array_equal(X1, X2))


class TestFeatureNormalizer(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.uniform(0, 100, (100, NUM_FEATURES))

    def test_fit_transform_zero_mean(self):
        norm = FeatureNormalizer()
        X_n = norm.fit_transform(self.X)
        np.testing.assert_allclose(X_n.mean(axis=0), np.zeros(NUM_FEATURES), atol=1e-10)

    def test_fit_transform_unit_std(self):
        norm = FeatureNormalizer()
        X_n = norm.fit_transform(self.X)
        np.testing.assert_allclose(X_n.std(axis=0), np.ones(NUM_FEATURES), atol=1e-10)

    def test_transform_before_fit_raises(self):
        norm = FeatureNormalizer()
        with self.assertRaises(RuntimeError):
            norm.transform(self.X)

    def test_constant_feature_no_division_by_zero(self):
        X = self.X.copy()
        X[:, 0] = 5.0  # constant column
        norm = FeatureNormalizer()
        X_n = norm.fit_transform(X)
        self.assertTrue(np.all(np.isfinite(X_n)))

    def test_fit_then_transform_consistent(self):
        norm = FeatureNormalizer()
        X_a = norm.fit_transform(self.X)
        X_b = norm.transform(self.X)
        np.testing.assert_array_equal(X_a, X_b)


class TestTrainTestSplit(unittest.TestCase):
    def test_sizes(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, NUM_FEATURES))
        y = rng.integers(0, NUM_CLASSES, 100)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        self.assertEqual(len(X_tr), 80)
        self.assertEqual(len(X_te), 20)

    def test_no_overlap(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, NUM_FEATURES))
        y = np.arange(50)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        self.assertEqual(len(set(y_tr.tolist()) & set(y_te.tolist())), 0)

    def test_reproducibility(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, NUM_FEATURES))
        y = rng.integers(0, 4, 40)
        split1 = train_test_split(X, y, random_seed=7)
        split2 = train_test_split(X, y, random_seed=7)
        for a, b in zip(split1, split2):
            np.testing.assert_array_equal(a, b)


class TestLabelHelpers(unittest.TestCase):
    def test_label_to_index_known(self):
        for i, cls in enumerate(CLASSES):
            self.assertEqual(label_to_index(cls), i)

    def test_label_to_index_unknown_raises(self):
        with self.assertRaises(ValueError):
            label_to_index("spaceship")

    def test_index_to_label_known(self):
        for i, cls in enumerate(CLASSES):
            self.assertEqual(index_to_label(i), cls)

    def test_index_to_label_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            index_to_label(NUM_CLASSES)

    def test_index_to_label_negative_raises(self):
        with self.assertRaises(IndexError):
            index_to_label(-1)

    def test_round_trip(self):
        for cls in CLASSES:
            self.assertEqual(index_to_label(label_to_index(cls)), cls)

"""
Unit tests for src/model.py
"""

import unittest
import numpy as np

from src.model import (
    CLASSES,
    NUM_CLASSES,
    NUM_FEATURES,
    FlyingObjectClassifier,
    cross_entropy_loss,
    relu,
    relu_derivative,
    softmax,
)


class TestActivationFunctions(unittest.TestCase):
    def test_relu_positive(self):
        x = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(relu(x), x)

    def test_relu_negative(self):
        x = np.array([[-1.0, -2.0, 0.0]])
        expected = np.array([[0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(relu(x), expected)

    def test_relu_derivative_positive(self):
        x = np.array([[1.0, 2.0]])
        np.testing.assert_array_equal(relu_derivative(x), np.ones_like(x))

    def test_relu_derivative_negative(self):
        x = np.array([[-1.0, -0.5]])
        np.testing.assert_array_equal(relu_derivative(x), np.zeros_like(x))

    def test_softmax_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = softmax(x)
        self.assertAlmostEqual(float(result.sum()), 1.0, places=10)

    def test_softmax_monotone(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = softmax(x)[0]
        for i in range(len(result) - 1):
            self.assertLess(result[i], result[i + 1])

    def test_softmax_numerical_stability(self):
        # Large values should not produce nan/inf
        x = np.array([[1000.0, 1000.0, 1000.0]])
        result = softmax(x)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_softmax_batch(self):
        x = np.random.default_rng(0).normal(size=(5, 4))
        result = softmax(x)
        np.testing.assert_allclose(result.sum(axis=1), np.ones(5), atol=1e-12)


class TestCrossEntropyLoss(unittest.TestCase):
    def test_perfect_prediction(self):
        probs = np.array([[1.0, 0.0, 0.0, 0.0]])
        labels = np.array([0])
        loss = cross_entropy_loss(probs, labels)
        self.assertAlmostEqual(loss, 0.0, places=5)

    def test_uniform_prediction(self):
        probs = np.full((1, NUM_CLASSES), 1.0 / NUM_CLASSES)
        labels = np.array([0])
        expected = np.log(NUM_CLASSES)
        self.assertAlmostEqual(cross_entropy_loss(probs, labels), expected, places=5)

    def test_loss_nonnegative(self):
        rng = np.random.default_rng(1)
        raw = rng.uniform(size=(10, NUM_CLASSES))
        probs = raw / raw.sum(axis=1, keepdims=True)
        labels = rng.integers(0, NUM_CLASSES, size=10)
        self.assertGreaterEqual(cross_entropy_loss(probs, labels), 0.0)


class TestFlyingObjectClassifier(unittest.TestCase):
    def _make_model(self):
        return FlyingObjectClassifier(hidden_sizes=[16, 8], random_seed=0)

    def test_forward_output_shape(self):
        model = self._make_model()
        X = np.random.default_rng(0).normal(size=(10, NUM_FEATURES))
        probs = model.forward(X)
        self.assertEqual(probs.shape, (10, NUM_CLASSES))

    def test_forward_probabilities_sum_to_one(self):
        model = self._make_model()
        X = np.random.default_rng(1).normal(size=(5, NUM_FEATURES))
        probs = model.forward(X)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-10)

    def test_predict_returns_valid_indices(self):
        model = self._make_model()
        X = np.random.default_rng(2).normal(size=(20, NUM_FEATURES))
        preds = model.predict(X)
        self.assertEqual(preds.shape, (20,))
        self.assertTrue(np.all(preds >= 0))
        self.assertTrue(np.all(preds < NUM_CLASSES))

    def test_predict_labels_returns_strings(self):
        model = self._make_model()
        X = np.random.default_rng(3).normal(size=(4, NUM_FEATURES))
        labels = model.predict_labels(X)
        self.assertEqual(len(labels), 4)
        for label in labels:
            self.assertIn(label, CLASSES)

    def test_score_range(self):
        model = self._make_model()
        X = np.random.default_rng(4).normal(size=(30, NUM_FEATURES))
        y = np.random.default_rng(4).integers(0, NUM_CLASSES, 30)
        acc = model.score(X, y)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_loss_decreases_after_training(self):
        """Training should reduce loss on the training data."""
        model = self._make_model()
        rng = np.random.default_rng(5)
        X = rng.normal(size=(80, NUM_FEATURES))
        y = rng.integers(0, NUM_CLASSES, 80)

        probs_before = model.forward(X)
        loss_before = cross_entropy_loss(probs_before, y)

        model.fit(X, y, epochs=100, batch_size=16, verbose=False)

        probs_after = model.forward(X)
        loss_after = cross_entropy_loss(probs_after, y)

        self.assertLess(loss_after, loss_before)

    def test_save_and_load(self):
        """Saved and reloaded model must produce identical predictions."""
        import tempfile, os
        model = self._make_model()
        rng = np.random.default_rng(6)
        X = rng.normal(size=(10, NUM_FEATURES))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model")
            model.save(path)
            npz_path = path + ".npz"
            self.assertTrue(os.path.exists(npz_path))

            model2 = self._make_model()
            model2.load(npz_path)

            np.testing.assert_array_equal(model.predict(X), model2.predict(X))

    def test_predict_proba_consistent_with_predict(self):
        model = self._make_model()
        X = np.random.default_rng(7).normal(size=(15, NUM_FEATURES))
        probs = model.predict_proba(X)
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, np.argmax(probs, axis=1))


class TestConstants(unittest.TestCase):
    def test_class_names(self):
        self.assertEqual(CLASSES, ["bird", "drone", "airplane", "helicopter"])

    def test_num_classes(self):
        self.assertEqual(NUM_CLASSES, 4)

    def test_num_features(self):
        self.assertEqual(NUM_FEATURES, 7)

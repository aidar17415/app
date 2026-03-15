"""
Integration tests for the full training and prediction pipeline.
"""

import os
import tempfile
import unittest

import numpy as np

from src.dataset import FeatureNormalizer, generate_dataset, train_test_split
from src.model import CLASSES, FlyingObjectClassifier
from src.predict import load_artifacts, predict_single
from src.train import train


class TestTrainPipeline(unittest.TestCase):
    """End-to-end: generate data → train → evaluate."""

    def test_train_returns_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = train(
                n_samples_per_class=40,
                hidden_sizes=[16, 8],
                learning_rate=0.01,
                epochs=50,
                batch_size=16,
                model_path=os.path.join(tmpdir, "model"),
                normalizer_path=os.path.join(tmpdir, "normalizer"),
                verbose=False,
            )
        self.assertIn("train_accuracy", metrics)
        self.assertIn("test_accuracy", metrics)
        self.assertGreaterEqual(metrics["train_accuracy"], 0.0)
        self.assertLessEqual(metrics["train_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["test_accuracy"], 0.0)
        self.assertLessEqual(metrics["test_accuracy"], 1.0)

    def test_trained_model_accuracy(self):
        """After sufficient training, accuracy should be well above random (25%)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = train(
                n_samples_per_class=100,
                hidden_sizes=[64, 32],
                learning_rate=0.01,
                epochs=200,
                batch_size=32,
                model_path=os.path.join(tmpdir, "model"),
                normalizer_path=os.path.join(tmpdir, "normalizer"),
                verbose=False,
            )
        self.assertGreater(metrics["test_accuracy"], 0.5)

    def test_artifacts_saved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model")
            normalizer_path = os.path.join(tmpdir, "normalizer")
            train(
                n_samples_per_class=20,
                hidden_sizes=[8],
                epochs=10,
                model_path=model_path,
                normalizer_path=normalizer_path,
                verbose=False,
            )
            self.assertTrue(os.path.exists(model_path + ".npz"))
            self.assertTrue(os.path.exists(normalizer_path + ".npz"))


class TestPredictPipeline(unittest.TestCase):
    """Test load_artifacts and predict_single."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.model_path = os.path.join(cls.tmpdir, "model")
        cls.normalizer_path = os.path.join(cls.tmpdir, "normalizer")
        train(
            n_samples_per_class=100,
            hidden_sizes=[64, 32],
            learning_rate=0.01,
            epochs=200,
            batch_size=32,
            model_path=cls.model_path,
            normalizer_path=cls.normalizer_path,
            verbose=False,
        )
        cls.model, cls.normalizer = load_artifacts(
            cls.model_path + ".npz",
            cls.normalizer_path + ".npz",
        )

    def test_predict_single_returns_valid_class(self):
        result = predict_single(
            speed_kmh=50.0, altitude_m=200.0, size_m=0.5, aspect_ratio=5.0,
            flapping=0.9, rotor_sound=0.0, heat_signature=0.4,
            model=self.model, normalizer=self.normalizer,
        )
        self.assertIn(result["predicted_class"], CLASSES)

    def test_predict_single_probabilities_sum_to_one(self):
        result = predict_single(
            speed_kmh=500.0, altitude_m=8000.0, size_m=40.0, aspect_ratio=9.0,
            flapping=0.0, rotor_sound=0.05, heat_signature=0.8,
            model=self.model, normalizer=self.normalizer,
        )
        total = sum(result["probabilities"].values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_airplane_features_predict_airplane(self):
        """Typical airplane features should be classified as airplane."""
        result = predict_single(
            speed_kmh=700.0, altitude_m=10000.0, size_m=60.0, aspect_ratio=10.0,
            flapping=0.0, rotor_sound=0.02, heat_signature=0.9,
            model=self.model, normalizer=self.normalizer,
        )
        self.assertEqual(result["predicted_class"], "airplane")

    def test_bird_features_predict_bird(self):
        """Typical bird features should be classified as bird."""
        result = predict_single(
            speed_kmh=40.0, altitude_m=100.0, size_m=0.6, aspect_ratio=6.0,
            flapping=0.95, rotor_sound=0.0, heat_signature=0.45,
            model=self.model, normalizer=self.normalizer,
        )
        self.assertEqual(result["predicted_class"], "bird")

    def test_drone_features_predict_drone(self):
        """Typical drone features should be classified as drone."""
        result = predict_single(
            speed_kmh=30.0, altitude_m=100.0, size_m=0.4, aspect_ratio=1.0,
            flapping=0.0, rotor_sound=0.8, heat_signature=0.15,
            model=self.model, normalizer=self.normalizer,
        )
        self.assertEqual(result["predicted_class"], "drone")

    def test_helicopter_features_predict_helicopter(self):
        """Typical helicopter features should be classified as helicopter."""
        result = predict_single(
            speed_kmh=150.0, altitude_m=500.0, size_m=12.0, aspect_ratio=1.2,
            flapping=0.0, rotor_sound=0.85, heat_signature=0.75,
            model=self.model, normalizer=self.normalizer,
        )
        self.assertEqual(result["predicted_class"], "helicopter")

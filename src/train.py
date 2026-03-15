"""
Training pipeline for the flying-object classifier.

Usage (from the repository root)::

    python -m src.train

The trained model weights are saved to ``model.npz`` and the normaliser
statistics to ``normalizer.npz``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.dataset import FeatureNormalizer, generate_dataset, train_test_split
from src.model import FlyingObjectClassifier


def train(
    n_samples_per_class: int = 200,
    hidden_sizes: list[int] | None = None,
    learning_rate: float = 0.01,
    epochs: int = 300,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_seed: int = 42,
    model_path: str = "model.npz",
    normalizer_path: str = "normalizer.npz",
    verbose: bool = True,
) -> dict[str, float]:
    """Train the model and save artifacts to disk.

    Args:
        n_samples_per_class: Synthetic samples per class.
        hidden_sizes:        Hidden-layer sizes (default [64, 32]).
        learning_rate:       Gradient-descent step size.
        epochs:              Number of training epochs.
        batch_size:          Mini-batch size.
        test_size:           Fraction of data for evaluation.
        random_seed:         Global random seed.
        model_path:          Where to save model weights (.npz).
        normalizer_path:     Where to save normaliser stats (.npz).
        verbose:             Print progress during training.

    Returns:
        Dict with 'train_accuracy' and 'test_accuracy'.
    """
    if hidden_sizes is None:
        hidden_sizes = [64, 32]

    if verbose:
        print("Generating dataset …")
    X, y = generate_dataset(n_samples_per_class, random_seed=random_seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_seed=random_seed
    )

    normalizer = FeatureNormalizer()
    X_train_n = normalizer.fit_transform(X_train)
    X_test_n = normalizer.transform(X_test)

    if verbose:
        print(
            f"Train size: {len(y_train)}, Test size: {len(y_test)}, "
            f"Features: {X_train_n.shape[1]}"
        )

    model = FlyingObjectClassifier(
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        random_seed=random_seed,
    )

    if verbose:
        print("Training …")
    model.fit(X_train_n, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    train_acc = model.score(X_train_n, y_train)
    test_acc = model.score(X_test_n, y_test)

    if verbose:
        print(f"\nFinal train accuracy : {train_acc:.4f}")
        print(f"Final test  accuracy : {test_acc:.4f}")

    model.save(model_path)
    np.savez(
        normalizer_path,
        mean=normalizer.mean_,
        std=normalizer.std_,
    )

    if verbose:
        print(f"Model saved to '{model_path}'")
        print(f"Normalizer saved to '{normalizer_path}'")

    return {"train_accuracy": train_acc, "test_accuracy": test_acc}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the flying-object classifier."
    )
    parser.add_argument("--samples", type=int, default=200,
                        help="Synthetic samples per class (default: 200).")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs (default: 300).")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size (default: 32).")
    parser.add_argument("--model-path", default="model.npz",
                        help="Output path for model weights.")
    parser.add_argument("--normalizer-path", default="normalizer.npz",
                        help="Output path for normalizer statistics.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    args = parser.parse_args()

    metrics = train(
        n_samples_per_class=args.samples,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.seed,
        model_path=args.model_path,
        normalizer_path=args.normalizer_path,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

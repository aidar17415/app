"""
Inference / prediction module for the flying-object classifier.

Usage (from the repository root)::

    python -m src.predict \\
        --speed 50 --altitude 200 --size 0.5 --aspect-ratio 5 \\
        --flapping 0.9 --rotor-sound 0.0 --heat-signature 0.4

    python -m src.predict --model-path model.npz \\
        --normalizer-path normalizer.npz \\
        --speed 500 --altitude 8000 --size 40 --aspect-ratio 9 \\
        --flapping 0 --rotor-sound 0.05 --heat-signature 0.8
"""

from __future__ import annotations

import argparse

import numpy as np

from src.dataset import FeatureNormalizer
from src.model import CLASSES, NUM_FEATURES, FlyingObjectClassifier


def load_artifacts(
    model_path: str = "model.npz",
    normalizer_path: str = "normalizer.npz",
    hidden_sizes: list[int] | None = None,
) -> tuple[FlyingObjectClassifier, FeatureNormalizer]:
    """Load a saved model and normalizer from disk.

    Args:
        model_path:      Path to model weights (.npz).
        normalizer_path: Path to normalizer statistics (.npz).
        hidden_sizes:    Must match the architecture used during training.

    Returns:
        (model, normalizer) tuple ready for inference.
    """
    if hidden_sizes is None:
        hidden_sizes = [64, 32]

    model = FlyingObjectClassifier(hidden_sizes=hidden_sizes)
    model.load(model_path)

    data = np.load(normalizer_path)
    normalizer = FeatureNormalizer()
    normalizer.mean_ = data["mean"]
    normalizer.std_ = data["std"]

    return model, normalizer


def predict_single(
    speed_kmh: float,
    altitude_m: float,
    size_m: float,
    aspect_ratio: float,
    flapping: float,
    rotor_sound: float,
    heat_signature: float,
    model: FlyingObjectClassifier,
    normalizer: FeatureNormalizer,
) -> dict[str, object]:
    """Predict the class of a single flying object.

    Args:
        speed_kmh:       Speed in km/h.
        altitude_m:      Altitude in metres.
        size_m:          Size (wingspan / body length) in metres.
        aspect_ratio:    Width-to-height ratio.
        flapping:        Flapping motion indicator (0 or 1).
        rotor_sound:     Rotor/propeller sound intensity [0, 1].
        heat_signature:  Infrared heat-signature intensity [0, 1].
        model:           Trained :class:`FlyingObjectClassifier`.
        normalizer:      Fitted :class:`FeatureNormalizer`.

    Returns:
        Dict with keys:
            - ``"predicted_class"`` : str – top predicted label.
            - ``"probabilities"``   : dict[str, float] – per-class probabilities.
    """
    x = np.array([[
        speed_kmh, altitude_m, size_m, aspect_ratio,
        flapping, rotor_sound, heat_signature,
    ]], dtype=float)
    x_norm = normalizer.transform(x)
    probs = model.predict_proba(x_norm)[0]
    predicted_idx = int(np.argmax(probs))

    return {
        "predicted_class": CLASSES[predicted_idx],
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a flying object using a trained neural network."
    )
    parser.add_argument("--model-path", default="model.npz",
                        help="Path to saved model weights (default: model.npz).")
    parser.add_argument("--normalizer-path", default="normalizer.npz",
                        help="Path to normalizer statistics (default: normalizer.npz).")
    parser.add_argument("--speed", type=float, required=True,
                        help="Speed in km/h.")
    parser.add_argument("--altitude", type=float, required=True,
                        help="Altitude in metres.")
    parser.add_argument("--size", type=float, required=True,
                        help="Approximate size in metres.")
    parser.add_argument("--aspect-ratio", type=float, required=True,
                        help="Width-to-height ratio.")
    parser.add_argument("--flapping", type=float, default=0.0,
                        help="Flapping motion (0 or 1).")
    parser.add_argument("--rotor-sound", type=float, default=0.0,
                        help="Rotor/propeller sound intensity [0, 1].")
    parser.add_argument("--heat-signature", type=float, default=0.5,
                        help="Infrared heat-signature intensity [0, 1].")
    args = parser.parse_args()

    model, normalizer = load_artifacts(args.model_path, args.normalizer_path)

    result = predict_single(
        speed_kmh=args.speed,
        altitude_m=args.altitude,
        size_m=args.size,
        aspect_ratio=args.aspect_ratio,
        flapping=args.flapping,
        rotor_sound=args.rotor_sound,
        heat_signature=args.heat_signature,
        model=model,
        normalizer=normalizer,
    )

    print(f"Predicted class : {result['predicted_class']}")
    print("Probabilities   :")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {cls:<12} {prob:.4f}  {bar}")


if __name__ == "__main__":
    main()

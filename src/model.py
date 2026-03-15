"""
Neural network model for classifying flying objects.

Architecture: Multi-Layer Perceptron (MLP) implemented with NumPy.
Classifies objects into: bird, drone, airplane, helicopter.

Input features:
    - speed_kmh       : speed in km/h
    - altitude_m      : altitude in metres
    - size_m          : approximate size (wingspan / body length) in metres
    - aspect_ratio    : width-to-height ratio
    - flapping        : 1.0 if flapping motion detected, else 0.0
    - rotor_sound     : rotor/propeller sound intensity (0-1 normalised)
    - heat_signature  : infrared heat signature (0-1 normalised)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------

CLASSES = ["bird", "drone", "airplane", "helicopter"]
NUM_CLASSES = len(CLASSES)
NUM_FEATURES = 7


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cross_entropy_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Categorical cross-entropy loss.

    Args:
        probs:  (N, C) probability matrix from softmax.
        labels: (N,) integer class indices.

    Returns:
        Scalar mean loss.
    """
    n = probs.shape[0]
    clipped = np.clip(probs[np.arange(n), labels], 1e-12, 1.0)
    return -np.mean(np.log(clipped))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FlyingObjectClassifier:
    """Multi-layer perceptron for flying-object classification.

    Parameters
    ----------
    hidden_sizes : list[int]
        Number of units in each hidden layer (default: [64, 32]).
    learning_rate : float
        Gradient-descent step size (default: 0.01).
    random_seed : int | None
        Seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        hidden_sizes: list | None = None,
        learning_rate: float = 0.01,
        random_seed: int | None = 42,
    ) -> None:
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(random_seed)

        # Build weight matrices and bias vectors
        layer_sizes = [NUM_FEATURES] + hidden_sizes + [NUM_CLASSES]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialisation for ReLU layers
            std = np.sqrt(2.0 / fan_in)
            self.weights.append(self.rng.normal(0.0, std, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

        # Cache for backpropagation
        self._activations: list[np.ndarray] = []
        self._pre_activations: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities.

        Args:
            X: (N, NUM_FEATURES) feature matrix.

        Returns:
            (N, NUM_CLASSES) probability matrix.
        """
        self._activations = [X]
        self._pre_activations = []

        current = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            self._pre_activations.append(z)
            if i < len(self.weights) - 1:
                current = relu(z)
            else:
                current = softmax(z)
            self._activations.append(current)

        return current  # shape (N, NUM_CLASSES)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, labels: np.ndarray) -> None:
        """Update weights via mini-batch gradient descent.

        Args:
            labels: (N,) integer class indices used during the last forward pass.
        """
        n = labels.shape[0]
        probs = self._activations[-1]

        # Gradient of cross-entropy + softmax combined
        delta = probs.copy()
        delta[np.arange(n), labels] -= 1.0
        delta /= n

        for i in reversed(range(len(self.weights))):
            a_prev = self._activations[i]
            dW = a_prev.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                da_prev = delta @ self.weights[i].T
                delta = da_prev * relu_derivative(self._pre_activations[i - 1])

            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list[float]:
        """Train the model.

        Args:
            X:          (N, NUM_FEATURES) feature matrix.
            y:          (N,) integer class labels.
            epochs:     Number of full passes through the dataset.
            batch_size: Mini-batch size.
            verbose:    Print loss every 50 epochs when True.

        Returns:
            List of per-epoch loss values.
        """
        n = X.shape[0]
        loss_history: list[float] = []

        for epoch in range(1, epochs + 1):
            indices = self.rng.permutation(n)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                probs = self.forward(X_batch)
                loss = cross_entropy_loss(probs, y_batch)
                self.backward(y_batch)

                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)

            if verbose and epoch % 50 == 0:
                acc = self.score(X, y)
                print(f"Epoch {epoch:>4d}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

        return loss_history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices.

        Args:
            X: (N, NUM_FEATURES) feature matrix.

        Returns:
            (N,) integer array of predicted class indices.
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix.

        Args:
            X: (N, NUM_FEATURES) feature matrix.

        Returns:
            (N, NUM_CLASSES) probability matrix.
        """
        return self.forward(X)

    def predict_labels(self, X: np.ndarray) -> list[str]:
        """Return human-readable class names.

        Args:
            X: (N, NUM_FEATURES) feature matrix.

        Returns:
            List of class-name strings.
        """
        indices = self.predict(X)
        return [CLASSES[i] for i in indices]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy (fraction of correctly classified samples).

        Args:
            X: (N, NUM_FEATURES) feature matrix.
            y: (N,) ground-truth class indices.

        Returns:
            Accuracy in [0, 1].
        """
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to a NumPy .npz file.

        Args:
            path: File path (will get .npz extension if not present).
        """
        arrays = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

    def load(self, path: str) -> None:
        """Load model weights from a NumPy .npz file.

        Args:
            path: File path produced by :meth:`save`.
        """
        data = np.load(path)
        num_layers = len(self.weights)
        for i in range(num_layers):
            self.weights[i] = data[f"W{i}"]
            self.biases[i] = data[f"b{i}"]

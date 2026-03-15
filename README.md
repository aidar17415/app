# Ауада ұшатын объектілерді жіктеу үшін жасанды нейрондық желілер

**Classification of Flying Objects Using Artificial Neural Networks**

---

## Жобаның мақсаты / Project Goal

Бұл жоба ауада ұшатын объектілерді (құс, дрон, ұшақ, тікұшақ) жасанды нейрондық желі (MLP) арқылы жіктеу үшін жасалған.

This project implements a **multi-layer perceptron (MLP)** neural network — built from scratch with NumPy — to classify flying objects into four categories:

| Класс / Class | Мысал / Example |
|---|---|
| `bird` | Құс / Birds |
| `drone` | Дрон / Unmanned aerial vehicles |
| `airplane` | Ұшақ / Fixed-wing aircraft |
| `helicopter` | Тікұшақ / Rotary-wing aircraft |

---

## Белгілер (сипаттамалар) / Input Features

The model uses **7 numerical features** extracted from sensor observations:

| # | Feature | Description |
|---|---------|-------------|
| 0 | `speed_kmh` | Speed in km/h |
| 1 | `altitude_m` | Altitude in metres |
| 2 | `size_m` | Approximate size (wingspan / body length) in metres |
| 3 | `aspect_ratio` | Width-to-height ratio |
| 4 | `flapping` | Flapping motion indicator (0 or 1) |
| 5 | `rotor_sound` | Rotor / propeller sound intensity [0, 1] |
| 6 | `heat_signature` | Infrared heat-signature intensity [0, 1] |

---

## Жоба құрылымы / Project Structure

```
app/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── model.py       # MLP neural network (forward, backward, fit, predict, save/load)
│   ├── dataset.py     # Synthetic dataset generation, normalisation, train/test split
│   ├── train.py       # Training pipeline (CLI + importable function)
│   └── predict.py     # Inference pipeline (CLI + importable function)
└── tests/
    ├── __init__.py
    ├── test_model.py       # Unit tests for the neural network
    ├── test_dataset.py     # Unit tests for dataset utilities
    └── test_integration.py # End-to-end training and prediction tests
```

---

## Орнату / Installation

```bash
pip install -r requirements.txt
```

---

## Пайдалану / Usage

### 1 · Train the model

```bash
python -m src.train
```

Options:

```
--samples     INT    Synthetic samples per class  (default: 200)
--epochs      INT    Training epochs              (default: 300)
--lr          FLOAT  Learning rate                (default: 0.01)
--batch-size  INT    Mini-batch size              (default: 32)
--model-path  PATH   Output file for weights      (default: model.npz)
--normalizer-path PATH  Output file for stats     (default: normalizer.npz)
--seed        INT    Random seed                  (default: 42)
```

Example output:

```
Generating dataset …
Train size: 640, Test size: 160, Features: 7
Training …
Epoch   50/300  loss=0.3241  acc=0.9219
Epoch  100/300  loss=0.1182  acc=0.9703
Epoch  150/300  loss=0.0671  acc=0.9859
Epoch  200/300  loss=0.0470  acc=0.9922
Epoch  250/300  loss=0.0364  acc=0.9938
Epoch  300/300  loss=0.0297  acc=0.9953

Final train accuracy : 0.9953
Final test  accuracy : 0.9875
```

### 2 · Classify a single object

```bash
# Bird-like observation
python -m src.predict \
  --speed 40 --altitude 100 --size 0.6 --aspect-ratio 6 \
  --flapping 0.95 --rotor-sound 0.0 --heat-signature 0.45

# Airplane-like observation
python -m src.predict \
  --speed 700 --altitude 10000 --size 60 --aspect-ratio 10 \
  --flapping 0 --rotor-sound 0.02 --heat-signature 0.9
```

Example output:

```
Predicted class : bird
Probabilities   :
  bird         0.9872  █████████████████████████████
  drone        0.0041
  airplane     0.0065
  helicopter   0.0022
```

### 3 · Run tests

```bash
python -m unittest discover -s tests -v
```

---

## Нейрондық желінің архитектурасы / Neural Network Architecture

```
Input  (7)  →  Hidden (64)  →  Hidden (32)  →  Output (4)
               ReLU             ReLU             Softmax
```

Training uses:
- **Mini-batch gradient descent** with backpropagation
- **He weight initialisation** for stable ReLU-network training
- **Z-score feature normalisation** to accelerate convergence
- **Cross-entropy loss** for multi-class classification

---

## Тестілеу / Testing

50 unit and integration tests covering:

- Activation functions (`relu`, `softmax`, `relu_derivative`)
- Cross-entropy loss correctness
- Model forward / backward pass shapes
- Probability mass conservation
- Weight save / load round-trip
- Dataset generation and reproducibility
- Feature normaliser (zero mean, unit variance)
- Train–test split (no overlap, correct sizes)
- End-to-end accuracy (>50% after training with moderate data)
- Per-class semantic correctness (bird, drone, airplane, helicopter)

---

## Лицензия / License

MIT

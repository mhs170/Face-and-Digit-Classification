
# 🧠 Face and Digit Classification – CS440 Project (Spring 2025)

This project implements and compares three machine learning algorithms — a Perceptron, a custom-built three-layer Neural Network, and a PyTorch-based Neural Network — to classify handwritten digits (0–9) and facial images (binary classification: face or not).

---

## 📂 Directory Structure

```
FaceAndDigitClassification/
│
├── code/
│   ├── perceptron.py             # Perceptron classifier from scratch
│   ├── neural_net.py             # Three-layer Neural Network (custom implementation)
│   ├── nn_pytorch.py             # Three-layer Neural Network using PyTorch
│   ├── run_perceptron.py         # Trains, evaluates, and plots Perceptron results
│   ├── run_nn.py                 # Trains and evaluates custom neural net
│   ├── run_nn_pytorch.py         # Trains and evaluates PyTorch neural net
│   ├── demo_perceptron.py        # Loads saved Perceptron and allows testing
│   ├── test_perceptron.py        # Reports accuracy on test sets
│   ├── data_loader.py            # Loads and formats image data
│   ├── util.py                   # Utility functions
│
├── data/
│   ├── digitdata/                # Handwritten digit dataset
│   └── facedata/                 # Preprocessed face image dataset
│
├── models/                       # Saved model weights for demo
├── results/                      # Accuracy/time plots and learning curves
└── README.md                     # Project overview (this file)
```

---

## 🧠 Algorithms Implemented

### 1. Perceptron (from scratch)
- Implements online Perceptron learning with weight updates.
- Supports multi-class classification (for digits) and binary (for faces).
- Performance improves with training size.

### 2. Neural Network (from scratch)
- Three-layer network: input → hidden1 → hidden2 → output.
- Implements forward propagation, backpropagation, and manual weight updates.
- Supports classification using sigmoid/softmax activations.

### 3. Neural Network (with PyTorch)
- Three-layer fully connected network implemented using PyTorch.
- Trained using `CrossEntropyLoss` and `Adam` optimizer.
- Much easier to train and tune, but requires external library.

---

## 📊 Training and Evaluation

For each algorithm, we:
- Trained on increasing percentages of the dataset (10% to 100%)
- Repeated each experiment 5 times with different random samples
- Recorded:
  - ⏱ Training time (in seconds)
  - ❌ Prediction error and 📉 Standard deviation
- Plotted learning curves for both accuracy and training time

---

## 🖼 Demo

To run the demo for any model:

```bash
python demo_perceptron.py        # or demo_nn.py / demo_nn_pytorch.py
```

You'll be prompted to choose between "digit" and "face" and can select individual test images for prediction. Output includes:
- ASCII visualization of the image
- Predicted and actual labels

---

## 📈 Example Results (Perceptron)

| Training Size | Digit Accuracy | Face Accuracy |
|---------------|----------------|---------------|
| 10%           | 71%            | 79%           |
| 100%          | 91%            | 87%           |

_Note: Neural networks performed better than Perceptron, especially on the digit data._

---

## 📌 Lessons Learned

Working on the Perceptron algorithm gave us a solid, hands-on understanding of how a simple learning model can be used for classification tasks. Writing the algorithm from scratch helped clarify how weight updates are driven by classification mistakes, and how those small changes add up over time to shape the model’s behavior. We also realized how important data size is—the model becomes noticeably more consistent and accurate as it sees more examples. One thing that stood out was how much harder digit classification was compared to face classification, likely due to the multi-class nature of the task. Along the way, we also learned the value of running multiple trials to capture average performance and variability, as well as the practical benefit of saving trained models to avoid retraining during a demo. Overall, the project helped us connect theoretical ideas about supervised learning to actual implementation and testing.

---

## 📎 Requirements

- Python 3.x
- `matplotlib`
- `torch` (for PyTorch model only)

Install with:

```bash
pip install matplotlib torch
```


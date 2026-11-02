"""Train and evaluate a small handwritten-digit classifier."""

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Save figures without requiring a desktop window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_mistakes(features, labels, predictions, output_path):
    """Show up to ten test images where the prediction was wrong."""
    mistakes = np.flatnonzero(predictions != labels)
    figure, axes = plt.subplots(2, 5, figsize=(10, 6))
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, index in zip(axes.flat, mistakes[:10]):
        # The features were normalized to [0, 1] before training.
        axis.imshow(features[index].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"True: {labels[index]}\nPredicted: {predictions[index]}")
    figure.suptitle(f"Test mistakes: {len(mistakes)} / {len(labels)}")
    figure.subplots_adjust(top=0.85, hspace=0.5)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    digits = load_digits()
    print(f"Features: {digits.data.shape}; images: {digits.images.shape}")
    print(f"Labels: {digits.target.shape}; first label: {digits.target[0]}")

    # Each row contains 64 pixel values. Labels are kept separate from inputs.
    # Dividing by the known maximum needs no statistics from the test set.
    features = digits.data / 16.0
    labels = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(y_train)}; test: {len(y_test)}; split seed: 42")

    # A reference score: always predict the most common training label.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    # fit() learns the weights; predict() applies them to unseen examples.
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Model: LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)")
    print(f"Solver iterations: {model.n_iter_.max()}")
    print(f"Baseline test accuracy: {baseline.score(X_test, y_test):.2%}")
    print(f"Model train accuracy:   {model.score(X_train, y_train):.2%}")
    print(f"Model test accuracy:    {model.score(X_test, y_test):.2%}")
    print(f"First test prediction: {predictions[0]}; true label: {y_test[0]}")

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    figure, axis = plt.subplots(figsize=(3, 3))
    axis.imshow(digits.images[0], cmap="gray", vmin=0, vmax=16)
    axis.set_title(f"Dataset sample: {digits.target[0]}")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_dir / "sample.png", dpi=150)
    plt.close(figure)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=digits.target_names, cmap="Blues"
    )
    display.ax_.set_title("Digit classification: test set")
    display.figure_.tight_layout()
    display.figure_.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(display.figure_)

    save_mistakes(X_test, y_test, predictions, output_dir / "mistakes.png")
    print(f"Saved sample.png, confusion_matrix.png, and mistakes.png to {output_dir}")


if __name__ == "__main__":
    main()

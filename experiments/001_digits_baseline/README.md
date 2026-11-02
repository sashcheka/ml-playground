# 001: Handwritten digit classification

## Goal

Train a first classifier that takes a digit image and predicts a label from 0 to 9.
Understand the full cycle: inspect data, split it, train a model, evaluate it, and
look at its mistakes.

This is a small introductory exercise inspired by studying machine learning,
rather than a reproduction of a particular book chapter.

## Dataset and model

Use scikit-learn's built-in [`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
dataset: 1,797 grayscale images, each 8 by 8 pixels, with values from 0 to 16.
Each image becomes 64 input features; its label is the digit it represents.
The dataset ships with scikit-learn, so no separate dataset download is needed.

Start with [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
Despite its name, this is a classification model. It gives us a small first
experiment before implementing the learning algorithm ourselves with NumPy.

Libraries:

- **scikit-learn:** dataset, data split, classifier, and evaluation.
- **NumPy:** inspecting and transforming arrays.
- **Matplotlib:** viewing images and mistakes.

## Setup

From the repository root:

```sh
cd experiments/001_digits_baseline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

For the exact versions used in the recorded run, install `requirements.lock.txt`
instead of `requirements.txt` in a fresh environment.

## What the script does

`main.py` loads the data, normalizes pixels, makes a fixed stratified 80/20 split,
trains a dummy baseline and logistic regression, and prints their accuracy.
The model is fitted only on the training set.

It saves three figures under `outputs/`, relative to the script's directory:

- `sample.png`: one input image and its label.
- `confusion_matrix.png`: true labels on rows and predictions on columns;
  off-diagonal entries are mistakes.
- `mistakes.png`: up to ten incorrect predictions, with true and predicted labels.

The figures are saved directly, so no interactive plot window is required.
The model is retrained on each run; this example does not save a model file.

## First reading task: understand one sample

Read the data-loading section of `main.py`. Inspect `digits.images[0]` and
`digits.target[0]` to connect the pixel values with the label.

Expected shapes are `(1797, 64)`, `(1797, 8, 8)`, and `(1797,)`.
Before moving on, explain why the first two arrays describe the same images
and why the label must not become an input feature.

## Read the training flow

Follow these steps in `main.py` after inspecting the data:

1. Split into 80% training and 20% test data. Use `random_state=42` and
   `stratify=target` to preserve class proportions.
2. Normalize pixel values by dividing by the known maximum, 16. This fixed
   transformation does not need to learn anything from the test data.
3. Train logistic regression on the training data only. Check for convergence
   warnings and adjust the iteration limit if necessary.
4. Report training and test accuracy. Compare test accuracy with a
   `DummyClassifier(strategy="most_frequent")` trained on the same training data.
5. Inspect a confusion matrix and display some misclassified test images.
6. Compare your output with the recorded run in `NOTES.md`.

Use training data or a separate validation split to choose settings; reserve the
test set for evaluation. If you later add a learned transformation such as
`StandardScaler`, fit it on training data only, preferably inside a pipeline.

## Done when

- The script completes without convergence warnings.
- It reports the split, seed, model settings, baseline score, and model scores.
- You can explain the input, target, prediction, and difference between training
  and test accuracy.
- Your notes include a look at the model's mistakes, not just its accuracy.

## Next experiment

Once this baseline makes sense, create a separate NumPy experiment implementing
multiclass logistic regression: scores, softmax, cross-entropy, and gradient
descent. Reuse the same dataset and split to compare results.

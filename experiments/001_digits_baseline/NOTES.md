# Experiment notes

## Question

How well can logistic regression recognize the built-in digit images?

## Run

- Date: 2026-11-02 (Europe/Minsk).
- Python version: 3.12.15, macOS on Apple Silicon.
- Dependencies: `requirements.lock.txt`; NumPy 2.5.4, scikit-learn 1.9.1,
  Matplotlib 3.11.2.
- Command: `python main.py` from this experiment's directory.
- Train/test split: 1,437 / 360 samples, stratified, `random_state=42`.
- Inputs: 64 pixel values per image, divided by 16; targets: digits 0 through 9.
- Model: `LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)`.
- Converged after 74 solver iterations; no convergence warnings.

## Results

- Dummy baseline test accuracy: **10.00%**.
- Model training accuracy: **98.68%**.
- Model test accuracy: **95.56%** (344 correct predictions out of 360).
- Errors: **16**. The confusion matrix shows six errors for digit 1, including
  two predictions of 9. Digit 8 was predicted as 1 three times; digit 9 was
  predicted as 4 twice.
- Figures: `outputs/sample.png`, `outputs/confusion_matrix.png`, and
  `outputs/mistakes.png`. These generated files are ignored by Git.

## What I learned

- One 8 by 8 image becomes a row of 64 features; the label stays separate.
- `fit()` learns weights from training data. `predict()` uses those weights on
  new inputs. Accuracy is the fraction of correct predictions.
- The dummy score gives the model a simple reference to beat.
- Training accuracy is higher than test accuracy. The 3.12 percentage-point gap
  is a reason to examine generalization; one split alone does not establish
  how the model will perform on other handwriting or image formats.
- Normalizing by the known pixel maximum requires no learned statistics.
- The first Matplotlib run may take longer while building its font cache.

## Verification

- Full training and plotting completed with Python warnings treated as errors.
- A repeat run produced the same displayed scores and iteration count.
- All three PNGs were validated; the confusion matrix and mistake grid were
  visually inspected.
- The mistake grid also handled zero, one, and more than ten errors.
- `python -m pip check` reported no broken requirements.

## Next question

Trace one image through the script and explain how its 64 inputs become one
predicted label. Then implement multiclass logistic regression with NumPy in a
new experiment to study softmax, cross-entropy, and gradient descent. Keep the
same split for comparison, and use validation data for any parameter tuning.

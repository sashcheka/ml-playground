# ML Playground

Small, independent Python experiments for learning machine learning by implementing,
training, and evaluating models.

## Structure

```text
experiments/
  _template/              # Copy this when starting a new experiment
  001_digits_baseline/   # First exercise: handwritten digit classification
    README.md            # Goal, setup, and implementation steps
    requirements.txt     # Dependencies for this experiment
    NOTES.md             # Observations and results
```

Each experiment owns its dependencies and virtual environment. This lets a later
experiment use NumPy, PyTorch, or another library without changing earlier work.
Use Python 3.12 for the initial experiment.

Start with a short `main.py` inside the experiment. Split it into modules when the
code needs it. Notebooks can live alongside scripts for exploration.

Create `data/` for downloaded datasets and `outputs/` for generated plots, metrics,
or saved models when needed. Both are ignored by Git inside any experiment.
Keep small result summaries in `NOTES.md` so they remain part of the repository.

## First experiment

From the repository root:

```sh
cd experiments/001_digits_baseline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then follow [the digit-recognition exercise](experiments/001_digits_baseline/README.md).
The scaffold contains instructions; you will write the model yourself.

## Adding an experiment

From the repository root:

```sh
cp -R experiments/_template experiments/002_short_name
```

Update its README and dependencies, then create its own `.venv` as above. Run
scripts from inside their experiment directory, and use `deactivate` when done.
Use the next available number; keep variations of the same question together.

Dependency files list the libraries needed to begin. Once an experiment has a
working result, capture the exact environment from its activated virtual
environment and commit the resulting file:

```sh
python -m pip freeze > requirements.lock.txt
```

To reproduce that environment, install `requirements.lock.txt` in a fresh virtual
environment instead of `requirements.txt`. Record the Python version in the notes.

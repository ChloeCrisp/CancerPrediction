# CancerPrediction
Binary classification model predicting breast cancer. Data from Kaggle.

## Setup with uv

This project uses the [uv](https://github.com/astral-sh/uv) Python package manager to
manage dependencies and reproducible environments.

To create the environment and install packages, run:

```bash
# install uv if you don't have it already
python -m pip install uv

# sync dependencies and create virtual environment
uv sync
```

A lockfile (`uv.lock`) is committed for reproducibility. To install exactly the
locked dependencies on another machine:

```bash
uv sync --locked
```

Dependencies include common data science libraries such as `numpy`, `pandas`,
`scikit-learn`, `matplotlib`, `seaborn`, and development tools like `black`,
`pytest`, and `flake8`.

## Pre-commit checks

A `pre-commit` hook is configured to run [ruff](https://github.com/charliermarsh/ruff) on
every commit. Install the hook once after environment setup:

```bash
uv run pre-commit install
```

It will block the commit if lint errors are detected, helping keep the code
clean. To run the checks manually:

```bash
uv run pre-commit run --all-files
```

Add or adjust hooks by editing `.pre-commit-config.yaml` and then re-installing
or running `pre-commit autoupdate` to bump revisions.



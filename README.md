# mllf
Machine Learning Landscape Flattening model building and training

## Installation

Recommended: create a virtual environment, install the package in editable mode with dev extras, and run tests.

Using venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

If you use conda, activate your environment then run:

```bash
pip install -e '.[dev]'
```

## Running tests

Run the test suite with pytest:

```bash
python -m pytest tests/ -v
```

Optional: run flake8 for linting (installed via dev extras):

```bash
flake8
```

## Documentation

Please see full documentation at https://mllf.readthedocs.io/en/latest for more detailed information.


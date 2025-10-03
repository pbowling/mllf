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

## Notes

- The repository uses a `src/` layout. The test suite currently contains a test for `read_bias_coeff` which parses an example `examples/mlp/variables85.inp` file.
- If you add runtime dependencies, list them under `[project.dependencies]` in `pyproject.toml` so they are installed by pip.

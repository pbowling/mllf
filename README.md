# mllf
![mllf logo](docs/mllf_logo.png)

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

## RLlib training (optional)

To use Ray RLlib for scalable training, install the optional `rl` extras which include Ray/RLlib:

```bash
pip install -e '.[rl]'
```

A tiny example trainer is provided at `src/mllf/rl/rllib_trainer.py`. To run a short test training run:

```python
from mllf.rl.rllib_trainer import train
train(num_iters=10)
```

Note: RLlib is a large dependency; if you only need the GNN prototype for quick experiments, you can run the `GNNPolicy` directly in a custom PyTorch loop instead.


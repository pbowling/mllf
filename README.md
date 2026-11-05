# mllf
![mllf logo](docs/mllf_logo.png)

Machine-learned landscape flattening: bias coefficients for multisite lambda dynamics,
pretrained on collected simulations and trained online against new ones.

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

## Usage

Installing puts the `mllf` command on your path:

```bash
mllf -h                                   # list the commands
mllf collect RUN_DIR -o pretraining/sys   # collect finished MSLD runs
mllf pretrain --pretraining-dir pretraining/sys --output-dir models/run --config pretrain.yaml
mllf train workflow.yaml                  # online training (rerun to resume)
mllf export workflow.yaml --label checkpoints_dir
mllf cost checkpoints_dir --pretraining-root pretraining
```

`-v` / `-q` (before or after the command) give more or less output; `mllf COMMAND -h`
lists a command's options. See `docs/cli.rst`.

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


<p align="center">
  <img src="https://raw.githubusercontent.com/pbowling/mllf/main/docs/mllf_logo.png"
       alt="mllf logo" width="260">
</p>

<p align="center">
  <a href="https://github.com/pbowling/mllf/actions/workflows/ci.yml"><img src="https://github.com/pbowling/mllf/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://mllf.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/mllf/badge/?version=latest" alt="Documentation"></a>
  <!-- once released: <a href="https://pypi.org/project/mllf/"><img src="https://img.shields.io/pypi/v/mllf" alt="PyPI"></a> -->
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python 3.10 | 3.11 | 3.12">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
</p>

# mllf: machine-learned landscape flattening

Multisite λ-dynamics (MSLD) samples many alchemical perturbations in one simulation, but
only after each system has bias coefficients that flatten its alchemical landscape.
Finding those biases iteratively costs many simulations per system, and nothing learned on
one system carries over to the next.

`mllf` predicts the biases instead. A shared model, conditioned on Uni-Mol embeddings of
each substituent in its environment, proposes biases for systems it has never simulated.
Each simulation then corrects both the model and the pair of substituents it sampled. The
aim is converged biases in fewer simulated nanoseconds than iterative flattening,
including for systems absent from the training data.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Development](#development)
- [Citing mllf](#citing-mllf)
- [License](#license)

## Installation

`mllf` needs Python 3.10 or newer. We recommend a fresh conda or venv environment.

Install PyTorch first, choosing the build that matches your hardware from
[pytorch.org](https://pytorch.org/get-started/locally/). For example, the CPU-only build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install `mllf` from source:

```bash
git clone https://github.com/pbowling/mllf.git
cd mllf
pip install -e ".[analysis]"     # the analysis extra adds the figures of mllf analyze
```

<!-- once released on PyPI:
```bash
pip install "mllf[analysis]"
```
-->

The extras are:

| Extra      | Adds                                                     |
|------------|----------------------------------------------------------|
| `analysis` | matplotlib, pandas, scipy and pillow for `mllf analyze`  |
| `dev`      | pytest, flake8, isort, pre-commit, build and twine       |

**Simulations.** Training and production runs launch MSLD simulations through CHARMM with
pyCHARMM and submit them with Slurm. Neither is installed by `pip`: the workflow loads
CHARMM through the `slurm:` block's `module` and `setup` settings. See the
[installation guide](https://mllf.readthedocs.io/en/latest/installation.html).

## Quick start

Installing puts the `mllf` command on your path. The commands, in pipeline order:

```bash
mllf combos ...                           # combination directories from site/sub files
mllf collect RUN_DIR -o pretraining/sys   # collect finished MSLD runs
mllf pretrain --pretraining-dir pretraining/sys --output-dir models/run --config pretrain.yaml
mllf train workflow.yaml                  # online training (rerun to resume)
mllf export workflow.yaml --label checkpoints_dir
mllf production production.yaml           # replicas with fixed biases, then free energies
mllf analyze ...                          # figures and a report for a model or a run
mllf cost checkpoints_dir --pretraining-root pretraining
mllf archive workflow.yaml                # one .tar.gz of combo directories per system
```

`mllf COMMAND -h` lists a command's options, and `-v` / `-q` give more or less output. The
console shows progress and per-epoch summaries. Every message, including per-combo detail,
also goes to a log file in the run's output directory (`training.log`, `pretrain.log`).

[`examples/minimal`](examples/minimal) runs the whole pipeline on one system (1,4-benzene in
water) and is small enough to check an installation end to end:

```bash
sbatch examples/minimal/run_pipeline.sh
```

## Documentation

The full documentation is at **[mllf.readthedocs.io](https://mllf.readthedocs.io/en/latest/)**.
It covers the background on MSLD biases, the workflow configuration, how the model learns,
analysis and production runs, file formats and the API.

## Development

```bash
pip install -e ".[analysis,dev]"
python -m pytest tests/ -v
pre-commit install            # flake8 and isort on every commit
pre-commit run --all-files
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the same hooks, the
tests against the built wheel on Python 3.10 and 3.12, and the documentation build.
Publishing a GitHub release uploads the package to PyPI
([`.github/workflows/release.yml`](.github/workflows/release.yml)).

## Citing mllf

If you use `mllf` in your work, please cite:

```bibtex
@software{bowling_mllf,
  author = {Bowling, Paige E. and Brooks, Charles L., III},
  title  = {mllf: machine-learned landscape flattening for multisite lambda dynamics},
  url    = {https://github.com/pbowling/mllf},
  year   = {2026}
}
```

## License

`mllf` is released under the [MIT license](LICENSE).

Installation
============

Recommended steps to get the project running locally.

Virtualenv (recommended)
------------------------

1. Create and activate a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the package and development requirements:

```bash
pip install -e .
pip install -r docs/requirements.txt
```

Conda (alternative)
--------------------

```bash
conda create -n mllf python=3.10
conda activate mllf
pip install -e .
pip install -r docs/requirements.txt
```

Running tests
-------------

Run the unit tests with pytest:

```bash
python -m pytest tests/ -v
```

Building the docs locally
-------------------------

See `docs/README.md` for details on building the documentation locally.

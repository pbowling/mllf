Building the docs locally

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r docs/requirements.txt
pip install -e .
```

2. Build HTML:

```bash
sphinx-build -b html . _build/html
```

Read the Docs
------------
This project includes a `.readthedocs.yaml` that configures RTD to use the project root
`conf.py` and to install the package and `docs/requirements.txt` prior to building.

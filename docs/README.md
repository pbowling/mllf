Building the docs locally

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r docs/requirements.txt
pip install -e .
```

2. Build HTML from the repository root:

```bash
sphinx-build -b html docs docs/_build/html
```

or, from `docs/`, `make html` (`make.bat html` on Windows). Open
`docs/_build/html/index.html` to view the result.

Read the Docs
------------
`.readthedocs.yaml` at the repository root configures Read the Docs to use `docs/conf.py`
and to install the package and `docs/requirements.txt` before building. `docs/index.rst`
is the landing page at https://mllf.readthedocs.io/en/latest/.

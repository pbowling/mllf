# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys


# make sure the package in ../src is importable when building docs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from mllf import __version__  # noqa: E402  (one version, in src/mllf/__init__.py)


project = 'Machine Learned Landscape Flattening'
copyright = '2025, Paige E. Bowling, Charles L. Brooks III'
author = 'Paige E. Bowling, Charles L. Brooks III'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinxcontrib.bibtex',
    'sphinx.ext.mathjax',
]

# generate autosummary stub pages
autosummary_generate = True

# Prefer the ReadTheDocs theme; fall back to alabaster if it is not installed
try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = 'sphinx_rtd_theme'
except ImportError:
    html_theme = 'alabaster'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

# configuration for sphinxcontrib-bibtex
bibtex_bibfiles = ['references.bib']

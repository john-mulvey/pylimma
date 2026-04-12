# Sphinx configuration for pylimma documentation.

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pylimma._version import __version__ as _release  # noqa: E402


project = "pylimma"
copyright = "2026, John Mulvey. Based on R limma by Gordon Smyth et al."
author = "John Mulvey"

release = _release
version = ".".join(_release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autosummary_generate = True

intersphinx_mapping = {
    "python":      ("https://docs.python.org/3", None),
    "numpy":       ("https://numpy.org/doc/stable/", None),
    "scipy":       ("https://docs.scipy.org/doc/scipy/", None),
    "pandas":      ("https://pandas.pydata.org/docs/", None),
    "anndata":     ("https://anndata.readthedocs.io/en/stable/", None),
    "matplotlib":  ("https://matplotlib.org/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

nbsphinx_execute = "never"

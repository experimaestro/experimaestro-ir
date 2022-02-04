# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# --- Allow not to depend on too many things when building the documentation

import re
from experimaestro.mkdocs.metaloader import DependencyInjectorFinder

DependencyInjectorFinder.install(
    re.compile(r"^(torch|faiss|pandas|bs4|pytorch_transformers|pytrec_eval|apex)($|\.)")
)


# -- Project information -----------------------------------------------------

project = "Experimaestro"
copyright = "2021, Benjamin Piwowarski"
author = "Benjamin Piwowarski"

# The full version, including alpha/beta/rc tags
release = "0.8.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "experimaestro.sphinx",
    "sphinx_rtd_theme",
    "sphinx.ext.autosummary",
    # Link to other documentations
    "sphinx.ext.intersphinx",
    # Google style docstrings
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


intersphinx_mapping = {
    "datamaestro_text": (("../../../datamaestro_text/docs/build/html", None))
}


autodoc_default_options = {"show-inheritance": True}

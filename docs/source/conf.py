import builtins
from experimaestro.experiments.mockmodule import mock_modules
from xpmir import version

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

# --- Set build mode
# Used to notify python modules that we are building
# a documentation

builtins.__sphinx_build__ = True


# -- Project information -----------------------------------------------------

project = "Experimaestro IR (XPM-IR)"
copyright = "2023, Benjamin Piwowarski"
author = "Benjamin Piwowarski"

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Experimaestro extension
    "experimaestro.sphinx",
    "datamaestro.sphinx",
    # Math in docs with MathJax
    "sphinx.ext.mathjax",
    # Read The Docs theme
    "sphinx_rtd_theme",
    # Use Markdown parser
    "myst_parser",
    # auto documention
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # Link to other documentations
    "sphinx.ext.intersphinx",
    # Google style docstrings
    "sphinx.ext.napoleon",
    # Named tuples
    "sphinx_toolbox.more_autodoc.autonamedtuple",
    # Code
    "sphinx.ext.viewcode",
    # Auto-link identifiers in code blocks to API docs
    "sphinx_codeautolink",
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
    "experimaestro": (
        "https://experimaestro-python.readthedocs.io/en/latest/",
        None,
    ),
    "datamaestro_ir": (
        "https://datamaestro-ir.readthedocs.io/en/latest/",
        None,
    ),
    "xpm-torch": ("https://xpm-torch.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Autodoc options

autodoc_default_options = {
    "show-inheritance": True,
}

autodoc_mock_imports = [
    "faiss",
    "pandas",
    "bs4",
    "pytorch_transformers",
    "pytrec_eval",
    "apex",
    "ir_measures",
    "huggingface_hub",
]

# Use experimaestro's mock module system for torch and transformers.
# This handles class inheritance, subscripting, decorators, and torch.autograd.Function.
mock_modules(
    [
        "torch",
        "transformers",
        "torchdata",
        "lightning",
        "lightning_fabric",
        "pytorch_lightning",
    ]
)

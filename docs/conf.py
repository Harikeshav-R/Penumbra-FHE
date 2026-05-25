"""Sphinx configuration for Penumbra."""

import importlib.metadata

project = "Penumbra"
copyright = "2026, Harikeshav R"  # noqa: A001
author = "Harikeshav R"

try:
    version = importlib.metadata.version("penumbra-fhe")
except importlib.metadata.PackageNotFoundError:
    version = "0.0.0-dev"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx.ext.todo",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
}
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    # ONNX docs don't have a reliable intersphinx inventory, omit for now.
}

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

napoleon_google_docstring = False
napoleon_numpy_docstring = False
napoleon_use_param = True

nitpicky = True

# Note: rustdoc lives at _build/html/rustdoc/
# The Sphinx index links to it.

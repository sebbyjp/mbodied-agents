# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Mbodied Agents"
copyright = "2024, mbodi ai team"
author = "mbodi ai team"

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "../stubs", "../examples", "stubs", "examples"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]


# Select theme for both light and dark mode
pygments_style = "default"
# Select a different theme for dark mode
pygments_style_dark = "monokai"

html_permalinks_icon = "<span>∞</span>"

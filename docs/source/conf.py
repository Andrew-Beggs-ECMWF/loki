# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# pylint: disable=invalid-name,redefined-builtin

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../scripts'))


# -- Project information -----------------------------------------------------

project = 'Loki'
copyright = '2018- European Centre for Medium-Range Weather Forecasts (ECMWF)'
author = 'Michael Lange, Balthasar Reuter'

# The full version, including alpha/beta/rc tags.
release = re.sub('^v', '', os.popen('git describe').read().strip())
# The short X.Y version.
version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo',  # include todos
    'sphinx.ext.autodoc',  # use docstrings
    'sphinx.ext.napoleon',  # understand docstrings also in other formats
    'recommonmark',  # read markdown
]

# The file extensions of source files. Sphinx considers the files with
# this suffix as sources. The value can be a dictionary mapping file
# extensions to file types.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinxawesome_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    "nav_include_hidden": True,
    "show_nav": True,
    "show_breadcrumbs": True,
    "breadcrumbs_separator": "/",
    "show_prev_next": False,
}

html_collapsible_definitions = True


# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for autodoc extension -------------------------------------------

autodoc_default_options = {
    'members': True,  # include members in the documentation
    'member-order': 'bysource',  # members in the order they appear in source
    'show-inheritance': True,  # list base classes
}

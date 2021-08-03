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
import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'cleanX'
copyright = '2021, doctormakeda@gmail.com'
author = 'doctormakeda@gmail.com'

# The full version, including alpha/beta/rc tags
release = '0.0.4'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.extlinks',
    'sphinx.ext.imgmath',
    'sphinx.ext.intersphinx',
]

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
html_theme = 'nature'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

extlinks = {
    'pd': (
        'https://pandas.pydata.org/docs/reference/api/pandas.%s.html',
        'pandas.',
    ),
    'pdcm': (
        'https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.%s.html',
        'pydicom.',
    )
}

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__,__iter__,__reduce__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'inherited-members': True,
}

intersphinx_mapping = {
    'python': (
        'https://docs.python.org/{.major}'.format(
            sys.version_info,
        ),
        None,
    ),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'matplotlib': ('http://matplotlib.org', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

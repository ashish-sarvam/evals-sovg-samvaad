# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project name can be changed
project = 'Sarvam'
copyright = '2023, Sarvam'
author = 'Sarvam'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    # 'sphinx.ext.viewcode', # uncommenting this will show source code in docs
    'sphinx.ext.napoleon',
    'sphinxcontrib.autodoc_pydantic'
]

autodoc_pydantic_model_show_json = True
autodoc_pydantic_settings_show_json = False

# add module names to the function signature in the document
add_module_names = False
# table of content to include function or class names
toc_object_entries = False

# python_maximum_signature_line_length = 150

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# For a full list of documentation options, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# ----------------------------------------------------------------------------

import os
from os.path import dirname as up

from datetime import date

import sphinx_bootstrap_theme
from riskslim import __version__
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

# -- Project information -----------------------------------------------------

# Set project information
project = 'riskslim'
copyright = '2021-{}'.format(date.today().year)
author = 'Berk Ustun'

# Get and set the current version number

version = __version__
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_remove_toctrees',
    'numpydoc'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

exclude_patterns = ['_build', 'ex_03', 'ex_02', '../examples/data']

# numpydoc interacts with autosummary, that creates excessive warnings
# This line is a 'hack' for that interaction that stops the warnings
numpydoc_show_class_members = False

# generate autosummary even if no references
autosummary_generate = True

# The suffix(es) of source filenames. Can be str or list of string
source_suffix = ['.rst', '.md', '.txt']

# Autodoc classes
autodoc_default_options = {
    'members': None,
    'inherited-members': None,
}

# The master toctree document.
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Settings for sphinx_copybutton
copybutton_prompt_text = "$ "

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Theme options are theme-specific and customize the look and feel of a
# theme further.
html_theme_options = {
    # Navigation bar title. (Default: ``project`` value)
    'navbar_title': project,

    # Tab name for entire site. (Default: "Site")
    'navbar_site_name': "Site",

    # A list of tuples containting pages to link to.  The value should
    # be in the form [(name, page), ..]
    'navbar_links': [
        ('API', 'api'),
        ('Examples', 'auto_examples/index'),
        ('Reference', 'reference'),
    ],

    # Render the next and previous page links in navbar. (Default: true)
    'navbar_sidebarrel': False,

    # Render the current pages TOC in the navbar. (Default: true)
    'navbar_pagenav': False,

    # Global TOC depth for "site" navbar tab. (Default: 1)
    # Switching to -1 shows all levels.
    'globaltoc_depth': 0,

    # Tab name for the current pages TOC. (Default: "Page")
    'navbar_pagenav_name': "Page",

    # Include hidden TOCs in Site navbar?
    #
    # Note: If this is "false", you cannot have mixed ``:hidden:`` and
    # non-hidden ``toctree`` directives in the same page, or else the build
    # will break.
    #
    # Values: "true" (default) or "false"
    'globaltoc_includehidden': "true",

    # HTML navbar class (Default: "navbar") to attach to <div> element.
    # For black navbar, do "navbar navbar-inverse"
    'navbar_class': "navbar",

    # Fix navigation bar to top of page?
    # Values: "true" (default) or "false"
    'navbar_fixed_top': "true",

    # Location of link to source.
    # Options are "nav" (default), "footer" or anything else to exclude.
    'source_link_position': "None",

    # Bootswatch (http://bootswatch.com/) theme.
    #
    # Options are nothing with "" (default) or the name of a valid theme
    # such as "amelia" or "cosmo".
    #
    # Note that this is served off CDN, so won't be available offline.
    #'bootswatch_theme': "cerulean",

    # Choose Bootstrap version.
    # Values: "3" (default) or "2" (in quotes)
    'bootstrap_version': "3",
}

# -- Extension configuration -------------------------------------------------

# Configurations for sphinx gallery
sphinx_gallery_conf = {
    'examples_dirs': ['../examples'],
    'within_subsection_order': FileNameSortKey,
    'subsection_order': ExplicitOrder(
        [
            '../examples/ex_01_quickstart',
            '../examples/ex_02_advanced_options',
            '../examples/ex_03_constraints'
        ]
    ),
    'gallery_dirs': ['auto_examples'],
    'backreferences_dir': 'generated',
    'doc_module': ('riskslim',),
    'reference_url': {'riskslim': None},
    'remove_config_comments': True,
    'filename_pattern': '../examples/ex_',
    'ignore_pattern': 'training_script.py',
    "image_scrapers": ('matplotlib',)
}


html_static_path = ['_static']
html_css_files = ['custom.css']

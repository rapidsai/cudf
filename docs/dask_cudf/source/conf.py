# Copyright (c) 2018-2025, NVIDIA CORPORATION.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime

import dask_cudf
from packaging.version import Version

DASK_CUDF_VERSION = Version(dask_cudf.__version__)

project = "dask-cudf"
copyright = f"2018-{datetime.datetime.today().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"
version = f"{DASK_CUDF_VERSION.major:02}.{DASK_CUDF_VERSION.minor:02}"
release = f"{DASK_CUDF_VERSION.major:02}.{DASK_CUDF_VERSION.minor:02}.{DASK_CUDF_VERSION.micro:02}"

language = "en"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "numpydoc",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = []

copybutton_prompt_text = ">>> "

# Enable automatic generation of systematic, namespaced labels for sections
myst_heading_anchors = 2


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_logo = "_static/RAPIDS-logo-purple.png"
htmlhelp_basename = "dask-cudfdoc"
html_use_modindex = True

html_static_path = ["_static"]

pygments_style = "sphinx"

html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/rapidsai/cudf",
    "twitter_url": "https://twitter.com/rapidsai",
    "show_toc_level": 1,
    "navbar_align": "right",
    "navigation_with_keys": True,
}
include_pandas_compat = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "cudf": ("https://docs.rapids.ai/api/cudf/stable/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "dask-cuda": ("https://docs.rapids.ai/api/dask-cuda/stable/", None),
}

numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False


def setup(app):
    app.add_css_file("https://docs.rapids.ai/assets/css/custom.css")
    app.add_js_file(
        "https://docs.rapids.ai/assets/js/custom.js", loading_method="defer"
    )

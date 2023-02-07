.. dask-cudf documentation coordinating file, created by
   sphinx-quickstart on Mon Feb  6 18:48:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dask-cudf's documentation!
=====================================

Dask-cuDF is an extension library for the `Dask <https://dask.org>`__
parallel computing framework that provides a `cuDF
<https://docs.rapids.ai/api/cudf/stable/>`__-backed distributed
dataframe with the same API as `Dask dataframes
<https://docs.dask.org/en/stable/dataframe.html>`__.

If you are familiar with Dask and `pandas <pandas.pydata.org>`__ or
`cuDF <https://docs.rapids.ai/api/cudf/stable/>`__, then Dask-cuDF
should feel familiar to you. If not, we recommend starting with `10
minutes to Dask
<https://docs.dask.org/en/stable/10-minutes-to-dask.html>`__ followed
by `10 minutes to cuDF and Dask-cuDF
<https://docs.rapids.ai/api/cudf/stable/user_guide/10min.html>`__.

When running on multi-GPU systems, `Dask-CUDA
<https://docs.rapids.ai/api/dask-cuda/stable/>`__ is recommended to
simplify the setup of the cluster, taking advantage of all features of
the GPU and networking hardware.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

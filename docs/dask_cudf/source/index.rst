.. dask-cudf documentation coordinating file, created by
   sphinx-quickstart on Mon Feb  6 18:48:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dask-cudf's documentation!
=====================================

**Dask-cuDF** (pronounced "DASK KOO-dee-eff") is an extension
library for the `Dask <https://dask.org>`__ parallel computing
framework that provides a `cuDF
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

Using Dask-cuDF
---------------

When installed, Dask-cuDF registers itself as a dataframe backend for
Dask. This means that in many cases, using cuDF-backed dataframes requires
only small changes to an existing workflow. The minimal change is to
select cuDF as the dataframe backend in :doc:`Dask's
configuration <dask:configuration>`. To do so, we must set the option
``dataframe.backend`` to ``cudf``. From Python, this can be achieved
like so::

  import dask

  dask.config.set({"dataframe.backend": "cudf"})

Alternatively, you can set ``DASK_DATAFRAME__BACKEND=cudf`` in the
environment before running your code.

Dataframe creation from on-disk formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your workflow creates Dask dataframes from on-disk formats
(for example using :func:`dask.dataframe.read_parquet`), then setting
the backend may well be enough to migrate your workflow.

For example, consider reading a dataframe from parquet::

   import dask.dataframe as dd

   # By default, we obtain a pandas-backed dataframe
   df = dd.read_parquet("data.parquet", ...)


To obtain a cuDF-backed dataframe, we must set the
``dataframe.backend`` configuration option::

  import dask
  import dask.dataframe as dd

  dask.config.set({"dataframe.backend": "cudf"})
  # This gives us a cuDF-backed dataframe
  df = dd.read_parquet("data.parquet", ...)

This code will use cuDF's GPU-accelerated :func:`parquet reader
<cudf.read_parquet>` to read partitions of the data.

Dataframe creation from in-memory formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have a dataframe in memory and want to convert it to a
cuDF-backend one, there are two options depending on whether the
dataframe is already a Dask one or not. If you have a Dask dataframe,
then you can call :func:`dask.dataframe.to_backend` passing ``"cudf"``
as the backend; if you have a pandas dataframe then you can either
call :func:`dask.dataframe.from_pandas` followed by
:func:`~dask.dataframe.to_backend` or first convert the dataframe with
:func:`cudf.from_pandas` and then parallelise this with
:func:`dask_cudf.from_cudf`.

API Reference
-------------

Generally speaking, Dask-cuDF tries to offer exactly the same API as
Dask itself. There are, however, some minor differences mostly because
cuDF does not :doc:`perfectly mirror <cudf:user_guide/PandasCompat>`
the pandas API, or because cuDF provides additional configuration
flags (these mostly occur in data reading and writing interfaces).

As a result, straightforward workflows can be migrated without too
much trouble, but more complex ones that utilise more features may
need a bit of tweaking. The API documentation describes details of the
differences and all functionality that Dask-cuDF supports.

.. toctree::
   :maxdepth: 2

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

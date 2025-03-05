===============
 API reference
===============

As recommended in the introduction, ``dask_cudf`` should be used through the
`Dask DataFrame`_ API with the backend set to cudf.

.. code-block:: python

   >>> import dask
   >>> dask.config.set({"dataframe.backend": "cudf"})

With this configuration, ``dask_cudf`` will behave as documented in the Dask
DataFrame API, but with ``cudf`` objects used in place of ``pandas`` objects.

The rest of this page documents the API you might use from ``dask_cudf``
explicitly.

Creating and storing DataFrames
===============================

:doc:`Like Dask <dask:dataframe-create>`, Dask-cuDF supports creation
of DataFrames from a variety of storage formats. In addition to the methods
documented there, Dask-cuDF provides some cuDF-specific methods:

.. automodule:: dask_cudf
   :members:
      from_cudf,
      read_text

For on-disk data that are not supported directly in Dask or Dask-cuDF, we
recommend using Dask's data reading facilities, followed by calling
:meth:`dask.dataframe.DataFrame.to_backend` with ``"cudf"`` to obtain a
Dask-cuDF object.

.. _Dask DataFrame: https://docs.dask.org/en/stable/dataframe.html

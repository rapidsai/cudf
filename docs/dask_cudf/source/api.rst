===============
 API reference
===============

Dask cuDF implements the Dask-DataFrame API with with ``cudf`` objects used in
place of ``pandas`` objects. As recommended in the introduction, the best way to
use ``dask_cudf`` is to use the `Dask DataFrame`_ API with the backend set to
cudf.

.. code-block:: python

   >>> import dask
   >>> dask.config.set({"dataframe.backend": "cudf"})

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

For on-disk data that are not supported directly in Dask-cuDF, we recommend
using one of

- Dask's data reading facilities, followed by
  :meth:`dask.dataframe.DataFrame.to_backend` with ``"cudf"`` to obtain a
  Dask-cuDF object
- :func:`dask.dataframe.from_map`
- :func:`dask.dataframe.from_delayed`

.. _Dask DataFrame: https://docs.dask.org/en/stable/dataframe.html

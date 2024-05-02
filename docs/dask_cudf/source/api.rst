===============
 API reference
===============

This page provides a list of all publicly accessible modules, methods,
and classes in the ``dask_cudf`` namespace.


Creating and storing DataFrames
===============================

:doc:`Like Dask <dask:dataframe-create>`, Dask-cuDF supports creation
of DataFrames from a variety of storage formats. For on-disk data that
are not supported directly in Dask-cuDF, we recommend using Dask's
data reading facilities, followed by calling
:meth:`*.to_backend("cudf")` to obtain a Dask-cuDF object.

.. automodule:: dask_cudf
   :members:
      from_cudf,
      from_delayed,
      read_csv,
      read_json,
      read_orc,
      to_orc,
      read_text,
      read_parquet

Grouping
========

As discussed in the :doc:`Dask documentation for groupby
<dask:dataframe-groupby>`, ``groupby``, ``join``, and ``merge``, and
similar operations that require matching up rows of a DataFrame become
significantly more challenging in a parallel setting than they are in
serial. Dask-cuDF has the same challenges, however for certain groupby
operations, we can take advantage of functionality in cuDF that allows
us to compute multiple aggregations at once. There are therefore two
interfaces to grouping in Dask-cuDF, the general
:meth:`DataFrame.groupby` which returns a
:class:`.CudfDataFrameGroupBy` object, and a specialized
:func:`.groupby_agg`. Generally speaking, you should not need to call
:func:`.groupby_agg` directly, since Dask-cuDF will arrange to call it
if possible.

.. autoclass:: dask_cudf.groupby.CudfDataFrameGroupBy
   :members:
   :inherited-members:
   :show-inheritance:

.. autofunction:: dask_cudf.groupby_agg


DataFrames and Series
=====================

The core distributed objects provided by Dask-cuDF are the
:class:`.DataFrame` and :class:`.Series`. These inherit respectively
from :class:`dask.dataframe.DataFrame` and
:class:`dask.dataframe.Series`, and so the API is essentially
identical. The full API is provided below.

.. autoclass:: dask_cudf.DataFrame
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: dask_cudf.Series
   :members:
   :inherited-members:
   :show-inheritance:

.. automodule:: dask_cudf
   :members:
      concat

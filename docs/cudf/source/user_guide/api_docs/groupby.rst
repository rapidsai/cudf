.. _api.groupby:

=======
GroupBy
=======
.. currentmodule:: cudf.core.groupby

GroupBy objects are returned by groupby calls: :func:`cudf.DataFrame.groupby`, :func:`cudf.Series.groupby`, etc.

Indexing, iteration
-------------------
.. autosummary::
   :toctree: api/

   GroupBy.__iter__
   GroupBy.groups

.. currentmodule:: cudf

.. autosummary::
   :toctree: api/

   Grouper

.. currentmodule:: cudf.core.groupby.groupby

Function application
--------------------
.. autosummary::
   :toctree: api/

   GroupBy.apply
   GroupBy.agg
   SeriesGroupBy.aggregate
   DataFrameGroupBy.aggregate
   GroupBy.pipe
   GroupBy.transform

Computations / descriptive stats
--------------------------------
.. autosummary::
   :toctree: api/

   GroupBy.bfill
   GroupBy.count
   GroupBy.cumcount
   GroupBy.cummax
   GroupBy.cummin
   GroupBy.cumsum
   GroupBy.diff
   GroupBy.ffill
   GroupBy.first
   GroupBy.get_group
   GroupBy.groups
   GroupBy.idxmax
   GroupBy.idxmin
   GroupBy.last
   GroupBy.max
   GroupBy.mean
   GroupBy.median
   GroupBy.min
   GroupBy.ngroup
   GroupBy.nth
   GroupBy.nunique
   GroupBy.prod
   GroupBy.shift
   GroupBy.size
   GroupBy.std
   GroupBy.sum
   GroupBy.var
   GroupBy.corr
   GroupBy.cov

The following methods are available in both ``SeriesGroupBy`` and
``DataFrameGroupBy`` objects, but may differ slightly, usually in that
the ``DataFrameGroupBy`` version usually permits the specification of an
axis argument, and often an argument indicating whether to restrict
application to columns of a specific data type.

.. autosummary::
   :toctree: api/

   DataFrameGroupBy.bfill
   DataFrameGroupBy.count
   DataFrameGroupBy.cumcount
   DataFrameGroupBy.cummax
   DataFrameGroupBy.cummin
   DataFrameGroupBy.cumsum
   DataFrameGroupBy.describe
   DataFrameGroupBy.diff
   DataFrameGroupBy.ffill
   DataFrameGroupBy.fillna
   DataFrameGroupBy.idxmax
   DataFrameGroupBy.idxmin
   DataFrameGroupBy.nunique
   DataFrameGroupBy.quantile
   DataFrameGroupBy.shift
   DataFrameGroupBy.size

The following methods are available only for ``SeriesGroupBy`` objects.

.. autosummary::
   :toctree: api/

   SeriesGroupBy.nunique
   SeriesGroupBy.unique

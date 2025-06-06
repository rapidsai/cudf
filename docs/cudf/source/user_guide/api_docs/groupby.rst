.. _api.groupby:

=======
GroupBy
=======
.. currentmodule:: cudf.core.groupby

``DataFrameGroupBy`` and ``SeriesGroupBy`` instances are returned by groupby calls
:func:`cudf.DataFrame.groupby` and :func:`cudf.Series.groupby` respectively.


Indexing, iteration
-------------------
.. autosummary::
   :toctree: api/

   DataFrameGroupBy.__iter__
   SeriesGroupBy.__iter__
   DataFrameGroupBy.groups
   SeriesGroupBy.groups
   DataFrameGroupBy.indices
   SeriesGroupBy.indices
   DataFrameGroupBy.get_group
   SeriesGroupBy.get_group

Function application
--------------------
.. autosummary::
   :toctree: api/

   SeriesGroupBy.apply
   DataFrameGroupBy.apply
   SeriesGroupBy.agg
   DataFrameGroupBy.agg
   SeriesGroupBy.aggregate
   DataFrameGroupBy.aggregate
   SeriesGroupBy.transform
   DataFrameGroupBy.transform
   SeriesGroupBy.pipe
   DataFrameGroupBy.pipe
   DataFrameGroupBy.filter
   SeriesGroupBy.filter

``DataFrameGroupBy`` computations / descriptive stats
-----------------------------------------------------
.. autosummary::
   :toctree: api/

   DataFrameGroupBy.all
   DataFrameGroupBy.any
   DataFrameGroupBy.bfill
   DataFrameGroupBy.corr
   DataFrameGroupBy.count
   DataFrameGroupBy.cov
   DataFrameGroupBy.cumcount
   DataFrameGroupBy.cummax
   DataFrameGroupBy.cummin
   DataFrameGroupBy.cumprod
   DataFrameGroupBy.cumsum
   DataFrameGroupBy.describe
   DataFrameGroupBy.diff
   DataFrameGroupBy.ewm
   DataFrameGroupBy.expanding
   DataFrameGroupBy.ffill
   DataFrameGroupBy.first
   DataFrameGroupBy.head
   DataFrameGroupBy.idxmax
   DataFrameGroupBy.idxmin
   DataFrameGroupBy.last
   DataFrameGroupBy.max
   DataFrameGroupBy.mean
   DataFrameGroupBy.median
   DataFrameGroupBy.min
   DataFrameGroupBy.ngroup
   DataFrameGroupBy.nth
   DataFrameGroupBy.nunique
   DataFrameGroupBy.ohlc
   DataFrameGroupBy.pct_change
   DataFrameGroupBy.prod
   DataFrameGroupBy.quantile
   DataFrameGroupBy.rank
   DataFrameGroupBy.resample
   DataFrameGroupBy.rolling
   DataFrameGroupBy.sample
   DataFrameGroupBy.shift
   DataFrameGroupBy.size
   DataFrameGroupBy.std
   DataFrameGroupBy.sum
   DataFrameGroupBy.var
   DataFrameGroupBy.tail
   DataFrameGroupBy.take
   DataFrameGroupBy.value_counts

``SeriesGroupBy`` computations / descriptive stats
--------------------------------------------------
.. autosummary::
   :toctree: api/

   SeriesGroupBy.all
   SeriesGroupBy.any
   SeriesGroupBy.bfill
   SeriesGroupBy.corr
   SeriesGroupBy.count
   SeriesGroupBy.cov
   SeriesGroupBy.cumcount
   SeriesGroupBy.cummax
   SeriesGroupBy.cummin
   SeriesGroupBy.cumprod
   SeriesGroupBy.cumsum
   SeriesGroupBy.describe
   SeriesGroupBy.diff
   SeriesGroupBy.ewm
   SeriesGroupBy.expanding
   SeriesGroupBy.ffill
   SeriesGroupBy.first
   SeriesGroupBy.head
   SeriesGroupBy.last
   SeriesGroupBy.idxmax
   SeriesGroupBy.idxmin
   SeriesGroupBy.is_monotonic_increasing
   SeriesGroupBy.is_monotonic_decreasing
   SeriesGroupBy.max
   SeriesGroupBy.mean
   SeriesGroupBy.median
   SeriesGroupBy.min
   SeriesGroupBy.ngroup
   SeriesGroupBy.nlargest
   SeriesGroupBy.nsmallest
   SeriesGroupBy.nth
   SeriesGroupBy.nunique
   SeriesGroupBy.unique
   SeriesGroupBy.ohlc
   SeriesGroupBy.pct_change
   SeriesGroupBy.prod
   SeriesGroupBy.quantile
   SeriesGroupBy.rank
   SeriesGroupBy.resample
   SeriesGroupBy.rolling
   SeriesGroupBy.sample
   SeriesGroupBy.shift
   SeriesGroupBy.size
   SeriesGroupBy.std
   SeriesGroupBy.sum
   SeriesGroupBy.var
   SeriesGroupBy.tail
   SeriesGroupBy.take
   SeriesGroupBy.value_counts

Plotting and visualization
--------------------------
.. autosummary::
   :toctree: api/

   DataFrameGroupBy.boxplot
   DataFrameGroupBy.hist
   SeriesGroupBy.hist
   DataFrameGroupBy.plot
   SeriesGroupBy.plot

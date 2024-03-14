=================
General Functions
=================
.. currentmodule:: cudf

Data manipulations
------------------

.. autosummary::
   :toctree: api/

   cudf.concat
   cudf.crosstab
   cudf.cut
   cudf.factorize
   cudf.get_dummies
   cudf.melt
   cudf.merge
   cudf.pivot
   cudf.pivot_table
   cudf.unstack

Top-level conversions
---------------------
.. autosummary::
   :toctree: api/

    cudf.to_numeric
    cudf.from_dataframe
    cudf.from_dlpack
    cudf.from_pandas

Top-level dealing with datetimelike data
----------------------------------------

.. autosummary::
   :toctree: api/

    cudf.to_datetime
    cudf.date_range

Top-level dealing with Interval data
------------------------------------

.. autosummary::
   :toctree: api/

    cudf.interval_range

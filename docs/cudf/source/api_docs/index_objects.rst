=============
Index objects
=============

Index
-----
.. currentmodule:: cudf

**Many of these methods or variants thereof are available on the objects
that contain an index (Series/DataFrame) and those should most likely be
used before calling these methods directly.**

.. autosummary::
   :toctree: api/
   :template: autosummary/class_with_autosummary.rst

   Index

Properties
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.empty
   Index.has_duplicates
   Index.duplicated
   Index.hasnans
   Index.is_monotonic
   Index.is_monotonic_increasing
   Index.is_monotonic_decreasing
   Index.is_unique
   Index.name
   Index.names
   Index.ndim
   Index.nlevels
   Index.shape
   Index.size
   Index.values

Modifying and computations
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.any
   Index.copy
   Index.drop_duplicates
   Index.equals
   Index.factorize
   Index.is_boolean
   Index.is_categorical
   Index.is_floating
   Index.is_integer
   Index.is_interval
   Index.is_mixed
   Index.is_numeric
   Index.is_object
   Index.min
   Index.max
   Index.rename
   Index.repeat
   Index.where
   Index.take
   Index.unique

Compatibility with MultiIndex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.set_names

Missing values
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.fillna
   Index.dropna
   Index.isna
   Index.notna

Memory usage
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.memory_usage

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.astype
   Index.to_arrow
   Index.to_cupy
   Index.to_list
   Index.to_numpy
   Index.to_series
   Index.to_frame
   Index.to_pandas
   Index.to_dlpack
   Index.from_pandas
   Index.from_arrow

Sorting
~~~~~~~
.. autosummary::
   :toctree: api/

   Index.argsort
   Index.searchsorted
   Index.sort_values

Time-specific operations
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.shift

Combining / joining / set operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.append
   Index.union
   Index.intersection
   Index.join
   Index.difference

Selecting
~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.get_level_values
   Index.get_loc
   Index.get_slice_bound
   Index.isin

.. _api.numericindex:

Numeric Index
-------------
.. autosummary::
   :toctree: api/
   :template: autosummary/class_without_autosummary.rst

   RangeIndex
   Int64Index
   UInt64Index
   Float64Index

.. _api.categoricalindex:

CategoricalIndex
----------------
.. autosummary::
   :toctree: api/
   :template: autosummary/class_without_autosummary.rst

   CategoricalIndex

Categorical components
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CategoricalIndex.codes
   CategoricalIndex.categories

Modifying and computations
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CategoricalIndex.equals

.. _api.intervalindex:

IntervalIndex
-------------
.. autosummary::
   :toctree: api/

   IntervalIndex

IntervalIndex components
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   IntervalIndex.from_breaks
   IntervalIndex.values
   IntervalIndex.get_loc

.. _api.multiindex:

MultiIndex
----------
.. autosummary::
   :toctree: api/
   :template: autosummary/class_without_autosummary.rst

   MultiIndex

MultiIndex constructors
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.from_tuples
   MultiIndex.from_product
   MultiIndex.from_frame
   MultiIndex.from_arrow

MultiIndex properties
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.names
   MultiIndex.levels
   MultiIndex.codes
   MultiIndex.nlevels

MultiIndex components
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.to_frame
   MultiIndex.droplevel

MultiIndex selecting
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.get_loc
   MultiIndex.get_level_values

.. _api.datetimeindex:

DatetimeIndex
-------------
.. autosummary::
   :toctree: api/
   :template: autosummary/class_without_autosummary.rst

   DatetimeIndex

Time/date components
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DatetimeIndex.year
   DatetimeIndex.month
   DatetimeIndex.day
   DatetimeIndex.hour
   DatetimeIndex.minute
   DatetimeIndex.second
   DatetimeIndex.microsecond
   DatetimeIndex.nanosecond
   DatetimeIndex.day_of_year
   DatetimeIndex.dayofyear
   DatetimeIndex.dayofweek
   DatetimeIndex.weekday
   DatetimeIndex.quarter
   DatetimeIndex.is_leap_year

   DatetimeIndex.isocalendar

Time-specific operations
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DatetimeIndex.round
   DatetimeIndex.ceil
   DatetimeIndex.floor

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DatetimeIndex.to_series
   DatetimeIndex.to_frame

TimedeltaIndex
--------------
.. autosummary::
   :toctree: api/
   :template: autosummary/class_without_autosummary.rst

   TimedeltaIndex

Components
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   TimedeltaIndex.days
   TimedeltaIndex.seconds
   TimedeltaIndex.microseconds
   TimedeltaIndex.nanoseconds
   TimedeltaIndex.components
   TimedeltaIndex.inferred_freq

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   TimedeltaIndex.to_series
   TimedeltaIndex.to_frame

======
Series
======
.. currentmodule:: cudf

Constructor
-----------
.. autosummary::
   :toctree: api/

   Series

Attributes
----------
**Axes**

.. autosummary::
   :toctree: api/

   Series.axes
   Series.index
   Series.values
   Series.data
   Series.dtype
   Series.dtypes
   Series.shape
   Series.ndim
   Series.nullable
   Series.nullmask
   Series.null_count
   Series.size
   Series.T
   Series.memory_usage
   Series.hasnans
   Series.has_nulls
   Series.empty
   Series.name
   Series.valid_count
   Series.values_host

Conversion
----------
.. autosummary::
   :toctree: api/

   Series.astype
   Series.convert_dtypes
   Series.copy
   Series.deserialize
   Series.device_deserialize
   Series.device_serialize
   Series.host_deserialize
   Series.host_serialize
   Series.serialize
   Series.to_list
   Series.tolist
   Series.__array__
   Series.scale


Indexing, iteration
-------------------
.. autosummary::
   :toctree: api/

   Series.loc
   Series.iloc
   Series.__iter__
   Series.items
   Series.iteritems
   Series.keys
   Series.squeeze

Binary operator functions
-------------------------
.. autosummary::
   :toctree: api/

   Series.add
   Series.sub
   Series.subtract
   Series.mul
   Series.multiply
   Series.truediv
   Series.div
   Series.divide
   Series.floordiv
   Series.mod
   Series.pow
   Series.radd
   Series.rsub
   Series.rmul
   Series.rdiv
   Series.rtruediv
   Series.rfloordiv
   Series.rmod
   Series.rpow
   Series.round
   Series.lt
   Series.gt
   Series.le
   Series.ge
   Series.ne
   Series.eq
   Series.product
   Series.dot

Function application, GroupBy & window
--------------------------------------
.. autosummary::
   :toctree: api/

   Series.apply
   Series.map
   Series.groupby
   Series.rolling
   Series.pipe

.. _api.series.stats:

Computations / descriptive stats
--------------------------------
.. autosummary::
   :toctree: api/

   Series.abs
   Series.all
   Series.any
   Series.autocorr
   Series.between
   Series.clip
   Series.corr
   Series.count
   Series.cov
   Series.cummax
   Series.cummin
   Series.cumprod
   Series.cumsum
   Series.describe
   Series.diff
   Series.digitize
   Series.ewm
   Series.factorize
   Series.kurt
   Series.max
   Series.mean
   Series.median
   Series.min
   Series.mode
   Series.nlargest
   Series.nsmallest
   Series.pct_change
   Series.prod
   Series.quantile
   Series.rank
   Series.skew
   Series.std
   Series.sum
   Series.var
   Series.kurtosis
   Series.unique
   Series.nunique
   Series.is_unique
   Series.is_monotonic_increasing
   Series.is_monotonic_decreasing
   Series.value_counts

Reindexing / selection / label manipulation
-------------------------------------------
.. autosummary::
   :toctree: api/

   Series.add_prefix
   Series.add_suffix
   Series.drop
   Series.drop_duplicates
   Series.duplicated
   Series.equals
   Series.first
   Series.head
   Series.isin
   Series.last
   Series.reindex
   Series.rename
   Series.reset_index
   Series.sample
   Series.take
   Series.tail
   Series.tile
   Series.truncate
   Series.where
   Series.mask

Missing data handling
---------------------
.. autosummary::
   :toctree: api/

   Series.backfill
   Series.bfill
   Series.dropna
   Series.ffill
   Series.fillna
   Series.interpolate
   Series.isna
   Series.isnull
   Series.nans_to_nulls
   Series.notna
   Series.notnull
   Series.pad
   Series.replace

Reshaping, sorting
------------------
.. autosummary::
   :toctree: api/

   Series.argsort
   Series.sort_values
   Series.sort_index
   Series.explode
   Series.searchsorted
   Series.repeat
   Series.transpose

Combining / comparing / joining / merging
-----------------------------------------
.. autosummary::
   :toctree: api/

   Series.update

Time Series-related
-------------------
.. autosummary::
   :toctree: api/

   Series.shift
   Series.resample

Accessors
---------

pandas provides dtype-specific methods under various accessors.
These are separate namespaces within :class:`Series` that only apply
to specific data types.

=========================== =================================
Data Type                   Accessor
=========================== =================================
Datetime, Timedelta         :ref:`dt <api.series.dt>`
String                      :ref:`str <api.series.str>`
Categorical                 :ref:`cat <api.series.cat>`
List                        :ref:`list <api.series.list>`
Struct                      :ref:`struct <api.series.struct>`
=========================== =================================

.. _api.series.dt:

Datetimelike properties
~~~~~~~~~~~~~~~~~~~~~~~

``Series.dt`` can be used to access the values of the series as
datetimelike and return several properties.
These can be accessed like ``Series.dt.<property>``.

.. currentmodule:: cudf
.. autosummary::
   :toctree: api/

   Series.dt

Datetime properties
^^^^^^^^^^^^^^^^^^^
.. currentmodule:: cudf.core.series.DatetimeProperties

.. autosummary::
   :toctree: api/

   year
   month
   day
   hour
   minute
   second
   microsecond
   nanosecond
   dayofweek
   weekday
   dayofyear
   day_of_year
   quarter
   is_month_start
   is_month_end
   is_quarter_start
   is_quarter_end
   is_year_start
   is_year_end
   is_leap_year
   days_in_month

Datetime methods
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: api/

   isocalendar
   strftime
   round
   floor
   ceil
   tz_localize


Timedelta properties
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: cudf.core.series.TimedeltaProperties
.. autosummary::
   :toctree: api/

   days
   seconds
   microseconds
   nanoseconds
   components

.. _api.series.str:
.. include:: string_handling.rst


.. _api.series.cat:

Categorical accessor
~~~~~~~~~~~~~~~~~~~~

Categorical-dtype specific methods and attributes are available under
the ``Series.cat`` accessor.

.. currentmodule:: cudf
.. autosummary::
   :toctree: api/

   Series.cat

.. currentmodule:: cudf.core.column.categorical.CategoricalAccessor
.. autosummary::
   :toctree: api/

   categories
   ordered
   codes
   reorder_categories
   add_categories
   remove_categories
   set_categories
   as_ordered
   as_unordered


.. _api.series.list:
.. include:: list_handling.rst


.. _api.series.struct:
.. include:: struct_handling.rst


..
    The following is needed to ensure the generated pages are created with the
    correct template (otherwise they would be created in the Series/Index class page)

..
    .. currentmodule:: cudf
    .. autosummary::
       :toctree: api/
       :template: autosummary/accessor.rst

       Series.str
       Series.cat
       Series.dt
       Index.str


Serialization / IO / conversion
-------------------------------
.. currentmodule:: cudf
.. autosummary::
   :toctree: api/

   Series.to_arrow
   Series.to_cupy
   Series.to_dict
   Series.to_dlpack
   Series.to_frame
   Series.to_hdf
   Series.to_json
   Series.to_numpy
   Series.to_pandas
   Series.to_string
   Series.from_arrow
   Series.from_categorical
   Series.from_masked_array
   Series.from_pandas
   Series.hash_values

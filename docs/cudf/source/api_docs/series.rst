.. meta::
   :my-var: a-for-apple

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

   Series.index
   Series.values
   Series.dtype
   Series.shape
   Series.ndim
   Series.size
   Series.memory_usage
   Series.has_nulls
   Series.empty
   Series.name

Conversion
----------
.. autosummary::
   :toctree: api/

   Series.astype
   Series.copy
   Series.to_list
   Series.__array__

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

For more information on ``.at``, ``.iat``, ``.loc``, and
``.iloc``,  see the :ref:`indexing documentation <indexing>`.

Binary operator functions
-------------------------
.. autosummary::
   :toctree: api/

   Series.add
   Series.sub
   Series.mul
   Series.truediv
   Series.floordiv
   Series.mod
   Series.pow
   Series.radd
   Series.rsub
   Series.rmul
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

Function application, GroupBy & window
--------------------------------------
.. autosummary::
   :toctree: api/

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
   Series.factorize
   Series.kurt
   Series.max
   Series.mean
   Series.median
   Series.min
   Series.mode
   Series.nlargest
   Series.nsmallest
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
   Series.is_monotonic
   Series.is_monotonic_increasing
   Series.is_monotonic_decreasing
   Series.value_counts

Reindexing / selection / label manipulation
-------------------------------------------
.. autosummary::
   :toctree: api/

   Series.drop
   Series.drop_duplicates
   Series.equals
   Series.head
   Series.isin
   Series.reindex
   Series.rename
   Series.reset_index
   Series.sample
   Series.take
   Series.tail
   Series.truncate
   Series.where
   Series.mask

Missing data handling
---------------------
.. autosummary::
   :toctree: api/

   Series.dropna
   Series.fillna
   Series.isna
   Series.isnull
   Series.notna
   Series.notnull
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

Combining / comparing / joining / merging
-----------------------------------------
.. autosummary::
   :toctree: api/

   Series.append
   Series.update

Time Series-related
-------------------
.. autosummary::
   :toctree: api/

   Series.shift

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
Sparse                      :ref:`sparse <api.series.sparse>`
=========================== =================================

.. _api.series.dt:

Datetimelike properties
~~~~~~~~~~~~~~~~~~~~~~~

``Series.dt`` can be used to access the values of the series as
datetimelike and return several properties.
These can be accessed like ``Series.dt.<property>``.

Datetime properties
^^^^^^^^^^^^^^^^^^^
.. currentmodule:: cudf.core.series.DatetimeProperties

.. autosummary::
   :toctree: api/

   day
   dayofweek
   hour
   minute
   month
   second
   weekday
   year

Datetime methods
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: api/

   strftime


Timedelta properties
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: cudf.core.series.TimedeltaProperties
.. autosummary::
   :toctree: api/

   components
   days
   microseconds
   nanoseconds
   seconds


.. _api.series.str:

String handling
~~~~~~~~~~~~~~~

``Series.str`` can be used to access the values of the series as
strings and apply several methods to it. These can be accessed like
``Series.str.<function/property>``.

.. currentmodule:: cudf.core.column.string.StringMethods
.. autosummary::
   :toctree: api/

   byte_count
   capitalize
   cat
   center
   character_ngrams
   character_tokenize
   code_points
   contains
   count
   detokenize
   edit_distance
   endswith
   extract
   filter_alphanum
   filter_characters
   filter_tokens
   find
   findall
   get
   get_json_object
   htoi
   index
   insert
   ip2int
   is_consonant
   is_vowel
   isalnum
   isalpha
   isdecimal
   isdigit
   isempty
   isfloat
   ishex
   isinteger
   isipv4
   isspace
   islower
   isnumeric
   isupper
   istimestamp
   join
   len
   ljust
   lower
   lstrip
   match
   ngrams
   ngrams_tokenize
   normalize_characters
   pad
   partition
   porter_stemmer_measure
   replace
   replace_tokens
   replace_with_backrefs
   rfind
   rindex
   rjust
   rpartition
   rstrip
   slice
   slice_from
   slice_replace
   split
   rsplit
   startswith
   strip
   subword_tokenize
   swapcase
   title
   token_count
   tokenize
   translate
   upper
   url_decode
   url_encode
   wrap
   zfill
   


..
    The following is needed to ensure the generated pages are created with the
    correct template (otherwise they would be created in the Series/Index class page)

..
    .. autosummary::
       :toctree: api/
       :template: autosummary/accessor.rst

       Series.str
       Series.cat
       Series.dt
       Series.sparse
       DataFrame.sparse
       Index.str

.. _api.series.cat:

Categorical accessor
~~~~~~~~~~~~~~~~~~~~

Categorical-dtype specific methods and attributes are available under
the ``Series.cat`` accessor.

.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_attribute.rst

   Series.cat.categories
   Series.cat.ordered
   Series.cat.codes

.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_method.rst

   Series.cat.rename_categories
   Series.cat.reorder_categories
   Series.cat.add_categories
   Series.cat.remove_categories
   Series.cat.remove_unused_categories
   Series.cat.set_categories
   Series.cat.as_ordered
   Series.cat.as_unordered


.. _api.series.sparse:

Sparse accessor
~~~~~~~~~~~~~~~

Sparse-dtype specific methods and attributes are provided under the
``Series.sparse`` accessor.

.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_attribute.rst

   Series.sparse.npoints
   Series.sparse.density
   Series.sparse.fill_value
   Series.sparse.sp_values

.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_method.rst

   Series.sparse.from_coo
   Series.sparse.to_coo

.. _api.series.flags:

Flags
~~~~~

Flags refer to attributes of the pandas object. Properties of the dataset (like
the date is was recorded, the URL it was accessed from, etc.) should be stored
in :attr:`Series.attrs`.

.. autosummary::
   :toctree: api/

   Flags

.. _api.series.metadata:

Metadata
~~~~~~~~

:attr:`Series.attrs` is a dictionary for storing global metadata for this Series.

.. warning:: ``Series.attrs`` is considered experimental and may change without warning.

.. autosummary::
   :toctree: api/

   Series.attrs


Plotting
--------
``Series.plot`` is both a callable method and a namespace attribute for
specific plotting methods of the form ``Series.plot.<kind>``.

.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_callable.rst

   Series.plot

.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_method.rst

   Series.plot.area
   Series.plot.bar
   Series.plot.barh
   Series.plot.box
   Series.plot.density
   Series.plot.hist
   Series.plot.kde
   Series.plot.line
   Series.plot.pie

.. autosummary::
   :toctree: api/

   Series.hist

Serialization / IO / conversion
-------------------------------
.. autosummary::
   :toctree: api/

   Series.to_pickle
   Series.to_csv
   Series.to_dict
   Series.to_excel
   Series.to_frame
   Series.to_xarray
   Series.to_hdf
   Series.to_sql
   Series.to_json
   Series.to_string
   Series.to_clipboard
   Series.to_latex
   Series.to_markdown

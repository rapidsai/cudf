API Reference
=============

.. currentmodule:: cudf.core.dataframe

DataFrame
---------
.. autoclass:: DataFrame
    :members:
    :exclude-members: serialize, deserialize

..
  For cudf.concat function
..
.. automodule:: cudf.core.reshape
    :members:

Series
------
.. currentmodule:: cudf.core.series

.. autoclass:: Series
    :members:
    :exclude-members: serialize, deserialize, logical_not, logical_or, logical_and, remainder, sum_of_squares, fill, merge

Strings
-------
.. currentmodule:: cudf.core.column.string

.. autoclass:: StringMethods
    :members:

Index
-----
.. currentmodule:: cudf.core.index
.. autoclass:: Index
    :members:
    :exclude-members: serialize, deserialize, is_monotonic, is_monotonic_increasing, is_monotonic_decreasing

RangeIndex
----------
.. currentmodule:: cudf.core.index
.. autoclass:: RangeIndex
    :members:
    :exclude-members: deserialize, serialize

GenericIndex
------------
.. currentmodule:: cudf.core.index
.. autoclass:: GenericIndex
    :members:

CategoricalIndex
----------------
.. currentmodule:: cudf.core.index
.. autoclass:: CategoricalIndex
    :members:

StringIndex
-----------
.. currentmodule:: cudf.core.index
.. autoclass:: StringIndex
    :members:

DatetimeIndex
-------------
.. currentmodule:: cudf.core.index
.. autoclass:: DatetimeIndex
    :members:

Categories
----------
.. currentmodule:: cudf.core.column.categorical

.. autoclass:: CategoricalAccessor
    :members:

GroupBy
-------
.. currentmodule:: cudf.core.groupby.groupby

.. autoclass:: GroupBy
    :members:
    :exclude-members: deserialize, serialize



IO
--
.. currentmodule:: cudf.io

.. automodule:: cudf.io.csv
    :members:
.. automodule:: cudf.io.parquet
    :members:
.. automodule:: cudf.io.orc
    :members:
.. automodule:: cudf.io.json
    :members:
.. automodule:: cudf.io.avro
    :members:
.. automodule:: cudf.io.dlpack
    :members:
.. automodule:: cudf.io.feather
    :members:
.. automodule:: cudf.io.hdf
    :members:

GpuArrowReader
--------------
.. currentmodule:: cudf.comm.gpuarrow
.. autoclass:: GpuArrowReader
    :members:

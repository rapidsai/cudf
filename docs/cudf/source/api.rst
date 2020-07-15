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

Int8Index
---------
.. currentmodule:: cudf.core.index
.. autoclass:: Int8Index
    :members:

Int16Index
---------
.. currentmodule:: cudf.core.index
.. autoclass:: Int16Index
    :members:

Int32Index
---------
.. currentmodule:: cudf.core.index
.. autoclass:: Int32Index
    :members:

Int64Index
---------
.. currentmodule:: cudf.core.index
.. autoclass:: Int64Index
    :members:

UInt8Index
---------
.. currentmodule:: cudf.core.index
.. autoclass:: UInt8Index
    :members:

UInt16Index
---------
.. currentmodule:: cudf.core.index
.. autoclass:: UInt16Index
    :members:

UInt32Index
---------
.. currentmodule:: cudf.core.index
.. autoclass:: UInt32Index
    :members:

UInt64Index
---------
.. currentmodule:: cudf.core.index
.. autoclass:: UInt64Index
    :members:

Float32Index
---------
.. currentmodule:: cudf.core.index
.. autoclass:: Float32Index
    :members:

Float64Index
---------
.. currentmodule:: cudf.core.index
.. autoclass:: Float64Index
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
